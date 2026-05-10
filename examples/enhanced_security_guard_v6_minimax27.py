# Enhanced InsightFace Security Guard System - Refined V6
#
# CHANGELOG from V5 (Identity Verification & Deduplication Pipeline):
#   [ADD] Strict identity verification threshold with voting mechanism
#   [ADD] TrackID-based spatial/temporal deduplication filter
#   [ADD] Temporal smoothing for state consistency (confirmation buffer)
#   [ADD] Dynamic timestamp-based directory creation (YYYYMMDD_HHMMSS)
#   [ENHANCE] Unified decision-making flow between FR and classification
#   [ENHANCE] Prevent identity conflict (Known vs Unknown flickering)
#   [ENHANCE] Cross-frame deduplication via tracking continuity
#
# Version: 6.0.0

"""
Enhanced InsightFace Security Guard System — V6

IDENTITY VERIFICATION & DEDUPLICATION PIPELINE IMPROVEMENTS (V6):
1. Strict Identity Verification Threshold
   - Configurable verification threshold (default: 0.85)
   - Voting mechanism: KNOWN classification suppresses UNKNOWN when confidence >= threshold
   - Prevents same person being classified as both Known and Unknown

2. TrackID-Based Spatial/Temporal Deduplication
   - Integrates YOLO tracking ID with deduplication system
   - Frame-skip mechanism (configurable window, default: 30 frames)
   - Temporal window (configurable, default: 5 seconds)
   - Prevents multiple screenshots of same individual within short time windows

3. Temporal Smoothing for State Consistency
   - Per-person rolling history buffer (default: 5 frames)
   - Minimum confirmation frames before state change (default: 3)
   - Hysteresis to prevent rapid state flickering
   - State transitions only after consistent evidence

4. Dynamic Timestamp-Based Directory Creation
   - Auto-generate timestamp folder: YYYYMMDD_HHMMSS format
   - Create known/ and unknown/ subdirectories within timestamp folder
   - Graceful handling of existing paths
   - Session isolation for captured faces
"""

# ============================================================================
# IMPORTS (consolidated, no duplicates)
# ============================================================================

import cv2
import numpy as np
import pygame
import os
import time
import json
import logging
import enum
import warnings
import uuid
import psutil
import threading
import hashlib
from pathlib import Path
from collections import OrderedDict, deque
from types import MappingProxyType
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from threading import Lock
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, Future
from functools import wraps
import math
from datetime import datetime

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Install with: pip install faiss-cpu")

# [FIX] Suppress CUDA initialization warnings (NVIDIA driver version warning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

from insightface.app import FaceAnalysis
from ultralytics import solutions
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "SecurityGuardConfig",
    "EnhancedSecurityGuard",
    "create_security_guard",
    "PersonRegistry",
    "FaceRecognizer",
    "InsightFaceDetector",
    "FaceScreenshotCapturer",
    "CircuitBreaker",
    "SecurityGuardMetrics",
    "MetricsExporter",
    "FaceDeduplicator",
    "IdentityStateManager",
    "TrackIdDeduplicator",
]

# ============================================================================
# CONSTANTS
# ============================================================================

"""
QUALITY ASSESSMENT WEIGHTS
===========================
These four weights control the composite face quality scoring algorithm.
"""

SIZE_WEIGHT = 0.30
BRIGHTNESS_WEIGHT = 0.25
SHARPNESS_WEIGHT = 0.25
CONTRAST_WEIGHT = 0.20

"""
QUALITY AND DETECTION THRESHOLDS
=================================
"""

DEFAULT_MIN_FACE_QUALITY = 20.0
DEFAULT_MIN_FACE_SIZE = 40
DEFAULT_IOU_THRESHOLD = 0.8
DEFAULT_FACE_TRACK_MAX_AGE = 30

"""
NUMERICAL STABILITY AND CACHING
================================
"""

NORMALIZATION_EPS = 1e-5
DEFAULT_CACHE_TTL = 30.0

"""
EMBEDDING DIMENSIONS
====================
InsightFace BuffaloL produces 512-dimensional embeddings
"""
INSIGHTFACE_EMBEDDING_DIM = 512

"""
[V6 NEW] IDENTITY VERIFICATION CONSTANTS
=========================================
"""
# Verification threshold for KNOWN classification
DEFAULT_IDENTITY_VERIFICATION_THRESHOLD = 0.85  # Cosine similarity threshold for KNOWN status
DEFAULT_MIN_CONFIRMATION_FRAMES = 3  # Frames required before state change
DEFAULT_STATE_HISTORY_SIZE = 5  # Rolling window for state history

# Frame-skip mechanism for deduplication
DEFAULT_FRAME_SKIP_WINDOW = 30  # Frames to skip between saves for same track ID
DEFAULT_TEMPORAL_DEDUP_WINDOW = 5.0  # Seconds between saves for same person

# Perceptual hash settings
DEFAULT_PHASH_SIZE = 16  # Higher = more detail, but more compute
DEFAULT_PHASH_THRESHOLD = 8  # Hamming distance threshold (lower = stricter)
DEFAULT_DCT_PHASH_THRESHOLD = 15  # DCT hash threshold

# Embedding-based deduplication
DEFAULT_EMBEDDING_SIM_THRESHOLD = 0.85  # Cosine similarity for cross-session dedup
DEFAULT_EMBEDDING_LOOKBACK_DAYS = 7  # How far back to check embeddings

# Temporal settings
DEFAULT_UNKNOWN_SAVE_INTERVAL = 30.0  # Minimum seconds between saves for same unknown
DEFAULT_MAX_SAVES_PER_UNKNOWN = 3  # Max saves per unknown ID per session
DEFAULT_DEDUP_STATE_TTL = 86400 * 7  # 7 days TTL for deduplication state

# Scene change detection
DEFAULT_SCENE_CHANGE_THRESHOLD = 0.3  # Frame difference ratio for scene change

# Quality threshold for saving
DEFAULT_SAVE_QUALITY_THRESHOLD = 40.0  # Minimum quality score to trigger save


# ============================================================================
# EMOTION / GENDER MAPPINGS
# ============================================================================

EMOTION_MAP: Dict[int, str] = MappingProxyType(
    {0: "happy", 1: "neutral", 2: "sad", 3: "angry", 4: "surprise", 5: "fear", 6: "disgust"}
)
GENDER_MAP: Dict[int, str] = MappingProxyType({0: "male", 1: "female"})
EMOTION_TO_INDEX: Dict[str, int] = MappingProxyType({v: k for k, v in EMOTION_MAP.items()})
GENDER_TO_INDEX: Dict[str, int] = MappingProxyType({v: k for k, v in GENDER_MAP.items()})
EMOTION_VALID_INDICES: Tuple[int, int] = (0, 6)
GENDER_VALID_INDICES: Tuple[int, int] = (0, 1)


def get_emotion_label(index: int) -> str:
    if not EMOTION_VALID_INDICES[0] <= index <= EMOTION_VALID_INDICES[1]:
        raise ValueError(f"Invalid emotion index {index}. Must be in range {EMOTION_VALID_INDICES}")
    return EMOTION_MAP[index]


def get_gender_label(index: int) -> str:
    if not GENDER_VALID_INDICES[0] <= index <= GENDER_VALID_INDICES[1]:
        raise ValueError(f"Invalid gender index {index}. Must be in range {EMOTION_VALID_INDICES}")
    return GENDER_MAP[index]


def get_emotion_index(label: str) -> Optional[int]:
    return EMOTION_TO_INDEX.get(label.lower())


def get_gender_index(label: str) -> Optional[int]:
    return GENDER_TO_INDEX.get(label.lower())


# ============================================================================
# LOGGING SETUP WITH CORRELATION ID
# ============================================================================


def _setup_logging(name: str = "SecurityGuard") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _get_base_dir() -> Path:
    try:
        return Path(__file__).parent.parent
    except NameError:
        return Path.cwd()


def _generate_timestamp_dir(base_dir: str) -> str:
    """
    [V6 NEW] Generate timestamp-based directory path.
    Creates: base_dir/YYYYMMDD_HHMMSS/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, timestamp)


def logged_operation(operation_name: str):
    """
    Decorator for tracing operations across the pipeline with correlation IDs.
    [V6 ENHANCE] Extended logging for deduplication operations.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            correlation_id = str(uuid.uuid4())[:8]
            start_time = time.time()

            logger = getattr(self, "_logger", None) or _setup_logging()
            logger.debug(f"[{correlation_id}] Starting {operation_name}")
            try:
                result = func(self, *args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"[{correlation_id}] Completed {operation_name} in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"[{correlation_id}] Failed {operation_name} after {duration:.3f}s: {e}")
                raise

        return wrapper

    return decorator


# ============================================================================
# CIRCUIT BREAKER - Fault Tolerance Pattern
# [V6] Minor enhancements for deduplication stability
# ============================================================================


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is in open state"""

    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests blocked
    - HALF_OPEN: Recovery timeout elapsed, testing with limited requests
    """

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0, success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = self.STATE_CLOSED
        self._lock = Lock()

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function through circuit breaker.
        Raises CircuitBreakerOpen if breaker is open.
        """
        with self._lock:
            if self.state == self.STATE_OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = self.STATE_HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit breaker is open. Retry after "
                        f"{self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
                    )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def _record_success(self):
        with self._lock:
            if self.state == self.STATE_HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = self.STATE_CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == self.STATE_CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    def _record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = self.STATE_OPEN

    def get_state(self) -> str:
        with self._lock:
            return self.state

    def reset(self):
        with self._lock:
            self.state = self.STATE_CLOSED
            self.failure_count = 0
            self.success_count = 0


# ============================================================================
# METRICS EXPORT FOR MONITORING
# [V6 ENHANCE] Added deduplication-specific metrics
# ============================================================================


@dataclass
class SecurityGuardMetrics:
    """Metrics exportable for monitoring systems"""

    frames_processed: int = 0
    fr_runs: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    avg_recognition_time_ms: float = 0.0
    registry_size: int = 0
    registry_matches: int = 0
    active_alarms: int = 0
    screenshots_saved: int = 0
    screenshots_skipped: int = 0  # [V6 ADD]
    memory_usage_mb: float = 0.0
    circuit_breaker_state: str = "closed"
    frames_since_cleanup: int = 0
    total_faces_detected: int = 0
    unknown_persons_detected: int = 0
    known_persons_detected: int = 0
    alerts_triggered: int = 0
    deduplication_hits: int = 0  # [V6 ADD] Times dedup prevented save
    cross_session_dedup_hits: int = 0  # [V6 ADD] Cross-session dedup prevented save
    track_id_dedup_hits: int = 0  # [V6 ADD] TrackID-based dedup prevented save
    state_transitions: int = 0  # [V6 ADD] State changes prevented


class MetricsExporter:
    """
    Export metrics for Prometheus/DataDog/etc.
    [V6 ENHANCE] Comprehensive deduplication metrics.
    """

    def __init__(self, export_interval: int = 30):
        self._interval = export_interval
        self._recognition_times: List[float] = []
        self._frame_times: List[float] = []
        self._lock = Lock()
        self._process = psutil.Process()

    def record_recognition_time(self, duration_ms: float):
        with self._lock:
            self._recognition_times.append(duration_ms)
            # Keep only last 1000 samples
            if len(self._recognition_times) > 1000:
                self._recognition_times = self._recognition_times[-1000:]

    def record_frame_time(self, duration_ms: float):
        with self._lock:
            self._frame_times.append(duration_ms)
            if len(self._frame_times) > 1000:
                self._frame_times = self._frame_times[-1000:]

    def _calculate_cache_hit_rate(self, guard: "EnhancedSecurityGuard") -> float:
        total = guard.stats.get("cache_hits", 0) + guard.stats.get("cache_misses", 0)
        if total == 0:
            return 0.0
        return guard.stats.get("cache_hits", 0) / total

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self._process.memory_info().rss / 1024 / 1024

    def get_metrics(self, guard: "EnhancedSecurityGuard") -> SecurityGuardMetrics:
        with self._lock:
            return SecurityGuardMetrics(
                frames_processed=guard.stats["frames_processed"],
                fr_runs=guard.stats["fr_runs"],
                cache_hits=guard.stats.get("cache_hits", 0),
                cache_misses=guard.stats.get("cache_misses", 0),
                cache_hit_rate=self._calculate_cache_hit_rate(guard),
                avg_recognition_time_ms=np.mean(self._recognition_times) if self._recognition_times else 0,
                registry_size=len(guard.person_registry),
                registry_matches=guard.stats.get("registry_matches", 0),
                active_alarms=1 if guard._alarm_active else 0,
                screenshots_saved=guard.stats.get("screenshots_saved", 0),
                screenshots_skipped=guard.stats.get("screenshots_skipped", 0),
                memory_usage_mb=self._get_memory_usage(),
                circuit_breaker_state=getattr(guard.face_detector, "_circuit_breaker_state", "unknown"),
                frames_since_cleanup=guard.stats["frames_processed"] % guard.config.registry_cleanup_interval,
                total_faces_detected=guard.stats.get("total_faces_detected", 0),
                unknown_persons_detected=guard.stats.get("unknown_persons_detected", 0),
                known_persons_detected=guard.stats.get("known_persons_detected", 0),
                alerts_triggered=guard.stats.get("alerts_triggered", 0),
                deduplication_hits=guard.stats.get("deduplication_hits", 0),
                cross_session_dedup_hits=guard.stats.get("cross_session_dedup_hits", 0),
                track_id_dedup_hits=guard.stats.get("track_id_dedup_hits", 0),
                state_transitions=guard.stats.get("state_transitions", 0),
            )

    def export_prometheus(self, metrics: SecurityGuardMetrics) -> str:
        """Export metrics in Prometheus text format"""
        lines = [
            "# HELP security_guard_frames_total Total frames processed",
            "# TYPE security_guard_frames_total counter",
            f"security_guard_frames_total {metrics.frames_processed}",
            "",
            "# HELP security_guard_fr_runs_total Total face recognition runs",
            "# TYPE security_guard_fr_runs_total counter",
            f"security_guard_fr_runs_total {metrics.fr_runs}",
            "",
            "# HELP security_guard_cache_hit_rate Cache hit rate",
            "# TYPE security_guard_cache_hit_rate gauge",
            f"security_guard_cache_hit_rate {metrics.cache_hit_rate:.4f}",
            "",
            "# HELP security_guard_avg_recognition_time_ms Average recognition time in milliseconds",
            "# TYPE security_guard_avg_recognition_time_ms gauge",
            f"security_guard_avg_recognition_time_ms {metrics.avg_recognition_time_ms:.2f}",
            "",
            "# HELP security_guard_registry_size Number of registered persons",
            "# TYPE security_guard_registry_size gauge",
            f"security_guard_registry_size {metrics.registry_size}",
            "",
            "# HELP security_guard_active_alarms Current alarm state",
            "# TYPE security_guard_active_alarms gauge",
            f"security_guard_active_alarms {metrics.active_alarms}",
            "",
            "# HELP security_guard_memory_usage_mb Memory usage in megabytes",
            "# TYPE security_guard_memory_usage_mb gauge",
            f"security_guard_memory_usage_mb {metrics.memory_usage_mb:.2f}",
            "",
            "# HELP security_guard_total_faces_detected Total faces detected",
            "# TYPE security_guard_total_faces_detected counter",
            f"security_guard_total_faces_detected {metrics.total_faces_detected}",
            "",
            "# HELP security_guard_unknown_persons_detected Total unknown persons detected",
            "# TYPE security_guard_unknown_persons_detected counter",
            f"security_guard_unknown_persons_detected {metrics.unknown_persons_detected}",
            "",
            "# HELP security_guard_known_persons_detected Total known persons detected",
            "# TYPE security_guard_known_persons_detected counter",
            f"security_guard_known_persons_detected {metrics.known_persons_detected}",
            "",
            "# HELP security_guard_alerts_triggered Total alerts triggered",
            "# TYPE security_guard_alerts_triggered counter",
            f"security_guard_alerts_triggered {metrics.alerts_triggered}",
            "",
            "# HELP security_guard_screenshots_saved Total screenshots saved",
            "# TYPE security_guard_screenshots_saved counter",
            f"security_guard_screenshots_saved {metrics.screenshots_saved}",
            "",
            "# HELP security_guard_screenshots_skipped Total screenshots skipped by dedup",
            "# TYPE security_guard_screenshots_skipped counter",
            f"security_guard_screenshots_skipped {metrics.screenshots_skipped}",
            "",
            "# HELP security_guard_deduplication_hits Times deduplication prevented duplicate save",
            "# TYPE security_guard_deduplication_hits counter",
            f"security_guard_deduplication_hits {metrics.deduplication_hits}",
            "",
            "# HELP security_guard_cross_session_dedup_hits Cross-session dedup prevented save",
            "# TYPE security_guard_cross_session_dedup_hits counter",
            f"security_guard_cross_session_dedup_hits {metrics.cross_session_dedup_hits}",
            "",
            "# HELP security_guard_track_id_dedup_hits TrackID-based dedup prevented save",
            "# TYPE security_guard_track_id_dedup_hits counter",
            f"security_guard_track_id_dedup_hits {metrics.track_id_dedup_hits}",
            "",
            "# HELP security_guard_state_transitions State transitions prevented",
            "# TYPE security_guard_state_transitions counter",
            f"security_guard_state_transitions {metrics.state_transitions}",
        ]
        return "\n".join(lines)

    def export_json(self, metrics: SecurityGuardMetrics) -> str:
        """Export metrics as JSON"""
        return json.dumps(
            {
                "frames_processed": metrics.frames_processed,
                "fr_runs": metrics.fr_runs,
                "cache_hit_rate": metrics.cache_hit_rate,
                "avg_recognition_time_ms": metrics.avg_recognition_time_ms,
                "registry_size": metrics.registry_size,
                "active_alarms": metrics.active_alarms,
                "memory_usage_mb": metrics.memory_usage_mb,
                "circuit_breaker_state": metrics.circuit_breaker_state,
                "total_faces_detected": metrics.total_faces_detected,
                "unknown_persons_detected": metrics.unknown_persons_detected,
                "known_persons_detected": metrics.known_persons_detected,
                "alerts_triggered": metrics.alerts_triggered,
                "screenshots_saved": metrics.screenshots_saved,
                "screenshots_skipped": metrics.screenshots_skipped,
                "deduplication_hits": metrics.deduplication_hits,
                "cross_session_dedup_hits": metrics.cross_session_dedup_hits,
                "track_id_dedup_hits": metrics.track_id_dedup_hits,
                "state_transitions": metrics.state_transitions,
                "timestamp": time.time(),
            },
            indent=2,
        )


# ============================================================================
# SOUND MANAGER (singleton, thread-safe)
# ============================================================================


class SoundManager:
    """Thread-safe singleton sound playback manager for security alerts."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._alarm_loaded = False
            self._logger = _setup_logging("SoundManager")
            self._sound_lock = Lock()
            self._load_alarm_sound()

    def _load_alarm_sound(self) -> None:
        try:
            base_dir = _get_base_dir()
            alarm_file = base_dir / "../media_files/Alarm-sound-samples/humordome-security-alert-sound-453297.mp3"
            if alarm_file.exists():
                pygame.mixer.init()
                pygame.mixer.music.load(str(alarm_file))
                self._alarm_loaded = True
                self._logger.info(f"Alarm sound loaded: {alarm_file}")
            else:
                self._logger.warning(f"Alarm file not found: {alarm_file}")
        except Exception as e:
            self._logger.warning(f"Failed to load alarm: {e}")

    def play_alarm(self) -> bool:
        with self._sound_lock:
            if not self._alarm_loaded:
                return False
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
                    return True
            except Exception as e:
                self._logger.error(f"Failed to play alarm: {e}")
        return False

    def stop_alarm(self) -> None:
        with self._sound_lock:
            try:
                if self._alarm_loaded and pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
            except Exception:
                pass

    def is_loaded(self) -> bool:
        return self._alarm_loaded


# ============================================================================
# EVENT LOGGER
# ============================================================================


class EventLogger:
    """Thread-safe event logger for security events."""

    def __init__(self, log_file: Optional[str] = None):
        base_dir = _get_base_dir()
        self.log_file = log_file or str(base_dir / "security_events.log")
        self._lock = Lock()
        self._logger = _setup_logging("EventLogger")
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else ".", exist_ok=True)

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        with self._lock:
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log_entry = {"timestamp": timestamp, "event": event_type, **data}
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                self._logger.error(f"Logging failed: {e}")


# ============================================================================
# [V6 NEW] IDENTITY STATE MANAGER - Temporal Smoothing for State Consistency
# ============================================================================


@dataclass
class IdentityStateEntry:
    """
    [V6 NEW] Tracks identity state history for temporal smoothing.
    Prevents rapid state flickering between Known and Unknown.
    """

    track_id: int
    persistent_id: Optional[int] = None
    name: str = "Unknown"
    is_known: bool = False
    confidence: float = 0.0
    state_history: List[bool] = field(default_factory=list)  # Rolling history of is_known
    frame_count: int = 0
    last_seen_frame: int = 0
    last_state_change_frame: int = 0
    confirmed_state: Optional[bool] = None  # Stable state after confirmation


class IdentityStateManager:
    """
    [V6 NEW] Manages identity state with temporal smoothing.

    Features:
    - Per-track-ID rolling state history buffer
    - Minimum confirmation frames before state change
    - Prevents identity flickering between frames
    - Voting mechanism: KNOWN suppresses UNKNOWN when confidence >= threshold
    """

    def __init__(
        self,
        history_size: int = DEFAULT_STATE_HISTORY_SIZE,
        min_confirmation_frames: int = DEFAULT_MIN_CONFIRMATION_FRAMES,
        verification_threshold: float = DEFAULT_IDENTITY_VERIFICATION_THRESHOLD,
    ):
        self.history_size = history_size
        self.min_confirmation_frames = min_confirmation_frames
        self.verification_threshold = verification_threshold

        self._states: Dict[int, IdentityStateEntry] = {}
        self._lock = Lock()
        self._logger = _setup_logging("IdentityStateManager")
        self._global_frame_count = 0

    def update(
        self, track_id: int, is_known: bool, confidence: float, name: str, persistent_id: Optional[int] = None
    ) -> Tuple[bool, bool, float]:
        """
        [V6 NEW] Update identity state with temporal smoothing.

        Returns: (final_is_known, state_changed, confidence)

        State Change Rules:
        1. If KNOWN with confidence >= verification_threshold -> immediately KNOWN
        2. If UNKNOWN with low confidence -> requires confirmation buffer
        3. State only changes after min_confirmation_frames consistent readings
        4. KNOWN takes priority over UNKNOWN (voting mechanism)
        """
        with self._lock:
            self._global_frame_count += 1
            current_frame = self._global_frame_count

            # Get or create state entry
            if track_id not in self._states:
                self._states[track_id] = IdentityStateEntry(track_id=track_id)

            entry = self._states[track_id]
            entry.last_seen_frame = current_frame
            entry.frame_count += 1

            # Add to state history
            entry.state_history.append(is_known)
            if len(entry.state_history) > self.history_size:
                entry.state_history.pop(0)

            entry.is_known = is_known
            entry.confidence = confidence
            entry.name = name
            entry.persistent_id = persistent_id

            # Determine final state with voting mechanism
            final_is_known = self._compute_stable_state(entry)

            # Track state transitions
            state_changed = False
            if entry.confirmed_state is None:
                entry.confirmed_state = final_is_known
            elif entry.confirmed_state != final_is_known:
                frames_since_change = current_frame - entry.last_state_change_frame
                if frames_since_change >= self.min_confirmation_frames:
                    # Sufficient frames since last change, allow transition
                    old_state = entry.confirmed_state
                    entry.confirmed_state = final_is_known
                    entry.last_state_change_frame = current_frame
                    state_changed = True
                    self._logger.debug(
                        f"TrackID={track_id} state change: {old_state} -> {final_is_known} "
                        f"(confirmed after {frames_since_change} frames)"
                    )
                else:
                    # Not enough frames, stay in previous state
                    final_is_known = entry.confirmed_state

            return final_is_known, state_changed, confidence

    def _compute_stable_state(self, entry: IdentityStateEntry) -> bool:
        """
        [V6 NEW] Compute stable state using voting mechanism.

        Voting Rules:
        1. If any reading has confidence >= verification_threshold:
           - KNOWN classification immediately wins
        2. Otherwise, use majority vote from history buffer
        3. KNOWN takes priority on ties
        """
        # Rule 1: High confidence KNOWN immediately confirmed
        if entry.is_known and entry.confidence >= self.verification_threshold:
            return True

        # Rule 2: Majority vote from history
        if len(entry.state_history) >= self.min_confirmation_frames:
            known_count = sum(1 for s in entry.state_history if s)
            unknown_count = len(entry.state_history) - known_count

            # KNOWN wins on ties
            if known_count >= unknown_count:
                return True
            else:
                return False

        # Not enough history, use current reading
        return entry.is_known

    def get_state(self, track_id: int) -> Optional[IdentityStateEntry]:
        """Get current state for a track ID"""
        with self._lock:
            return self._states.get(track_id)

    def cleanup_stale(self, max_age_frames: int = 100) -> int:
        """Remove stale entries not seen for max_age_frames"""
        with self._lock:
            current_frame = self._global_frame_count
            to_remove = []

            for track_id, entry in self._states.items():
                if current_frame - entry.last_seen_frame > max_age_frames:
                    to_remove.append(track_id)

            for track_id in to_remove:
                del self._states[track_id]

            if to_remove:
                self._logger.debug(f"Cleaned up {len(to_remove)} stale identity states")

            return len(to_remove)

    def reset(self) -> None:
        """Reset all state"""
        with self._lock:
            self._states.clear()
            self._global_frame_count = 0
            self._logger.info("Identity state manager reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self._lock:
            return {
                "tracked_ids": len(self._states),
                "global_frame_count": self._global_frame_count,
            }


# ============================================================================
# [V6 NEW] TRACK ID DEDUPLICATOR - Spatial/Temporal Deduplication
# ============================================================================


@dataclass
class TrackDeduplicationEntry:
    """[V6 NEW] Track ID deduplication entry"""

    track_id: int
    persistent_id: int
    last_saved_frame: int = 0
    last_saved_time: float = 0.0
    save_count: int = 0


class TrackIdDeduplicator:
    """
    [V6 NEW] TrackID-based deduplication for preventing duplicate saves.

    Features:
    - Frame-skip mechanism (skip saves within N frames of same track ID)
    - Temporal window (skip saves within N seconds of same person)
    - Integrates with YOLO tracking for spatial continuity
    """

    def __init__(
        self,
        frame_skip_window: int = DEFAULT_FRAME_SKIP_WINDOW,
        temporal_window: float = DEFAULT_TEMPORAL_DEDUP_WINDOW,
    ):
        self.frame_skip_window = frame_skip_window
        self.temporal_window = temporal_window

        self._entries: Dict[int, TrackDeduplicationEntry] = {}
        self._lock = Lock()
        self._logger = _setup_logging("TrackIdDeduplicator")
        self._stats = {
            "frame_skips": 0,
            "temporal_skips": 0,
            "total_checks": 0,
        }

    def should_save(self, track_id: int, persistent_id: int, current_frame: int) -> Tuple[bool, str]:
        """
        [V6 NEW] Check if save should proceed based on TrackID deduplication.

        Returns: (should_save, reason)
        Reasons:
            - "proceed": Save should proceed
            - "frame_skip": Within frame-skip window
            - "temporal_skip": Within temporal window
        """
        with self._lock:
            self._stats["total_checks"] += 1

            now = time.time()

            if track_id not in self._entries:
                # First time seeing this track ID
                self._entries[track_id] = TrackDeduplicationEntry(
                    track_id=track_id,
                    persistent_id=persistent_id,
                    last_saved_frame=0,
                    last_saved_time=0,
                    save_count=0,
                )
                return True, "proceed"

            entry = self._entries[track_id]

            # Check frame-skip window
            frames_since_save = current_frame - entry.last_saved_frame
            if frames_since_save < self.frame_skip_window:
                self._stats["frame_skips"] += 1
                self._logger.debug(
                    f"Frame-skip dedup: TrackID={track_id} skipped (frames_since_save={frames_since_save})"
                )
                return False, "frame_skip"

            # Check temporal window
            time_since_save = now - entry.last_saved_time
            if time_since_save < self.temporal_window:
                self._stats["temporal_skips"] += 1
                self._logger.debug(
                    f"Temporal dedup: TrackID={track_id} skipped (time_since_save={time_since_save:.1f}s)"
                )
                return False, "temporal_skip"

            return True, "proceed"

    def record_save(self, track_id: int, persistent_id: int, current_frame: int) -> None:
        """[V6 NEW] Record a save event for deduplication tracking"""
        with self._lock:
            now = time.time()

            if track_id in self._entries:
                entry = self._entries[track_id]
                entry.last_saved_frame = current_frame
                entry.last_saved_time = now
                entry.save_count += 1
            else:
                self._entries[track_id] = TrackDeduplicationEntry(
                    track_id=track_id,
                    persistent_id=persistent_id,
                    last_saved_frame=current_frame,
                    last_saved_time=now,
                    save_count=1,
                )

    def reset_session(self) -> None:
        """[V6 NEW] Reset session-specific tracking"""
        with self._lock:
            self._entries.clear()
            self._logger.info("TrackID deduplicator session reset")

    def get_stats(self) -> Dict[str, Any]:
        """[V6 NEW] Get deduplication statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats["active_tracks"] = len(self._entries)
            return stats


# ============================================================================
# [V6] FACE DEDUPLICATOR - Enhanced with TrackID Integration
# ============================================================================


@dataclass
class DeduplicationEntry:
    """
    [V6 ENHANCE] Persistent entry for deduplication state.
    Stores both perceptual hashes and face embeddings for cross-session dedup.
    """

    persistent_id: int
    timestamp: float
    phash_16: int  # 16x16 perceptual hash
    phash_8: int  # 8x8 perceptual hash (coarse)
    dct_hash: int  # DCT-based hash for rotation invariance
    embedding_hash: str  # MD5 hash of normalized embedding
    embedding_preview: np.ndarray = None  # First 64 dims for quick comparison
    first_seen_box: list = None  # First detected bounding box
    save_count: int = 0  # Number of times this person was saved
    track_ids: Set[int] = field(default_factory=set)  # [V6 ADD] Associated track IDs


class FaceDeduplicator:
    """
    [V6 ENHANCE] Dedicated deduplication engine for face screenshots.

    Multi-layer deduplication strategy:
    1. Immediate: Perceptual hash (aHash, pHash, DCT) comparison
    2. Short-term: Temporal threshold (configurable)
    3. Medium-term: Session count limit
    4. Long-term: Embedding similarity via FAISS index
    5. Cross-session: Persistent state loaded from disk
    6. [V6 ADD] TrackID-based deduplication

    Features:
    - Persistent state file for cross-session deduplication
    - FAISS index for embedding-based similarity search
    - Multi-scale perceptual hashing (8x8, 16x16, 32x32)
    - DCT-based hash for rotation/scale invariance
    - Configurable TTL for stale entries
    - [V6 ADD] TrackID integration for spatial continuity
    """

    def __init__(
        self,
        state_file: Optional[str] = None,
        phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
        dct_threshold: int = DEFAULT_DCT_PHASH_THRESHOLD,
        embedding_sim_threshold: float = DEFAULT_EMBEDDING_SIM_THRESHOLD,
        save_interval: float = DEFAULT_UNKNOWN_SAVE_INTERVAL,
        max_saves_per_session: int = DEFAULT_MAX_SAVES_PER_UNKNOWN,
        entry_ttl: float = DEFAULT_DEDUP_STATE_TTL,
    ):
        self.state_file = state_file or str(_get_base_dir() / "dedup_state.json")
        self.phash_threshold = phash_threshold
        self.dct_threshold = dct_threshold
        self.embedding_sim_threshold = embedding_sim_threshold
        self.save_interval = save_interval
        self.max_saves_per_session = max_saves_per_session
        self.entry_ttl = entry_ttl

        self._entries: Dict[int, DeduplicationEntry] = {}
        self._last_save_time: Dict[int, float] = {}
        self._save_count: Dict[int, int] = {}
        self._lock = Lock()
        self._logger = _setup_logging("FaceDeduplicator")

        # FAISS index for embedding-based similarity
        self._faiss_index = None
        self._embedding_dim = 64  # Use reduced dimension for efficiency
        self._id_to_pid: List[int] = []
        self._pid_to_faiss_idx: Dict[int, int] = {}

        # Load persistent state
        self._load_state()
        self._init_faiss_index()

    def _init_faiss_index(self):
        """Initialize FAISS index for embedding similarity search"""
        if not FAISS_AVAILABLE:
            self._logger.warning("FAISS not available, embedding dedup disabled")
            return

        try:
            self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)
            self._logger.info("FAISS dedup index initialized")
        except Exception as e:
            self._logger.error(f"Failed to create FAISS index: {e}")

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity"""
        norm = np.linalg.norm(emb)
        if norm < 1e-10:
            return emb
        return emb / norm

    def _load_state(self):
        """Load deduplication state from disk"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    data = json.load(f)

                now = time.time()
                for pid_str, entry_data in data.items():
                    pid = int(pid_str)
                    # Skip stale entries
                    if now - entry_data["timestamp"] > self.entry_ttl:
                        continue

                    entry = DeduplicationEntry(
                        persistent_id=pid,
                        timestamp=entry_data["timestamp"],
                        phash_16=entry_data.get("phash_16", 0),
                        phash_8=entry_data.get("phash_8", 0),
                        dct_hash=entry_data.get("dct_hash", 0),
                        embedding_hash=entry_data.get("embedding_hash", ""),
                        save_count=entry_data.get("save_count", 0),
                    )

                    # Restore preview embedding if available
                    if "embedding_preview" in entry_data and entry_data["embedding_preview"]:
                        entry.embedding_preview = np.array(entry_data["embedding_preview"])

                    self._entries[pid] = entry
                    self._last_save_time[pid] = entry.timestamp
                    self._save_count[pid] = entry.save_count

                self._logger.info(f"Loaded {len(self._entries)} deduplication entries from state file")
        except Exception as e:
            self._logger.error(f"Failed to load dedup state: {e}")

    def _save_state(self):
        """Save deduplication state to disk"""
        try:
            data = {}
            for pid, entry in self._entries.items():
                entry_data = {
                    "timestamp": entry.timestamp,
                    "phash_16": entry.phash_16,
                    "phash_8": entry.phash_8,
                    "dct_hash": entry.dct_hash,
                    "embedding_hash": entry.embedding_hash,
                    "save_count": entry.save_count,
                }
                if entry.embedding_preview is not None:
                    entry_data["embedding_preview"] = entry.embedding_preview.tolist()
                data[str(pid)] = entry_data

            with open(self.state_file, "w") as f:
                json.dump(data, f)

            self._logger.debug(f"Saved {len(self._entries)} deduplication entries to state file")
        except Exception as e:
            self._logger.error(f"Failed to save dedup state: {e}")

    @staticmethod
    def _compute_ahash(image: np.ndarray, size: int = 8) -> int:
        """Compute average hash (aHash) for an image."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
            mean = np.mean(resized)
            diff = resized > mean

            hash_val = 0
            for i, val in enumerate(diff.flatten()):
                if val:
                    hash_val |= 1 << i
            return hash_val
        except Exception:
            return 0

    @staticmethod
    def _compute_phash(image: np.ndarray, size: int = 16) -> int:
        """Compute perceptual hash (pHash) using DCT."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            resized = cv2.resize(gray, (size + 1, size + 1), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(resized.astype(np.float32))
            dct_cropped = dct[:size, :size]
            median = np.median(dct_cropped[1:])
            diff = dct_cropped > median

            hash_val = 0
            for i, val in enumerate(diff.flatten()):
                if val:
                    hash_val |= 1 << i
            return hash_val
        except Exception:
            return 0

    @staticmethod
    def _compute_dct_hash(image: np.ndarray, size: int = 32) -> int:
        """Compute DCT-based hash with larger size for rotation invariance."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(resized.astype(np.float32))
            quad_size = size // 2
            dct_quad = dct[:quad_size, :quad_size]
            mean = np.mean(dct_quad[1:])
            diff = dct_quad > mean

            hash_val = 0
            for i, val in enumerate(diff.flatten()):
                if val:
                    hash_val |= 1 << i
            return hash_val
        except Exception:
            return 0

    @staticmethod
    def _hamming_distance(hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two hashes"""
        return bin(hash1 ^ hash2).count("1")

    def _compute_embedding_hash(self, embedding: np.ndarray) -> str:
        """Compute MD5 hash of normalized embedding"""
        norm_emb = embedding / (np.linalg.norm(embedding) + NORMALIZATION_EPS)
        return hashlib.md5(norm_emb.tobytes()).hexdigest()

    def _compute_embedding_preview(self, embedding: np.ndarray) -> np.ndarray:
        """Compute reduced-dimension preview for quick comparison"""
        norm_emb = self._normalize(embedding)
        return norm_emb[: self._embedding_dim]

    def _add_to_faiss_index(self, pid: int, embedding: np.ndarray):
        """Add embedding to FAISS index for similarity search"""
        if self._faiss_index is None:
            return

        try:
            preview = self._compute_embedding_preview(embedding)
            self._faiss_index.add(preview.reshape(1, -1).astype("float32"))
            self._id_to_pid.append(pid)
            self._pid_to_faiss_idx[pid] = len(self._id_to_pid) - 1
        except Exception as e:
            self._logger.error(f"Failed to add to FAISS index: {e}")

    def _find_embedding_match(self, embedding: np.ndarray) -> Optional[Tuple[int, float]]:
        """Find matching persistent_id using embedding similarity."""
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return None

        try:
            preview = self._compute_embedding_preview(embedding)
            query = preview.reshape(1, -1).astype("float32")

            k = min(5, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(query, k)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                if score >= self.embedding_sim_threshold:
                    pid = self._id_to_pid[idx]
                    return pid, float(score)
        except Exception as e:
            self._logger.error(f"FAISS search failed: {e}")

        return None

    def _check_cross_session_dedup(
        self, embedding: np.ndarray, phash_16: int, dct_hash: int
    ) -> Optional[Tuple[str, int, float]]:
        """Check for duplicates across sessions."""
        now = time.time()

        # Check embedding similarity first (fast with FAISS)
        match = self._find_embedding_match(embedding)
        if match:
            pid, sim = match
            self._logger.debug(f"Embedding match found: PID={pid}, similarity={sim:.3f}")
            return ("embedding_match", pid, sim)

        # Check perceptual hashes against all entries
        for pid, entry in self._entries.items():
            if now - entry.timestamp > self.entry_ttl:
                continue

            dct_dist = self._hamming_distance(dct_hash, entry.dct_hash)
            if dct_dist <= self.dct_threshold:
                self._logger.debug(f"DCT hash match: PID={pid}, distance={dct_dist}")
                return ("dct_hash_match", pid, 1.0 - dct_dist / 64)

            phash_dist = self._hamming_distance(phash_16, entry.phash_16)
            if phash_dist <= self.phash_threshold:
                self._logger.debug(f"Perceptual hash match: PID={pid}, distance={phash_dist}")
                return ("phash_match", pid, 1.0 - phash_dist / 256)

        return None

    def should_save(
        self, persistent_id: int, face: np.ndarray, embedding: np.ndarray, is_known: bool
    ) -> Tuple[bool, str, Optional[int], Optional[float]]:
        """
        Comprehensive deduplication check.

        Returns:
            (should_save, reason, matched_pid, confidence)
        """
        # Known persons are always saved
        if is_known:
            return True, "known_person", None, None

        now = time.time()

        # Layer 1: Session count limit
        save_count = self._save_count.get(persistent_id, 0)
        if self.max_saves_per_session > 0 and save_count >= self.max_saves_per_session:
            self._logger.debug(
                f"Session limit reached for PID={persistent_id}: {save_count}/{self.max_saves_per_session}"
            )
            return False, "session_limit", persistent_id, None

        # Layer 2: Temporal threshold
        last_save = self._last_save_time.get(persistent_id, 0)
        if now - last_save < self.save_interval:
            self._logger.debug(
                f"Temporal threshold not met for PID={persistent_id}: {now - last_save:.1f}s < {self.save_interval}s"
            )
            return False, "saved_recently", persistent_id, (now - last_save) / self.save_interval

        # Layer 3: Compute hashes for this face
        phash_16 = self._compute_phash(face, size=16)
        phash_8 = self._compute_ahash(face, size=8)
        dct_hash = self._compute_dct_hash(face, size=32)

        # Layer 4: Cross-session deduplication
        cross_match = self._check_cross_session_dedup(embedding, phash_16, dct_hash)
        if cross_match:
            reason, matched_pid, confidence = cross_match
            if matched_pid != persistent_id:
                self._logger.info(
                    f"Cross-session duplicate detected: PID={persistent_id} matches existing PID={matched_pid} ({reason})"
                )
                return False, f"cross_session_{reason}", matched_pid, confidence

        # Layer 5: Perceptual hash within session
        if persistent_id in self._entries:
            entry = self._entries[persistent_id]
            phash_dist = self._hamming_distance(phash_16, entry.phash_16)
            dct_dist = self._hamming_distance(dct_hash, entry.dct_hash)

            if phash_dist <= self.phash_threshold or dct_dist <= self.dct_threshold:
                self._logger.debug(
                    f"Perceptual match within session: PID={persistent_id}, phash_dist={phash_dist}, dct_dist={dct_dist}"
                )
                return False, "perceptual_match", persistent_id, None

        # Should save
        return True, "new_person", None, None

    def record_save(
        self,
        persistent_id: int,
        face: np.ndarray,
        embedding: np.ndarray,
        box: Optional[list] = None,
        track_id: Optional[int] = None,
    ):
        """Record a successful save for deduplication tracking"""
        with self._lock:
            now = time.time()

            # Compute hashes
            phash_16 = self._compute_phash(face, size=16)
            phash_8 = self._compute_ahash(face, size=8)
            dct_hash = self._compute_dct_hash(face, size=32)
            embedding_hash = self._compute_embedding_hash(embedding)
            embedding_preview = self._compute_embedding_preview(embedding)

            # Update or create entry
            if persistent_id in self._entries:
                entry = self._entries[persistent_id]
                entry.timestamp = now
                entry.phash_16 = phash_16
                entry.phash_8 = phash_8
                entry.dct_hash = dct_hash
                entry.embedding_hash = embedding_hash
                entry.embedding_preview = embedding_preview
                entry.save_count += 1
                if box:
                    entry.first_seen_box = box
                if track_id is not None:
                    entry.track_ids.add(track_id)
            else:
                entry = DeduplicationEntry(
                    persistent_id=persistent_id,
                    timestamp=now,
                    phash_16=phash_16,
                    phash_8=phash_8,
                    dct_hash=dct_hash,
                    embedding_hash=embedding_hash,
                    embedding_preview=embedding_preview,
                    first_seen_box=box,
                    save_count=1,
                    track_ids={track_id} if track_id is not None else set(),
                )
                self._entries[persistent_id] = entry

                # Add to FAISS index
                self._add_to_faiss_index(persistent_id, embedding)

            # Update session tracking
            self._last_save_time[persistent_id] = now
            self._save_count[persistent_id] = self._save_count.get(persistent_id, 0) + 1

            self._logger.debug(
                f"Recorded save for PID={persistent_id}: "
                f"phash={phash_16:#x}, dct={dct_hash:#x}, count={entry.save_count}"
            )

    def reset_session(self):
        """Reset session-specific tracking"""
        with self._lock:
            self._last_save_time.clear()
            self._save_count.clear()
            self._logger.info("Session tracking reset")

    def cleanup_stale(self) -> int:
        """Remove stale entries based on TTL."""
        with self._lock:
            now = time.time()
            to_remove = []

            for pid, entry in self._entries.items():
                if now - entry.timestamp > self.entry_ttl:
                    to_remove.append(pid)

            for pid in to_remove:
                del self._entries[pid]
                self._last_save_time.pop(pid, None)
                self._save_count.pop(pid, None)

            # Rebuild FAISS index
            if self._faiss_index is not None and to_remove:
                self._faiss_index.reset()
                self._id_to_pid.clear()
                self._pid_to_faiss_idx.clear()
                for pid, entry in self._entries.items():
                    if entry.embedding_preview is not None:
                        self._faiss_index.add(entry.embedding_preview.reshape(1, -1).astype("float32"))
                        self._id_to_pid.append(pid)
                        self._pid_to_faiss_idx[pid] = len(self._id_to_pid) - 1

            if to_remove:
                self._logger.info(f"Cleaned up {len(to_remove)} stale dedup entries")
                self._save_state()

            return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return {
            "total_entries": len(self._entries),
            "active_pids": len([e for e in self._entries.values() if time.time() - e.timestamp < self.entry_ttl]),
            "total_saves": sum(e.save_count for e in self._entries.values()),
            "faiss_size": self._faiss_index.ntotal if self._faiss_index else 0,
        }


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class SecurityGuardConfig:
    """Configuration for the Security Guard system."""

    # Processing intervals
    frame_interval: int = 10
    face_recognition_interval: float = 0.1

    # Detection settings
    enable_new_person_detection: bool = True
    face_tolerance: float = 0.2
    embedding_smoothing_frames: int = 5
    min_face_size: Tuple[int, int] = (DEFAULT_MIN_FACE_SIZE, DEFAULT_MIN_FACE_SIZE)

    # InsightFace settings
    insightface_model: str = "buffalo_l"
    insightface_det_size: Tuple[int, int] = (640, 640)

    # Face tracking settings
    enable_face_tracking: bool = True
    face_track_max_age: int = DEFAULT_FACE_TRACK_MAX_AGE
    face_track_iou_threshold: float = DEFAULT_IOU_THRESHOLD

    # Display settings
    enable_liveness: bool = False
    show_attributes: bool = True

    # Display colors (BGR format for OpenCV)
    known_color: Tuple[int, int, int] = (0, 255, 0)
    unknown_color: Tuple[int, int, int] = (0, 165, 255)
    no_face_color: Tuple[int, int, int] = (128, 128, 128)

    # Screenshot settings
    capture_faces: bool = True
    screenshot_dir: Optional[str] = None
    min_face_quality: float = DEFAULT_MIN_FACE_QUALITY

    # Feature flags
    enable_logging: bool = True
    enable_keypoints_extraction: bool = True
    enable_keypoints_display: bool = True

    # Performance optimizations
    min_recognition_quality: float = 30.0
    max_face_yaw: float = 30.0
    known_cache_ttl: float = 60.0
    unknown_cache_ttl: float = 5.0
    min_confidence: float = 0.6

    # Person re-identification settings
    reid_similarity_threshold: float = 0.7
    registry_max_embeddings_per_person: int = 5
    alarm_debounce_frames: int = 3
    alarm_clear_hysteresis: int = 2
    registry_ttl: float = 300.0
    identity_history_max_frames: int = 50
    registry_cleanup_interval: int = 100
    max_seen_track_ids: int = 10000

    # [V4 NEW] Advanced settings
    use_faiss_index: bool = True
    faiss_nprobe: int = 10
    enable_async_screenshots: bool = True
    max_identity_history_size: int = 1000
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout: float = 30.0

    # [V6 NEW] Identity Verification Settings
    identity_verification_threshold: float = DEFAULT_IDENTITY_VERIFICATION_THRESHOLD
    min_confirmation_frames: int = DEFAULT_MIN_CONFIRMATION_FRAMES
    state_history_size: int = DEFAULT_STATE_HISTORY_SIZE

    # [V6 NEW] TrackID Deduplication Settings
    frame_skip_window: int = DEFAULT_FRAME_SKIP_WINDOW
    temporal_dedup_window: float = DEFAULT_TEMPORAL_DEDUP_WINDOW

    # [V6] Deduplication settings
    enable_cross_session_dedup: bool = True
    dedup_state_file: Optional[str] = None
    dedup_phash_threshold: int = DEFAULT_PHASH_THRESHOLD
    dedup_dct_threshold: int = DEFAULT_DCT_PHASH_THRESHOLD
    dedup_embedding_threshold: float = DEFAULT_EMBEDDING_SIM_THRESHOLD
    dedup_save_interval: float = DEFAULT_UNKNOWN_SAVE_INTERVAL
    dedup_max_saves_per_session: int = DEFAULT_MAX_SAVES_PER_UNKNOWN
    dedup_entry_ttl: float = DEFAULT_DEDUP_STATE_TTL

    # YOLO detection settings
    classes: Optional[List[int]] = None

    def validate(self) -> "SecurityGuardConfig":
        if self.frame_interval < 0:
            raise ValueError("frame_interval must be non-negative")
        if not 0.3 <= self.face_tolerance <= 0.7:
            warnings.warn("face_tolerance should be between 0.3 and 0.7")
        if self.min_face_size[0] <= 0 or self.min_face_size[1] <= 0:
            raise ValueError("min_face_size must have positive dimensions")
        if self.use_faiss_index and not FAISS_AVAILABLE:
            warnings.warn("FAISS not available, falling back to linear search")
            self.use_faiss_index = False
        return self

    def __repr__(self) -> str:
        return (
            f"SecurityGuardConfig(tolerance={self.face_tolerance}, "
            f"reid_thresh={self.reid_similarity_threshold}, "
            f"debounce={self.alarm_debounce_frames}, "
            f"use_faiss={self.use_faiss_index}, "
            f"verify_thresh={self.identity_verification_threshold}, "
            f"frame_skip={self.frame_skip_window}, "
            f"cross_session_dedup={self.enable_cross_session_dedup})"
        )


# ============================================================================
# PERSON REGISTRY — with FAISS support
# ============================================================================


@dataclass
class RegistryEntry:
    """Entry in the persistent person registry."""

    persistent_id: int
    name: str
    category: str  # "KNOWN" or "UNKNOWN"
    embeddings: List[np.ndarray]
    last_seen: float
    attributes: Dict


class PersonRegistry:
    """
    Maintains persistent identity database based on face embeddings.
    Uses FAISS for O(log n) approximate nearest neighbor search.
    """

    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self.entries: Dict[int, RegistryEntry] = {}
        self.next_id = 0
        self._lock = Lock()
        self._logger = _setup_logging("PersonRegistry")

        # FAISS index for fast approximate nearest neighbor search
        self._use_faiss = config.use_faiss_index and FAISS_AVAILABLE
        self._embedding_dim = INSIGHTFACE_EMBEDDING_DIM
        self._faiss_index = None
        self._id_to_pid: List[int] = []
        self._pid_to_faiss_idx: Dict[int, int] = {}

        if self._use_faiss:
            self._init_faiss_index()
            self._logger.info("FAISS index initialized for fast embedding search")
        else:
            self._logger.warning("Using linear search (FAISS not available)")

    def _init_faiss_index(self):
        """Initialize FAISS index with Inner Product (cosine similarity)"""
        try:
            self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)
            self._logger.info(f"FAISS index created: dim={self._embedding_dim}")
        except Exception as e:
            self._logger.error(f"Failed to create FAISS index: {e}")
            self._use_faiss = False

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        return emb / (np.linalg.norm(emb) + NORMALIZATION_EPS)

    def find_match(self, embedding: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Find best matching persistent ID using FAISS ANN search.
        Returns (persistent_id, similarity) or None.
        """
        embedding_norm = self._normalize(embedding)

        with self._lock:
            if not self.entries:
                return None

            if self._use_faiss and self._faiss_index is not None and self._faiss_index.ntotal > 0:
                return self._find_match_faiss(embedding_norm)
            else:
                return self._find_match_linear(embedding_norm)

    def _find_match_faiss(self, embedding_norm: np.ndarray) -> Optional[Tuple[int, float]]:
        """FAISS-based approximate nearest neighbor search"""
        try:
            query = embedding_norm.reshape(1, -1).astype("float32")
            k = min(self.config.faiss_nprobe, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(query, k)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                pid = self._id_to_pid[idx]
                if score >= self.config.reid_similarity_threshold:
                    return pid, float(score)
        except Exception as e:
            self._logger.error(f"FAISS search failed: {e}")
            return self._find_match_linear(embedding_norm)
        return None

    def _find_match_linear(self, embedding_norm: np.ndarray) -> Optional[Tuple[int, float]]:
        """Linear search through all embeddings (fallback)"""
        best_id = None
        best_sim = 0.0
        for pid, entry in self.entries.items():
            for stored_emb in entry.embeddings:
                sim = float(np.dot(embedding_norm, stored_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid
        if best_sim >= self.config.reid_similarity_threshold:
            return best_id, best_sim
        return None

    def add_person(self, embedding: np.ndarray, name: str, category: str, attributes: Dict) -> int:
        embedding_norm = self._normalize(embedding)
        with self._lock:
            pid = self.next_id
            self.next_id += 1
            self.entries[pid] = RegistryEntry(
                persistent_id=pid,
                name=name,
                category=category,
                embeddings=[embedding_norm],
                last_seen=time.time(),
                attributes=attributes,
            )

            if self._use_faiss and self._faiss_index is not None:
                try:
                    self._faiss_index.add(embedding_norm.reshape(1, -1).astype("float32"))
                    self._id_to_pid.append(pid)
                    self._pid_to_faiss_idx[pid] = len(self._id_to_pid) - 1
                except Exception as e:
                    self._logger.error(f"Failed to add to FAISS index: {e}")

        self._logger.debug(f"Added person to registry: {name} ({category}) ID={pid}")
        return pid

    def update_person(self, persistent_id: int, embedding: np.ndarray, attributes: Dict = None):
        embedding_norm = self._normalize(embedding)
        with self._lock:
            if persistent_id in self.entries:
                entry = self.entries[persistent_id]
                if len(entry.embeddings) >= self.config.registry_max_embeddings_per_person:
                    entry.embeddings.pop(0)
                    if self._use_faiss:
                        self._rebuild_faiss_index()
                entry.embeddings.append(embedding_norm)
                entry.last_seen = time.time()
                if attributes:
                    entry.attributes.update(attributes)

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index when embeddings are updated"""
        if not self._use_faiss or not self.entries:
            return

        try:
            self._faiss_index.reset()
            self._id_to_pid.clear()
            self._pid_to_faiss_idx.clear()

            for pid, entry in self.entries.items():
                for emb in entry.embeddings:
                    self._faiss_index.add(emb.reshape(1, -1).astype("float32"))
                    self._id_to_pid.append(pid)
                    self._pid_to_faiss_idx[pid] = len(self._id_to_pid) - 1
        except Exception as e:
            self._logger.error(f"Failed to rebuild FAISS index: {e}")

    def get_info(self, persistent_id: int) -> Optional[Tuple[str, str, Dict]]:
        with self._lock:
            entry = self.entries.get(persistent_id)
            if entry:
                return entry.name, entry.category, entry.attributes
        return None

    def get_embedding(self, persistent_id: int) -> Optional[np.ndarray]:
        """Get primary embedding for a persistent ID"""
        with self._lock:
            entry = self.entries.get(persistent_id)
            if entry and entry.embeddings:
                return entry.embeddings[-1]
        return None

    def remove_old(self, max_age: float) -> int:
        """Remove stale entries. Returns count removed."""
        now = time.time()
        with self._lock:
            to_remove = [pid for pid, e in self.entries.items() if now - e.last_seen > max_age]
            for pid in to_remove:
                del self.entries[pid]

            if to_remove and self._use_faiss:
                self._rebuild_faiss_index()

        return len(to_remove)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"PersonRegistry(size={len(self.entries)}, next_id={self.next_id}, use_faiss={self._use_faiss})"


# ============================================================================
# FACE SCREENSHOT CAPTURER — with V6 Enhanced deduplication & TrackID
# ============================================================================


@dataclass
class FaceData:
    """Data class for storing face information."""

    persistent_id: int
    track_id: int  # [V6 ADD] Track ID for deduplication
    box: list
    quality: float
    frame: np.ndarray
    is_known: bool
    name: str
    timestamp: float = field(default_factory=time.time)
    embedding: np.ndarray = None
    confidence: float = 0.0  # [V6 ADD] Recognition confidence


class FaceScreenshotCapturer:
    """
    Captures and saves face screenshots with multi-layer deduplication.
    [V6 ENHANCE] Full integration with FaceDeduplicator, TrackIdDeduplicator, and timestamp directories.
    """

    def __init__(
        self,
        output_dir: str = "../captured_faces",
        min_quality: float = DEFAULT_MIN_FACE_QUALITY,
        async_enabled: bool = True,
        config: Optional[SecurityGuardConfig] = None,
    ):
        # [V6 ENHANCE] Generate timestamp-based directory
        timestamp_dir = _generate_timestamp_dir(output_dir)
        self.output_dir = timestamp_dir
        self.known_dir = os.path.join(timestamp_dir, "known")
        self.unknown_dir = os.path.join(timestamp_dir, "unknown")

        # Graceful handling of existing paths
        for d in [self.known_dir, self.unknown_dir]:
            os.makedirs(d, exist_ok=True)

        self.min_quality = min_quality
        self.best_faces: Dict[int, FaceData] = {}
        self._unknown_count = 0
        self._captured_persistent_ids: set = set()
        self._lock = Lock()
        self._logger = _setup_logging("ScreenshotCapturer")
        self._logger.info(f"Capturing faces to: {timestamp_dir}")

        self._config = config

        # [V6 ENHANCE] TrackID deduplicator
        self._track_deduplicator = TrackIdDeduplicator(
            frame_skip_window=config.frame_skip_window if config else DEFAULT_FRAME_SKIP_WINDOW,
            temporal_window=config.temporal_dedup_window if config else DEFAULT_TEMPORAL_DEDUP_WINDOW,
        )

        # [V6 ENHANCE] Cross-session deduplicator
        dedup_state_file = config.dedup_state_file if config else None
        if config and config.enable_cross_session_dedup:
            self._deduplicator = FaceDeduplicator(
                state_file=dedup_state_file,
                phash_threshold=config.dedup_phash_threshold,
                dct_threshold=config.dedup_dct_threshold,
                embedding_sim_threshold=config.dedup_embedding_threshold,
                save_interval=config.dedup_save_interval,
                max_saves_per_session=config.dedup_max_saves_per_session,
                entry_ttl=config.dedup_entry_ttl,
            )
            self._logger.info("Cross-session deduplication enabled")
        else:
            self._deduplicator = None
            self._logger.info("Cross-session deduplication disabled")

        # Async I/O
        self._async_enabled = async_enabled
        self._executor = ThreadPoolExecutor(max_workers=2) if async_enabled else None
        self._pending_saves: List[Future] = []

        # Stats
        self._stats = {
            "saved": 0,
            "skipped_dedup": 0,
            "cross_session_dedup": 0,
            "temporal_dedup": 0,
            "perceptual_dedup": 0,
            "session_limit_dedup": 0,
            "frame_skip_dedup": 0,
            "temporal_window_dedup": 0,
        }

    def assess_quality(self, frame: np.ndarray, box: list) -> float:
        """Assess face quality based on size, brightness, sharpness, contrast."""
        try:
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return 0.0
            face_area = (x2 - x1) * (y2 - y1)
            frame_area = w * h
            size_score = min(100, (face_area / frame_area) * 1000)
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            bright_score = max(0, 100 - abs(np.mean(gray) - 128) * 0.8)
            sharp_score = min(100, cv2.Laplacian(gray, cv2.CV_64F).var() / 10)
            contrast_score = min(100, np.std(gray) / 30 * 100)
            return (
                size_score * SIZE_WEIGHT
                + bright_score * BRIGHTNESS_WEIGHT
                + sharp_score * SHARPNESS_WEIGHT
                + contrast_score * CONTRAST_WEIGHT
            )
        except Exception:
            return 0.0

    def update_best_face(
        self,
        persistent_id: int,
        track_id: int,  # [V6 ADD]
        box: list,
        frame: np.ndarray,
        is_known: bool,
        name: str = "Unknown",
        embedding: np.ndarray = None,
        confidence: float = 0.0,  # [V6 ADD]
    ) -> bool:
        """Update best face for a track ID"""
        quality = self.assess_quality(frame, box)
        if quality < self.min_quality:
            return False
        with self._lock:
            if persistent_id not in self.best_faces or quality > self.best_faces[persistent_id].quality:
                self.best_faces[persistent_id] = FaceData(
                    persistent_id=persistent_id,
                    track_id=track_id,
                    box=list(box),
                    quality=quality,
                    frame=frame.copy(),
                    is_known=is_known,
                    name=name,
                    embedding=embedding.copy() if embedding is not None else None,
                    confidence=confidence,
                )
                return True
        return False

    def save_all(self) -> int:
        """Save all best faces"""
        saved = 0
        with self._lock:
            for pid, data in list(self.best_faces.items()):
                try:
                    if self._save_face_image(pid, data):
                        saved += 1
                except Exception as e:
                    self._logger.error(f"Save failed for PID {pid}: {e}")
        return saved

    def _save_face_image(self, persistent_id: int, data: FaceData) -> Optional[str]:
        """
        [V6 ENHANCE] Integrated deduplication via FaceDeduplicator and TrackIdDeduplicator.
        """
        x1, y1, x2, y2 = map(int, data.box)
        h, w = data.frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        face = data.frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        embedding = data.embedding
        track_id = data.track_id

        # [V6 ENHANCE] Layer 0: TrackID deduplication (first line of defense)
        current_frame = getattr(self, "_current_frame_count", 0)
        track_should_save, track_reason = self._track_deduplicator.should_save(track_id, persistent_id, current_frame)

        if not track_should_save:
            self._stats["skipped_dedup"] += 1
            if track_reason == "frame_skip":
                self._stats["frame_skip_dedup"] += 1
            elif track_reason == "temporal_skip":
                self._stats["temporal_window_dedup"] += 1
            self._logger.debug(f"TrackID dedup skip: TrackID={track_id}, PID={persistent_id}: {track_reason}")
            return None

        # [V6 ENHANCE] Layer 1: Cross-session deduplication
        if self._deduplicator is not None:
            should_save, reason, matched_pid, confidence = self._deduplicator.should_save(
                persistent_id, face, embedding, data.is_known
            )

            if not should_save:
                self._stats["skipped_dedup"] += 1
                if "cross_session" in reason:
                    self._stats["cross_session_dedup"] += 1
                elif reason == "temporal":
                    self._stats["temporal_dedup"] += 1
                elif reason == "perceptual_match":
                    self._stats["perceptual_dedup"] += 1
                elif reason == "session_limit":
                    self._stats["session_limit_dedup"] += 1

                self._logger.debug(
                    f"Cross-session dedup skip PID {persistent_id}: {reason} "
                    f"(matched_pid={matched_pid}, confidence={confidence})"
                )
                return None

        timestamp = int(time.time())
        if data.is_known:
            filename = f"{data.name}_{data.quality:.0f}_{timestamp}.jpg"
            path = os.path.join(self.known_dir, filename)
        else:
            filename = f"unknown_{persistent_id}_{data.quality:.0f}_{timestamp}.jpg"
            path = os.path.join(self.unknown_dir, filename)

        success = cv2.imwrite(path, face)
        if success:
            self._stats["saved"] += 1
            self._logger.info(f"Saved {'known' if data.is_known else 'unknown'} face: {filename}")

            # Record save in deduplicators
            if self._track_deduplicator:
                self._track_deduplicator.record_save(track_id, persistent_id, current_frame)

            if self._deduplicator is not None and not data.is_known:
                self._deduplicator.record_save(persistent_id, face, embedding, data.box, track_id)
        return path if success else None

    def save_unknown_face(
        self,
        frame: np.ndarray,
        box: list,
        persistent_id: int,
        track_id: int,  # [V6 ADD]
        is_known: bool,
        name: str = "Unknown",
        embedding: np.ndarray = None,
        confidence: float = 0.0,  # [V6 ADD]
    ) -> Optional[str]:
        """Public method for saving unknown faces."""
        face_data = FaceData(
            persistent_id=persistent_id,
            track_id=track_id,
            box=box,
            quality=self.assess_quality(frame, box),
            frame=frame,
            is_known=is_known,
            name=name,
            embedding=embedding,
            confidence=confidence,
        )
        return self._save_face_image(persistent_id, face_data)

    def schedule_save(self, persistent_id: int) -> None:
        """Schedule async save operation"""
        if not self._async_enabled or not self._executor:
            return

        future = self._executor.submit(self._async_save, persistent_id)
        self._pending_saves.append(future)
        self._pending_saves = [f for f in self._pending_saves if not f.done()]

    def _async_save(self, persistent_id: int) -> Optional[str]:
        """Async save implementation"""
        with self._lock:
            if persistent_id not in self.best_faces:
                return None
            data = self.best_faces[persistent_id]
        return self._save_face_image(persistent_id, data)

    def reset_captured_ids(self) -> None:
        with self._lock:
            self._captured_persistent_ids.clear()

    def reset_session(self) -> None:
        """Reset session-specific deduplication state"""
        if self._deduplicator:
            self._deduplicator.reset_session()
        if self._track_deduplicator:
            self._track_deduplicator.reset_session()
        self._logger.info("Screenshot capturer session reset")

    def cleanup_dedup_state(self) -> int:
        """Cleanup stale deduplication entries"""
        cleaned = 0
        if self._deduplicator:
            cleaned += self._deduplicator.cleanup_stale()
        return cleaned

    def get_best_face(self, persistent_id: int) -> Optional[FaceData]:
        with self._lock:
            return self.best_faces.get(persistent_id)

    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics"""
        stats = self._stats.copy()
        if self._deduplicator:
            stats.update(self._deduplicator.get_stats())
        if self._track_deduplicator:
            stats.update(self._track_deduplicator.get_stats())
        return stats

    def shutdown(self):
        """Cleanup executor on shutdown"""
        if self._executor:
            self._executor.shutdown(wait=True)


# ============================================================================
# FACE ATTRIBUTE ANALYZER
# ============================================================================


class FaceAttributeAnalyzer:
    """Formats face attributes for display overlay."""

    EMOTION = EMOTION_MAP
    GENDER = GENDER_MAP

    @staticmethod
    def format(
        age: float, gender: float, emotion: Optional[np.ndarray] = None, pose: Optional[np.ndarray] = None
    ) -> str:
        age_str = f"{int(age)}y"
        gender_str = FaceAttributeAnalyzer.GENDER.get(int(gender > 0.5), "?")
        emo_idx = 1
        if emotion is not None and len(emotion) > 0:
            try:
                emo_idx = int(np.argmax(emotion))
            except Exception:
                pass
        emo_str = FaceAttributeAnalyzer.EMOTION.get(emo_idx, "neutral")
        pose_str = ""
        if pose is not None and len(pose) >= 3:
            try:
                pose_str = f" | P:{int(pose[0])}deg Y:{int(pose[1])}deg"
            except Exception:
                pass
        return f"{age_str} {gender_str} {emo_str}{pose_str}"


# ============================================================================
# PERSON CACHE (track-ID based, secondary to registry)
# ============================================================================


@dataclass
class PersonCacheEntry:
    name: str
    is_known: bool
    timestamp: float
    attributes: Dict
    confidence: float = 0.0


class PersonCache:
    """LRU cache with separate TTLs for known/unknown persons."""

    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self._cache: OrderedDict[int, PersonCacheEntry] = OrderedDict()
        self._lock = Lock()

    def get(self, person_id: int) -> Optional[PersonCacheEntry]:
        with self._lock:
            if person_id in self._cache:
                entry = self._cache[person_id]
                ttl = self.config.known_cache_ttl if entry.is_known else self.config.unknown_cache_ttl
                if time.time() - entry.timestamp < ttl:
                    self._cache.move_to_end(person_id)
                    return entry
                del self._cache[person_id]
        return None

    def set(
        self, person_id: int, name: str, is_known: bool, attributes: Optional[Dict] = None, confidence: float = 0.0
    ) -> None:
        with self._lock:
            self._cache[person_id] = PersonCacheEntry(
                name=name, is_known=is_known, timestamp=time.time(), attributes=attributes or {}, confidence=confidence
            )

    def invalidate(self, person_id: int) -> None:
        with self._lock:
            self._cache.pop(person_id, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ============================================================================
# NEW PERSON TRACKER
# ============================================================================


class NewPersonTracker:
    """
    Tracks newly detected persons. Uses PersonRegistry to avoid
    re-triggering for returning persons with new track IDs.
    """

    def __init__(self, registry: PersonRegistry, max_seen: int = 10000):
        self._seen_track_ids: set = set()
        self._new_track_ids: set = set()
        self._lock = Lock()
        self.registry = registry
        self._max_seen = max_seen

    def update(self, person_ids: list, face_embeddings: Dict[int, np.ndarray]) -> list:
        """Return list of genuinely new track IDs (not in registry)."""
        current = set(person_ids)
        new_track_ids = []
        with self._lock:
            if len(self._seen_track_ids) > self._max_seen:
                self._seen_track_ids = set(sorted(self._seen_track_ids)[-self._max_seen // 2 :])

            for pid in current:
                if pid in self._seen_track_ids:
                    continue
                if pid in face_embeddings:
                    match = self.registry.find_match(face_embeddings[pid])
                    if match is not None:
                        self._seen_track_ids.add(pid)
                        continue
                new_track_ids.append(pid)
                self._new_track_ids.add(pid)
            self._seen_track_ids.update(current)
        return new_track_ids

    def mark_processed(self, person_id: int) -> None:
        with self._lock:
            self._new_track_ids.discard(person_id)

    def is_new(self, person_id: int) -> bool:
        with self._lock:
            return person_id in self._new_track_ids

    def reset(self) -> None:
        with self._lock:
            self._seen_track_ids.clear()
            self._new_track_ids.clear()


# ============================================================================
# FRAME CONTROLLER (OPTIMIZED)
# ============================================================================


class FrameController:
    """Controls when to run face recognition based on configuration."""

    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self._frame_count = 0
        self._last_fr_time = 0.0
        self._lock = Lock()

    def should_run_face_recognition(
        self, person_ids: list, new_person_tracker: Optional["NewPersonTracker"] = None
    ) -> Tuple[bool, bool]:
        """Returns tuple of (should_run, has_new_persons)"""
        with self._lock:
            self._frame_count += 1
            current_time = time.time()

            time_based = False
            if self.config.face_recognition_interval > 0:
                if current_time - self._last_fr_time >= self.config.face_recognition_interval:
                    time_based = True
                    self._last_fr_time = current_time

            frame_based = self.config.frame_interval > 0 and self._frame_count % self.config.frame_interval == 0

            has_new = False
            if self.config.enable_new_person_detection and person_ids and new_person_tracker:
                with new_person_tracker._lock:
                    has_new = any(pid not in new_person_tracker._seen_track_ids for pid in person_ids)

            should_run = time_based or frame_based or has_new
            return should_run, has_new

    def reset(self) -> None:
        with self._lock:
            self._frame_count = 0
            self._last_fr_time = 0.0


# ============================================================================
# INSIGHTFACE DETECTOR (with Circuit Breaker)
# ============================================================================


class InsightFaceDetector:
    """Face detection and embedding extraction using InsightFace with GPU."""

    def __init__(
        self,
        model: str = "buffalo_l",
        detection_size: Tuple[int, int] = (640, 640),
        config: Optional[SecurityGuardConfig] = None,
    ):
        self.model_name = model
        self.detection_size = detection_size
        self.config = config
        self._logger = _setup_logging("InsightFaceDetector")

        if config:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold, recovery_timeout=config.circuit_breaker_timeout
            )
        else:
            self._circuit_breaker = CircuitBreaker()

        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                self._logger.info("CUDA not available, using CPUExecutionProvider")
                warnings.filterwarnings("ignore", message=".*CUDA.*")
        except ImportError:
            cuda_available = False

        try:
            if cuda_available:
                self.app = FaceAnalysis(name=model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                self.app.prepare(ctx_id=0, det_size=detection_size)
                self._logger.info(f"InsightFace initialized: {model} with GPU support")
            else:
                self.app = FaceAnalysis(name=model, providers=["CPUExecutionProvider"])
                self.app.prepare(ctx_id=-1, det_size=detection_size)
                self._logger.info(f"InsightFace initialized: {model} with CPU support")
        except Exception as e:
            self._logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    @property
    def circuit_breaker_state(self) -> str:
        return self._circuit_breaker.get_state()

    def detect(self, frame: np.ndarray) -> Tuple[list, list, list, list]:
        """Detect faces with circuit breaker protection."""
        try:
            return self._circuit_breaker.call(self._detect_impl, frame)
        except CircuitBreakerOpen:
            self._logger.warning("Circuit breaker open, skipping face detection")
            return [], [], [], []
        except Exception as e:
            self._logger.error(f"Face detection failed: {e}")
            self._circuit_breaker._record_failure()
            return [], [], [], []

    def _detect_impl(self, frame: np.ndarray) -> Tuple[list, list, list, list]:
        """Internal detection implementation"""
        faces = self.app.get(frame)
        boxes, embeddings, landmarks, attributes = [], [], [], []
        for face in faces:
            bbox = face.bbox
            if (bbox[2] - bbox[0]) < DEFAULT_MIN_FACE_SIZE or (bbox[3] - bbox[1]) < DEFAULT_MIN_FACE_SIZE:
                continue
            boxes.append(bbox.tolist())
            embeddings.append(face.embedding)
            landmarks.append(face.kps)
            attributes.append(
                {
                    "age": face.age,
                    "gender": face.gender,
                    "emotion": getattr(face, "emotion", np.array([0.5])),
                    "pose": getattr(face, "pose", np.array([0, 0, 0])),
                }
            )
        return boxes, embeddings, landmarks, attributes


# ============================================================================
# FACE RECOGNIZER (Thread-Safe with Cleanup)
# ============================================================================


class FaceRecognizer:
    """
    Face recognition against known faces with multi-embedding + temporal smoothing.
    """

    def __init__(
        self,
        known_embeddings_dict: Dict[str, list],
        tolerance: float = 0.5,
        embedding_smoothing_frames: int = 5,
        min_confidence: float = 0.6,
        identity_max_frames: int = 50,
        max_history_size: int = 1000,
    ):
        self.tolerance = tolerance
        self.embedding_smoothing_frames = embedding_smoothing_frames
        self.min_confidence = min_confidence
        self.identity_max_frames = identity_max_frames
        self.max_history_size = max_history_size
        self._logger = _setup_logging("FaceRecognizer")

        self.known_dict: Dict[str, list] = {}
        self._flat_embeddings = []
        self._flat_names = []
        for name, emb_list in known_embeddings_dict.items():
            norm_list = []
            for emb in emb_list:
                norm = emb / (np.linalg.norm(emb) + NORMALIZATION_EPS)
                norm_list.append(norm)
                self._flat_embeddings.append(norm)
                self._flat_names.append(name)
            self.known_dict[name] = norm_list

        if self._flat_embeddings:
            self._embedding_matrix = np.stack(self._flat_embeddings)
            self._logger.info(
                f"Loaded {len(self.known_dict)} known persons ({len(self._flat_embeddings)} total embeddings)"
            )
        else:
            self._embedding_matrix = np.array([]).reshape(0, 0)
            self._logger.warning("No known faces loaded")

        self._embedding_buffers: Dict[int, list] = {}
        self._identity_history: Dict[int, Tuple[str, float, int]] = {}
        self._identity_lock = Lock()

    def _smooth_embedding(self, track_id: int, embedding: np.ndarray) -> np.ndarray:
        if track_id not in self._embedding_buffers:
            self._embedding_buffers[track_id] = []
        buf = self._embedding_buffers[track_id]
        buf.append(embedding)
        if len(buf) > self.embedding_smoothing_frames:
            buf.pop(0)
        if len(buf) == 1:
            return buf[0]
        weights = np.linspace(0.5, 1.0, len(buf))
        weights /= weights.sum()
        smoothed = np.average(buf, axis=0, weights=weights)
        return smoothed / (np.linalg.norm(smoothed) + NORMALIZATION_EPS)

    def identify(self, embedding: np.ndarray, track_id: Optional[int] = None) -> Tuple[str, bool, float]:
        """Returns: (name, is_known, confidence)."""
        if self._embedding_matrix.size == 0:
            return "Unknown", False, 0.0

        try:
            with self._identity_lock:
                if track_id is not None and track_id in self._identity_history:
                    name, conf, frames = self._identity_history[track_id]
                    if frames >= 3 and conf > 0.8 and frames <= self.identity_max_frames:
                        return name, True, conf
                    elif frames > self.identity_max_frames:
                        del self._identity_history[track_id]
                        self._embedding_buffers.pop(track_id, None)

                if len(self._identity_history) > self.max_history_size:
                    to_remove = len(self._identity_history) - self.max_history_size
                    oldest_keys = sorted(self._identity_history.keys(), key=lambda k: self._identity_history[k][2])[
                        :to_remove
                    ]
                    for key in oldest_keys:
                        del self._identity_history[key]
                        self._embedding_buffers.pop(key, None)

            if track_id is not None:
                embedding = self._smooth_embedding(track_id, embedding)

            query_norm = embedding / (np.linalg.norm(embedding) + NORMALIZATION_EPS)
            similarities = self._embedding_matrix @ query_norm
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            best_name = self._flat_names[best_idx]

            distance = 1 - best_score
            if distance < self.tolerance and best_score >= self.min_confidence:
                if track_id is not None:
                    with self._identity_lock:
                        frames_seen = len(self._embedding_buffers.get(track_id, []))
                        self._identity_history[track_id] = (best_name, best_score, frames_seen)
                return best_name, True, best_score

        except Exception as e:
            self._logger.error(f"Identification error: {e}")

        return "Unknown", False, 0.0

    def clear_track(self, track_id: int) -> None:
        with self._identity_lock:
            self._embedding_buffers.pop(track_id, None)
            self._identity_history.pop(track_id, None)

    def clear_all_tracks(self) -> None:
        with self._identity_lock:
            self._embedding_buffers.clear()
            self._identity_history.clear()


# ============================================================================
# FACE TRACKER (IoU-based, optional)
# ============================================================================


class FaceTracker:
    """IoU-based face tracking with thread safety (optional, for future use)."""

    def __init__(self, max_age: int = DEFAULT_FACE_TRACK_MAX_AGE, iou_threshold: float = DEFAULT_IOU_THRESHOLD):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        self._lock = Lock()

    def _compute_iou(self, b1: list, b2: list) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, boxes, embeddings, attributes) -> Dict[int, Dict]:
        with self._lock:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["age"] += 1
            self.tracks = {k: v for k, v in self.tracks.items() if v["age"] < self.max_age}
            matched = set()
            for box, emb, attr in zip(boxes, embeddings, attributes):
                best_iou, best_id = 0, None
                for tid, td in self.tracks.items():
                    if tid in matched:
                        continue
                    iou = self._compute_iou(box, td["box"])
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou, best_id = iou, tid
                if best_id is not None:
                    self.tracks[best_id] = {"box": box, "embedding": emb, "attributes": attr, "age": 0}
                    matched.add(best_id)
                else:
                    self.tracks[self.next_id] = {"box": box, "embedding": emb, "attributes": attr, "age": 0}
                    matched.add(self.next_id)
                    self.next_id += 1
            return {tid: d for tid, d in self.tracks.items() if tid in matched}

    def reset(self) -> None:
        with self._lock:
            self.tracks.clear()
            self.next_id = 0


# ============================================================================
# SECURITY GUARD MAIN CLASS (V6 with Identity Verification & TrackID Dedup)
# ============================================================================


class EnhancedSecurityGuard(solutions.VisionEye):
    """
    Main security guard: YOLO + InsightFace + persistent re-identification.
    [V6 ENHANCEMENTS]
    - Strict identity verification threshold with voting mechanism
    - TrackID-based spatial/temporal deduplication
    - Temporal smoothing for state consistency
    - Dynamic timestamp-based directory creation
    """

    def __init__(
        self,
        *args,
        config: Optional[SecurityGuardConfig] = None,
        known_embeddings_dict: Optional[Dict[str, list]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config or SecurityGuardConfig()
        self.config.validate()
        self._logger = _setup_logging("SecurityGuard")
        self._initialize_components(known_embeddings_dict)
        self._initialize_stats()
        self._initialize_metrics()
        self._frame_count = 0  # [V6 ADD] Global frame counter for TrackID dedup

        self._logger.info(
            f"Security Guard V6 initialized with "
            f"{len(known_embeddings_dict or {})} known persons, "
            f"FAISS={self.config.use_faiss_index}, "
            f"VerifyThreshold={self.config.identity_verification_threshold}, "
            f"FrameSkip={self.config.frame_skip_window}, "
            f"CrossSessionDedup={self.config.enable_cross_session_dedup}"
        )

    def _initialize_components(self, known_embeddings_dict):
        # Face detector with circuit breaker
        self.face_detector = InsightFaceDetector(
            self.config.insightface_model, self.config.insightface_det_size, self.config
        )

        # Face recognizer with thread-safe history
        self.face_recognizer = FaceRecognizer(
            known_embeddings_dict or {},
            self.config.face_tolerance,
            self.config.embedding_smoothing_frames,
            self.config.min_confidence,
            self.config.identity_history_max_frames,
            self.config.max_identity_history_size,
        )

        self.face_tracker = FaceTracker(self.config.face_track_max_age, self.config.face_track_iou_threshold)
        self.person_cache = PersonCache(self.config)
        self.person_registry = PersonRegistry(self.config)
        self.new_person_tracker = NewPersonTracker(self.person_registry, self.config.max_seen_track_ids)
        self.frame_controller = FrameController(self.config)
        self.event_logger = EventLogger() if self.config.enable_logging else None

        # [V6 NEW] Identity state manager for temporal smoothing
        self.identity_state_manager = IdentityStateManager(
            history_size=self.config.state_history_size,
            min_confirmation_frames=self.config.min_confirmation_frames,
            verification_threshold=self.config.identity_verification_threshold,
        )

        # Screenshot capturer with V6 enhanced deduplication and timestamp directories
        self.screenshot_capturer = None
        if self.config.capture_faces:
            sd = self.config.screenshot_dir or str(_get_base_dir() / "captured_faces")
            self.screenshot_capturer = FaceScreenshotCapturer(
                sd, self.config.min_face_quality, self.config.enable_async_screenshots, self.config
            )

        self.sound_manager = SoundManager()
        self._alarm_active = False
        self._alarm_pending = 0
        self._clear_streak = 0

    def _initialize_stats(self):
        self.stats = {
            "frames_processed": 0,
            "fr_runs": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "new_person_triggers": 0,
            "total_faces_detected": 0,
            "known_persons_detected": 0,
            "unknown_persons_detected": 0,
            "poses_detected": 0,
            "screenshots_saved": 0,
            "screenshots_skipped": 0,
            "alerts_triggered": 0,
            "skipped_low_quality": 0,
            "skipped_yaw": 0,
            "registry_matches": 0,
            "registry_evictions": 0,
            "circuit_breaker_trips": 0,
            "deduplication_hits": 0,
            "cross_session_dedup_hits": 0,
            "track_id_dedup_hits": 0,
            "state_transitions": 0,
            "frame_times_ms": [],
        }

    def _initialize_metrics(self):
        """Initialize metrics exporter"""
        self._metrics_exporter = MetricsExporter(export_interval=30)

    def _build_face_person_map(self, face_boxes: list, person_boxes: list) -> Dict[int, int]:
        """Build mapping from face_idx -> person_idx by checking if face center falls inside person box."""
        face_to_person = {}
        for fi, fbox in enumerate(face_boxes):
            face_cx = (fbox[0] + fbox[2]) / 2
            face_cy = (fbox[1] + fbox[3]) / 2
            for pi, pbox in enumerate(person_boxes):
                if pbox[0] <= face_cx <= pbox[2] and pbox[1] <= face_cy <= pbox[3]:
                    face_to_person[fi] = pi
                    break
        return face_to_person

    def _get_face_for_person(self, person_idx: int, face_to_person: Dict[int, int]) -> Optional[int]:
        """Return the face_idx associated with a given person_idx, or None."""
        for fi, pi in face_to_person.items():
            if pi == person_idx:
                return fi
        return None

    def _process_person(
        self,
        person_id: int,
        person_idx: int,
        face_boxes: list,
        face_embeddings: list,
        face_attributes: list,
        face_to_person: Dict[int, int],
        run_fr: bool,
    ) -> Tuple[str, bool, Dict, Optional[int], float]:
        """
        [V6 ENHANCE] Returns: (name, is_known, attributes, persistent_id, confidence)
        Implements strict identity verification threshold with voting mechanism.
        """
        # Try track-based cache first
        cached = self.person_cache.get(person_id)
        if cached:
            self.stats["cache_hits"] += 1
            return cached.name, cached.is_known, cached.attributes, None, cached.confidence

        self.stats["cache_misses"] += 1

        if not (run_fr and face_embeddings):
            return "Unknown", False, {}, None, 0.0

        # Find the face associated with this person
        face_idx = self._get_face_for_person(person_idx, face_to_person)
        if face_idx is None or face_idx >= len(face_embeddings):
            return "Unknown", False, {}, None, 0.0

        # Quality check
        if self.screenshot_capturer:
            quality = self.screenshot_capturer.assess_quality(self._current_frame, face_boxes[face_idx])
            if quality < self.config.min_recognition_quality:
                self.stats["skipped_low_quality"] += 1
                return "Unknown", False, {}, None, 0.0

        # Yaw check
        if face_idx < len(face_attributes):
            pose = face_attributes[face_idx].get("pose")
            if pose is not None and len(pose) >= 3 and abs(pose[1]) > self.config.max_face_yaw:
                self.stats["skipped_yaw"] += 1
                return "Unknown", False, {}, None, 0.0

        embedding = face_embeddings[face_idx]
        attrs = face_attributes[face_idx] if face_idx < len(face_attributes) else {}

        # [V6 ENHANCE] Step 1: Registry match with verification threshold
        match = self.person_registry.find_match(embedding)
        registry_confidence = 0.0
        if match is not None:
            persistent_id, sim = match
            registry_confidence = sim
            self.stats["registry_matches"] += 1
            info = self.person_registry.get_info(persistent_id)
            if info:
                name, category, reg_attrs = info

                # [V6 ENHANCE] VOTE: If registry match confidence >= verification threshold
                # -> Classify as KNOWN immediately, suppress UNKNOWN
                if registry_confidence >= self.config.identity_verification_threshold:
                    is_known = True
                    self._logger.debug(
                        f"TrackID={person_id} verified KNOWN via registry: "
                        f"PID={persistent_id}, confidence={registry_confidence:.3f} >= "
                        f"threshold={self.config.identity_verification_threshold}"
                    )
                else:
                    # Below threshold - need additional confirmation
                    is_known = category == "KNOWN"

                self.person_registry.update_person(persistent_id, embedding, attrs)
                self.person_cache.set(person_id, name, is_known, reg_attrs, registry_confidence)

                if self.screenshot_capturer:
                    self.screenshot_capturer.update_best_face(
                        persistent_id,
                        person_id,
                        face_boxes[face_idx],
                        self._current_frame,
                        is_known,
                        name,
                        embedding,
                        registry_confidence,
                    )
                return name, is_known, reg_attrs, persistent_id, registry_confidence

        # [V6 ENHANCE] Step 2: Fallback to face recognizer
        name, is_known, fr_confidence = self.face_recognizer.identify(embedding, track_id=person_id)

        # [V6 ENHANCE] VOTE: If FR confidence >= verification threshold -> KNOWN
        if is_known and fr_confidence >= self.config.identity_verification_threshold:
            self._logger.debug(
                f"TrackID={person_id} verified KNOWN via FR: "
                f"confidence={fr_confidence:.3f} >= threshold={self.config.identity_verification_threshold}"
            )

        category = "KNOWN" if is_known else "UNKNOWN"

        # Add to registry
        persistent_id = self.person_registry.add_person(embedding, name, category, attrs)
        self.person_cache.set(person_id, name, is_known, attrs, fr_confidence)

        if self.screenshot_capturer:
            self.screenshot_capturer.update_best_face(
                persistent_id,
                person_id,
                face_boxes[face_idx],
                self._current_frame,
                is_known,
                name,
                embedding,
                fr_confidence,
            )

        if "pose" in attrs and attrs["pose"] is not None:
            self.stats["poses_detected"] += 1

        return name, is_known, attrs, persistent_id, fr_confidence

    def _apply_temporal_smoothing(
        self, person_id: int, raw_is_known: bool, raw_confidence: float, name: str, persistent_id: Optional[int]
    ) -> Tuple[bool, bool, float]:
        """
        [V6 NEW] Apply temporal smoothing to prevent state flickering.
        Returns: (final_is_known, state_changed, final_confidence)
        """
        final_is_known, state_changed, confidence = self.identity_state_manager.update(
            track_id=person_id,
            is_known=raw_is_known,
            confidence=raw_confidence,
            name=name,
            persistent_id=persistent_id,
        )

        if state_changed:
            self.stats["state_transitions"] += 1

        return final_is_known, state_changed, confidence

    def _trigger_alarm(self, unknown_count: int) -> None:
        """Alarm with debouncing"""
        self._alarm_pending += 1
        self._clear_streak = 0
        if self._alarm_pending >= self.config.alarm_debounce_frames and not self._alarm_active:
            if self.sound_manager.play_alarm():
                self._alarm_active = True
                self.stats["alerts_triggered"] += 1
                self._logger.warning(f"ALARM: {unknown_count} unknown person(s)")

    def _clear_alarm(self) -> None:
        """Hysteresis: require N consecutive clear frames before reset."""
        self._clear_streak += 1
        if self._clear_streak >= self.config.alarm_clear_hysteresis:
            self._alarm_pending = 0
            if self._alarm_active:
                self.sound_manager.stop_alarm()
                self._alarm_active = False

    def _save_alarm_screenshots(self, detected_persons: list) -> None:
        """[V6 ENHANCE] Save screenshots for UNKNOWN persons with TrackID dedup"""
        if not self.screenshot_capturer:
            return

        for p in detected_persons:
            if p.get("is_known"):
                continue
            pid = p.get("persistent_id")
            track_id = p.get("id")
            if pid is not None:
                best = self.screenshot_capturer.get_best_face(pid)
                if best:
                    # [V6 ENHANCE] Use TrackID-aware save method
                    path = self.screenshot_capturer.save_unknown_face(
                        self._current_frame,
                        best.box,
                        persistent_id=pid,
                        track_id=track_id,
                        is_known=False,
                        name=p["name"],
                        embedding=best.embedding,
                        confidence=best.confidence,
                    )
                    if path:
                        self.stats["screenshots_saved"] += 1
                    else:
                        self.stats["screenshots_skipped"] += 1
                        # Track dedup statistics
                        dedup_stats = self.screenshot_capturer.get_stats()
                        self.stats["track_id_dedup_hits"] += dedup_stats.get("frame_skip_dedup", 0)
                        self.stats["cross_session_dedup_hits"] += dedup_stats.get("cross_session_dedup", 0)

    def _log_event(self, event_type: str, data: Dict) -> None:
        if self.event_logger:
            self.event_logger.log(event_type, data)

    def _extract_person_data(self) -> Tuple[list, list]:
        person_ids, person_boxes = [], []
        for cls, tid, box in zip(self.clss, self.track_ids, self.boxes):
            if int(cls) == 0:
                person_ids.append(int(tid))
                person_boxes.append(box.tolist())
        return person_ids, person_boxes

    def _extract_pose_keypoints(self) -> Dict[int, list]:
        kpts_dict = {}
        if (
            self.config.enable_keypoints_extraction
            and hasattr(self.tracks, "keypoints")
            and self.tracks.keypoints is not None
        ):
            try:
                kpts_data = self.tracks.keypoints.xy.cpu().tolist()
                if kpts_data:
                    for i, (cls, tid) in enumerate(zip(self.clss, self.track_ids)):
                        if int(cls) == 0 and i < len(kpts_data):
                            kpts_dict[int(tid)] = kpts_data[i]
            except Exception as e:
                self._logger.debug(f"Pose extraction error: {e}")
        return kpts_dict

    def _build_label(self, name: str, is_known: bool, attributes: Dict, confidence) -> str:
        label = name if is_known else "Unknown"
        if confidence is not None and confidence > 0:
            label = f"{label} {confidence:.2f}"
        if self.config.show_attributes and attributes:
            attr_str = FaceAttributeAnalyzer.format(
                attributes.get("age", 0), attributes.get("gender", 0), attributes.get("emotion"), attributes.get("pose")
            )
            label = f"{label}\n{attr_str}"
        return label

    def _annotate_person(self, annotator, box, label, person_id, person_box, color):
        base = self.adjust_box_label(0, 0.0, person_id)
        prefix = str(self.CFG.get("person_label_prefix", label))
        final_label = f"{prefix}: {base}" if base else prefix
        annotator.box_label(box, label=final_label, color=color)
        annotator.visioneye(box, self.vision_point)
        if self.config.enable_keypoints_display and person_id in self._pose_keypoints:
            kpts = self._pose_keypoints[person_id]
            if kpts:
                kpts_array = np.array(kpts, dtype=np.float32)
                annotator.kpts(kpts_array, shape=self._current_frame.shape[:2], kpt_line=True)

    def _annotate_object(self, annotator, cls, tid, box, conf):
        """Added missing method for non-person class annotations."""
        label = f"cls:{int(cls)} id:{int(tid)} {float(conf):.2f}"
        annotator.box_label(box, label=label, color=(200, 200, 200))

    @logged_operation("process_frame")
    def __call__(self, im0: np.ndarray) -> SolutionResults:
        """[V6 ENHANCE] Enhanced deduplication with TrackID and temporal smoothing"""
        frame_start_time = time.time()
        self._current_frame = im0
        self.stats["frames_processed"] += 1
        self._frame_count += 1

        # Update frame counter in screenshot capturer for TrackID dedup
        if self.screenshot_capturer:
            self.screenshot_capturer._current_frame_count = self._frame_count

        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)

        person_ids, person_boxes = self._extract_person_data()
        self._pose_keypoints = self._extract_pose_keypoints()

        run_fr, has_new = self.frame_controller.should_run_face_recognition(person_ids, self.new_person_tracker)

        face_boxes, face_embeddings, face_attributes = [], [], []

        if run_fr:
            self.stats["fr_runs"] += 1
            face_boxes, face_embeddings, _, face_attributes = self.face_detector.detect(im0)
            self.stats["total_faces_detected"] += len(face_boxes)

        # Build face-to-person mapping
        face_to_person = self._build_face_person_map(face_boxes, person_boxes)

        # Build track_id -> embedding for registry lookup
        track_to_embedding = {}
        for fi, pi in face_to_person.items():
            if pi < len(person_ids) and fi < len(face_embeddings):
                track_to_embedding[person_ids[pi]] = face_embeddings[fi]

        # Detect new persons (uses registry for deduplication)
        new_person_ids = self.new_person_tracker.update(person_ids, track_to_embedding)
        if new_person_ids:
            self.stats["new_person_triggers"] += len(new_person_ids)

        # If new persons detected but FR was not run, run it now
        if not run_fr and self.config.enable_new_person_detection and new_person_ids:
            face_boxes, face_embeddings, _, face_attributes = self.face_detector.detect(im0)
            self.stats["total_faces_detected"] += len(face_boxes)
            self.stats["fr_runs"] += 1
            run_fr = True
            face_to_person = self._build_face_person_map(face_boxes, person_boxes)
            track_to_embedding.clear()
            for fi, pi in face_to_person.items():
                if pi < len(person_ids) and fi < len(face_embeddings):
                    track_to_embedding[person_ids[pi]] = face_embeddings[fi]

        # Process each person
        unknown_count = 0
        detected_persons = []
        person_idx_map = {pid: idx for idx, pid in enumerate(person_ids)}

        for cls, tid, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            if int(cls) != 0:
                self._annotate_object(annotator, cls, tid, box, conf)
                continue

            person_id = int(tid)
            person_box = box.tolist()
            person_idx = person_idx_map.get(person_id, -1)

            # [V6 ENHANCE] Get raw classification
            name, raw_is_known, attrs, persistent_id, raw_confidence = self._process_person(
                person_id, person_idx, face_boxes, face_embeddings, face_attributes, face_to_person, run_fr
            )

            # [V6 ENHANCE] Apply temporal smoothing for state consistency
            is_known, state_changed, confidence = self._apply_temporal_smoothing(
                person_id, raw_is_known, raw_confidence, name, persistent_id
            )

            if person_id in new_person_ids:
                self.new_person_tracker.mark_processed(person_id)

            if is_known:
                color = self.config.known_color
                self.stats["known_persons_detected"] += 1
            else:
                face_idx = self._get_face_for_person(person_idx, face_to_person)
                color = self.config.unknown_color if face_idx is not None else self.config.no_face_color
                unknown_count += 1
                self.stats["unknown_persons_detected"] += 1

            label = self._build_label(name, is_known, attrs, confidence)
            self._annotate_person(annotator, box, label, person_id, person_box, color)

            detected_persons.append(
                {
                    "id": person_id,
                    "persistent_id": persistent_id,
                    "name": name,
                    "is_known": is_known,
                    "box": person_box,
                    "attributes": attrs,
                    "confidence": confidence,
                }
            )

        # Alarm handling with debouncing + hysteresis
        alarm_threshold = self.CFG.get("records", 1)
        if unknown_count >= alarm_threshold:
            self._trigger_alarm(unknown_count)
            if self._alarm_active:
                self._save_alarm_screenshots(detected_persons)
                self._log_event("ALARM", {"unknown_count": unknown_count, "persons": detected_persons})
        else:
            self._clear_alarm()

        # Periodic registry eviction
        if self.stats["frames_processed"] % self.config.registry_cleanup_interval == 0:
            evicted = self.person_registry.remove_old(self.config.registry_ttl)
            if evicted:
                self.stats["registry_evictions"] += evicted

            # Cleanup deduplication state
            if self.screenshot_capturer:
                cleaned = self.screenshot_capturer.cleanup_dedup_state()
                if cleaned:
                    self._logger.debug(f"Cleaned {cleaned} stale dedup entries")

            # Cleanup identity state
            self.identity_state_manager.cleanup_stale(max_age_frames=100)

        output_frame = annotator.result()
        self.display_output(output_frame)

        # Record frame time
        frame_time_ms = (time.time() - frame_start_time) * 1000
        self._metrics_exporter.record_frame_time(frame_time_ms)
        self.stats["frame_times_ms"].append(frame_time_ms)
        if len(self.stats["frame_times_ms"]) > 100:
            self.stats["frame_times_ms"] = self.stats["frame_times_ms"][-100:]

        # Extended stats logging
        if self.stats["frames_processed"] % 30 == 0:
            dedup_stats = self.screenshot_capturer.get_stats() if self.screenshot_capturer else {}
            self._logger.info(
                f"Frames:{self.stats['frames_processed']} FR:{self.stats['fr_runs']} "
                f"Cache:{self.stats['cache_hits']} Faces:{self.stats['total_faces_detected']} "
                f"Registry:{len(self.person_registry)} "
                f"Saved:{self.stats['screenshots_saved']} "
                f"Skip:{self.stats['screenshots_skipped']} "
                f"TrackDedup:{dedup_stats.get('frame_skip_dedup', 0)} "
                f"CrossDedup:{dedup_stats.get('cross_session_dedup', 0)} "
                f"StateTransitions:{self.stats['state_transitions']}"
            )

        # HUD overlay with extended stats
        dedup_stats = self.screenshot_capturer.get_stats() if self.screenshot_capturer else {}
        stats_text = (
            f"Tracks:{len(person_ids)} Known:{self.stats['known_persons_detected']} "
            f"Unknown:{self.stats['unknown_persons_detected']}"
        )
        cv2.putText(output_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        status = (
            f"FR:{'ON' if run_fr else 'OFF'} Face:{len(face_boxes)} "
            f"Reg:{len(self.person_registry)} "
            f"Verify:{self.config.identity_verification_threshold:.2f}"
        )
        cv2.putText(output_frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Dedup status in HUD
        dedup_text = f"TrackDedup:{dedup_stats.get('frame_skip_dedup', 0)} CrossDedup:{dedup_stats.get('cross_session_dedup', 0)}"
        cv2.putText(output_frame, dedup_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

        # Frame time
        avg_frame_time = np.mean(self.stats["frame_times_ms"]) if self.stats["frame_times_ms"] else 0
        perf_text = f"FPS:{1000 / avg_frame_time:.1f}" if avg_frame_time > 0 else "FPS: N/A"
        cv2.putText(output_frame, perf_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        return SolutionResults(plot_im=output_frame, total_tracks=len(self.track_ids))

    def save_screenshots(self) -> int:
        return self.screenshot_capturer.save_all() if self.screenshot_capturer else 0

    def get_metrics(self) -> SecurityGuardMetrics:
        """Get current metrics"""
        return self._metrics_exporter.get_metrics(self)

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return self._metrics_exporter.export_prometheus(self.get_metrics())

    def reset(self) -> None:
        self.person_cache.clear()
        self.new_person_tracker.reset()
        self.frame_controller.reset()
        self.face_tracker.reset()
        self.face_recognizer.clear_all_tracks()
        self.identity_state_manager.reset()
        if self.screenshot_capturer:
            self.screenshot_capturer.reset_captured_ids()
            self.screenshot_capturer.reset_session()
        self._alarm_active = False
        self._alarm_pending = 0
        self._clear_streak = 0
        self._frame_count = 0
        self._logger.info("Security Guard reset")

    def shutdown(self):
        """Graceful shutdown"""
        if self.screenshot_capturer:
            self.screenshot_capturer.shutdown()
        self._logger.info("Security Guard shutdown complete")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_security_guard(
    config: Optional[SecurityGuardConfig] = None, face_directory: Optional[str] = None, **kwargs
) -> EnhancedSecurityGuard:
    """
    Create a Security Guard instance.
    Loads known faces from face_directory/person_name/*.jpg -> dict of embeddings.
    """
    cfg = config or SecurityGuardConfig()
    cfg.validate()
    logger = _setup_logging("Factory")
    detector = InsightFaceDetector(cfg.insightface_model, cfg.insightface_det_size, cfg)

    embeddings_dict: Dict[str, list] = {}
    face_dir = face_directory or str(_get_base_dir() / "family_members")
    if os.path.exists(face_dir):
        logger.info(f"Loading known faces from: {face_dir}")
        for person_name in os.listdir(face_dir):
            person_path = os.path.join(face_dir, person_name)
            if not os.path.isdir(person_path):
                continue
            person_embeddings = []
            for img_file in os.listdir(person_path):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(person_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    _, face_embs, _, _ = detector.detect(img)
                    if face_embs:
                        person_embeddings.append(face_embs[0])
                except Exception as e:
                    logger.warning(f"Failed loading {person_name}/{img_file}: {e}")
            if person_embeddings:
                embeddings_dict[person_name] = person_embeddings
                logger.info(f"Loaded {len(person_embeddings)} embeddings for {person_name}")
            else:
                logger.warning(f"No valid faces for {person_name}")
    else:
        logger.warning(f"Face directory not found: {face_dir}")

    return EnhancedSecurityGuard(config=cfg, known_embeddings_dict=embeddings_dict, classes=cfg.classes, **kwargs)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    config = SecurityGuardConfig(
        frame_interval=10,
        face_recognition_interval=0.11,
        enable_new_person_detection=True,
        face_tolerance=0.5,
        enable_face_tracking=True,
        show_attributes=True,
        capture_faces=True,
        enable_logging=True,
        min_recognition_quality=30.0,
        max_face_yaw=30.0,
        known_cache_ttl=60.0,
        unknown_cache_ttl=5.0,
        min_confidence=0.6,
        reid_similarity_threshold=0.7,
        registry_max_embeddings_per_person=5,
        alarm_debounce_frames=3,
        alarm_clear_hysteresis=2,
        registry_ttl=300.0,
        identity_history_max_frames=50,
        # Performance and deduplication optimizations
        use_faiss_index=True,
        enable_async_screenshots=True,
        circuit_breaker_threshold=3,
        # [V6 NEW] Identity Verification Settings
        identity_verification_threshold=0.85,
        min_confirmation_frames=3,
        state_history_size=5,
        # [V6 NEW] TrackID Deduplication Settings
        frame_skip_window=30,
        temporal_dedup_window=5.0,
        # [V6 NEW] Cross-session deduplication
        enable_cross_session_dedup=True,
        dedup_phash_threshold=8,
        dedup_dct_threshold=15,
        dedup_embedding_threshold=0.85,
        dedup_save_interval=30.0,
        dedup_max_saves_per_session=3,
        dedup_entry_ttl=86400 * 7,  # 7 days
    )

    print("\n" + "=" * 60)
    print("Enhanced Security Guard V6 - YOLO + InsightFace + Re-ID + FAISS")
    print("Identity Verification + TrackID Dedup + Temporal Smoothing")
    print("=" * 60 + "\n")

    video_path = "../media_files/WIN_20260227_22_00_29_Pro.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "output_reid_v6.avi"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    guard = create_security_guard(
        config=config,
        show=True,
        model="yolo26m-pose.pt",
        conf=0.6,
        iou=0.8,
        records=1,
    )

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            result = guard(frame)
            writer.write(result.plot_im)
    finally:
        print("\n[INFO] Saving screenshots...")
        saved = guard.save_screenshots()
        print(f"[INFO] Saved {saved} screenshots")
        print(f"[INFO] Output directory: {guard.screenshot_capturer.output_dir}")

        # Print deduplication stats
        if guard.screenshot_capturer:
            dedup_stats = guard.screenshot_capturer.get_stats()
            print("\n" + "=" * 60)
            print("DEDUPLICATION STATISTICS")
            print("=" * 60)
            for key, value in dedup_stats.items():
                print(f"  {key}: {value}")
            print("=" * 60)

        # Export final metrics
        print("\n" + "=" * 60)
        print("METRICS EXPORT (Prometheus format)")
        print("=" * 60)
        print(guard.export_prometheus_metrics())

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        guard.shutdown()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for key, value in guard.stats.items():
            print(f"  {key}: {value}")
        print("=" * 60)
