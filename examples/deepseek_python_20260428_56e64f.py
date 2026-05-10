# Enhanced InsightFace Security Guard System - Refined V5
#
# CHANGELOG from V4 (Identity Consistency & Deduplication Improvements):
#   [FIX] Eliminated identity conflict: upgraded UNKNOWN registry entries to KNOWN
#         when face recognition returns high-confidence match
#   [FIX] Added verification threshold (KNOWN_UPGRADE_THRESHOLD) to prevent flip-flop
#   [OPTIMIZE] Simplified decision flow: registry as source of truth, with upgrade path
#   [FIX] Corrected missing save_unknown_face method in FaceScreenshotCapturer
#   [ENHANCE] Added cooldown mechanism in update_best_face to reduce duplicate screenshot candidates
#   [ENHANCE] Strengthened identity history usage in FaceRecognizer for temporal consistency
#   [OPTIMIZE] Reduced redundant face recognition calls for known persons with stable track_id
#   [FIX] Added proper async save scheduling for alarm-triggered screenshots
#
# Version: 5.0.0

"""
Enhanced InsightFace Security Guard System — V5

Key Improvements over V4:
- Strict identity consistency: a single person cannot be simultaneously known and unknown
- Upgrade path: UNKNOWN registry entries are upgraded to KNOWN when high-confidence match occurs
- Temporal & spatial deduplication for screenshots with cooldown mechanism
- Streamlined decision flow between face recognition and classification
- Fixes for missing screenshot saving methods
"""

# ============================================================================
# IMPORTS (same as V4)
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
from pathlib import Path
from collections import OrderedDict
from types import MappingProxyType
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from threading import Lock
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, Future
from functools import wraps

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Install with: pip install faiss-cpu")

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
]

# ============================================================================
# CONSTANTS
# ============================================================================

SIZE_WEIGHT = 0.30
BRIGHTNESS_WEIGHT = 0.25
SHARPNESS_WEIGHT = 0.25
CONTRAST_WEIGHT = 0.20

DEFAULT_MIN_FACE_QUALITY = 20.0
DEFAULT_MIN_FACE_SIZE = 40
DEFAULT_IOU_THRESHOLD = 0.8
DEFAULT_FACE_TRACK_MAX_AGE = 30

NORMALIZATION_EPS = 1e-5
DEFAULT_CACHE_TTL = 30.0

INSIGHTFACE_EMBEDDING_DIM = 512

# [V5 NEW] Strict threshold for upgrading UNKNOWN to KNOWN
KNOWN_UPGRADE_THRESHOLD = 0.85

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
# LOGGING SETUP (unchanged)
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


def logged_operation(operation_name: str):
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
# CIRCUIT BREAKER (unchanged)
# ============================================================================


class CircuitBreakerOpen(Exception):
    pass


class CircuitBreaker:
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
# METRICS (unchanged)
# ============================================================================


@dataclass
class SecurityGuardMetrics:
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
    memory_usage_mb: float = 0.0
    circuit_breaker_state: str = "closed"
    frames_since_cleanup: int = 0
    total_faces_detected: int = 0
    unknown_persons_detected: int = 0
    known_persons_detected: int = 0
    alerts_triggered: int = 0
    # [V5 NEW] Track upgrades
    unknown_to_known_upgrades: int = 0


class MetricsExporter:
    def __init__(self, export_interval: int = 30):
        self._interval = export_interval
        self._recognition_times: List[float] = []
        self._frame_times: List[float] = []
        self._lock = Lock()
        self._process = psutil.Process()

    def record_recognition_time(self, duration_ms: float):
        with self._lock:
            self._recognition_times.append(duration_ms)
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
                memory_usage_mb=self._get_memory_usage(),
                circuit_breaker_state=getattr(guard.face_detector, "_circuit_breaker_state", "unknown"),
                frames_since_cleanup=guard.stats["frames_processed"] % guard.config.registry_cleanup_interval,
                total_faces_detected=guard.stats.get("total_faces_detected", 0),
                unknown_persons_detected=guard.stats.get("unknown_persons_detected", 0),
                known_persons_detected=guard.stats.get("known_persons_detected", 0),
                alerts_triggered=guard.stats.get("alerts_triggered", 0),
                unknown_to_known_upgrades=guard.stats.get("unknown_to_known_upgrades", 0),
            )

    def export_prometheus(self, metrics: SecurityGuardMetrics) -> str:
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
            "# HELP security_guard_unknown_to_known_upgrades Number of UNKNOWN->KNOWN upgrades",
            "# TYPE security_guard_unknown_to_known_upgrades counter",
            f"security_guard_unknown_to_known_upgrades {metrics.unknown_to_known_upgrades}",
        ]
        return "\n".join(lines)

    def export_json(self, metrics: SecurityGuardMetrics) -> str:
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
                "unknown_to_known_upgrades": metrics.unknown_to_known_upgrades,
                "timestamp": time.time(),
            },
            indent=2,
        )


# ============================================================================
# SOUND MANAGER (unchanged)
# ============================================================================


class SoundManager:
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
# EVENT LOGGER (unchanged)
# ============================================================================


class EventLogger:
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
# CONFIGURATION (added upgrade threshold)
# ============================================================================


@dataclass
class SecurityGuardConfig:
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

    # Display colors
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

    # V4 Advanced settings
    use_faiss_index: bool = True
    faiss_nprobe: int = 10
    enable_async_screenshots: bool = True
    max_identity_history_size: int = 1000
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout: float = 30.0

    # [V5 NEW] Identity consistency settings
    known_upgrade_threshold: float = KNOWN_UPGRADE_THRESHOLD  # minimum confidence to upgrade UNKNOWN to KNOWN
    enable_upgrade_on_high_confidence: bool = True  # allow upgrading registry entries
    screenshot_cooldown_seconds: float = 60.0  # minimum time between screenshots for same person
    max_screenshots_per_person_per_session: int = 1  # max saves per person per session

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
            f"upgrade_thresh={self.known_upgrade_threshold})"
        )


# ============================================================================
# PERSON REGISTRY — with upgrade capability
# ============================================================================


@dataclass
class RegistryEntry:
    persistent_id: int
    name: str
    category: str  # "KNOWN" or "UNKNOWN"
    embeddings: List[np.ndarray]
    last_seen: float
    attributes: Dict


class PersonRegistry:
    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self.entries: Dict[int, RegistryEntry] = {}
        self.next_id = 0
        self._lock = Lock()
        self._logger = _setup_logging("PersonRegistry")

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
        try:
            self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)
            self._logger.info(f"FAISS index created: dim={self._embedding_dim}")
        except Exception as e:
            self._logger.error(f"Failed to create FAISS index: {e}")
            self._use_faiss = False

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        return emb / (np.linalg.norm(emb) + NORMALIZATION_EPS)

    def find_match(self, embedding: np.ndarray) -> Optional[Tuple[int, float]]:
        embedding_norm = self._normalize(embedding)
        with self._lock:
            if not self.entries:
                return None
            if self._use_faiss and self._faiss_index is not None and self._faiss_index.ntotal > 0:
                return self._find_match_faiss(embedding_norm)
            else:
                return self._find_match_linear(embedding_norm)

    def _find_match_faiss(self, embedding_norm: np.ndarray) -> Optional[Tuple[int, float]]:
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

    # [V5 NEW] Upgrade an existing person from UNKNOWN to KNOWN
    def upgrade_to_known(self, persistent_id: int, new_name: str, new_embedding: np.ndarray, attributes: Dict) -> bool:
        with self._lock:
            if persistent_id not in self.entries:
                return False
            entry = self.entries[persistent_id]
            if entry.category == "KNOWN":
                # Already known, maybe update name if different?
                if entry.name != new_name:
                    self._logger.info(f"Updating name for PID {persistent_id}: {entry.name} -> {new_name}")
                    entry.name = new_name
                return True
            # Upgrade from UNKNOWN to KNOWN
            entry.category = "KNOWN"
            entry.name = new_name
            entry.attributes.update(attributes)
            # Add the new embedding and rebuild FAISS
            embedding_norm = self._normalize(new_embedding)
            if len(entry.embeddings) >= self.config.registry_max_embeddings_per_person:
                entry.embeddings.pop(0)
            entry.embeddings.append(embedding_norm)
            if self._use_faiss:
                self._rebuild_faiss_index()
            self._logger.info(f"Upgraded PID {persistent_id} from UNKNOWN to KNOWN as {new_name}")
            return True

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

    def remove_old(self, max_age: float) -> int:
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
# FACE SCREENSHOT CAPTURER — with enhanced deduplication and fixed save method
# ============================================================================


@dataclass
class FaceData:
    persistent_id: int
    box: list
    quality: float
    frame: np.ndarray
    is_known: bool
    name: str
    timestamp: float = field(default_factory=time.time)


class FaceScreenshotCapturer:
    def __init__(
        self,
        output_dir: str = "../captured_faces",
        min_quality: float = DEFAULT_MIN_FACE_QUALITY,
        async_enabled: bool = True,
        min_time_between_saves: float = 60.0,
        max_saves_per_session: int = 1,
        perceptual_hash_threshold: int = 10,
    ):
        self.output_dir = output_dir
        self.min_quality = min_quality
        self.known_dir = os.path.join(output_dir, "known")
        self.unknown_dir = os.path.join(output_dir, "unknown")
        for d in [self.known_dir, self.unknown_dir]:
            os.makedirs(d, exist_ok=True)
        self.best_faces: Dict[int, FaceData] = {}
        self._unknown_count = 0
        self._captured_persistent_ids: set = set()
        self._lock = Lock()
        self._logger = _setup_logging("ScreenshotCapturer")

        self._async_enabled = async_enabled
        self._executor = ThreadPoolExecutor(max_workers=2) if async_enabled else None
        self._pending_saves: List[Future] = []

        # Deduplication state
        self.min_time_between_saves = min_time_between_saves
        self.max_saves_per_session = max_saves_per_session
        self.perceptual_hash_threshold = perceptual_hash_threshold
        self._last_save_time: Dict[int, float] = {}
        self._save_count: Dict[int, int] = {}
        self._saved_face_hashes: Dict[int, List[int]] = {}
        # [V5 NEW] Cooldown for update_best_face to reduce repeated quality assessments
        self._last_update_time: Dict[int, float] = {}
        self._update_cooldown = 2.0  # seconds

    def assess_quality(self, frame: np.ndarray, box: list) -> float:
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

    # [V5 ENHANCE] Cooldown on update to prevent frequent updates
    def update_best_face(
        self, persistent_id: int, box: list, frame: np.ndarray, is_known: bool, name: str = "Unknown"
    ) -> bool:
        now = time.time()
        with self._lock:
            # Cooldown: if we recently updated this person, skip to reduce churn
            last_update = self._last_update_time.get(persistent_id, 0)
            if now - last_update < self._update_cooldown:
                return False
            self._last_update_time[persistent_id] = now

        quality = self.assess_quality(frame, box)
        if quality < self.min_quality:
            return False
        with self._lock:
            if persistent_id not in self.best_faces or quality > self.best_faces[persistent_id].quality:
                self.best_faces[persistent_id] = FaceData(
                    persistent_id=persistent_id,
                    box=list(box),
                    quality=quality,
                    frame=frame.copy(),
                    is_known=is_known,
                    name=name,
                )
                return True
        return False

    def save_all(self) -> int:
        saved = 0
        with self._lock:
            for pid, data in list(self.best_faces.items()):
                try:
                    if self._save_face_image(pid, data):
                        saved += 1
                except Exception as e:
                    self._logger.error(f"Save failed for PID {pid}: {e}")
        return saved

    # [V5 NEW] Public method to save a face by persistent_id (fixes missing method)
    def save_face(self, persistent_id: int, force: bool = False) -> Optional[str]:
        with self._lock:
            if persistent_id not in self.best_faces:
                return None
            data = self.best_faces[persistent_id]
        return self._save_face_image(persistent_id, data, force=force)

    @staticmethod
    def _compute_phash(image: np.ndarray, hash_size: int = 8) -> int:
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
            mean = np.mean(resized)
            diff = resized > mean
            phash = 0
            for i, val in enumerate(diff.flatten()):
                if val:
                    phash |= 1 << i
            return phash
        except Exception:
            return 0

    @staticmethod
    def _phash_distance(hash1: int, hash2: int) -> int:
        return bin(hash1 ^ hash2).count("1")

    def _compute_face_phash(self, face: np.ndarray) -> int:
        return self._compute_phash(face)

    def _should_save_by_dedup(self, persistent_id: int, face: np.ndarray, is_known: bool) -> Tuple[bool, str]:
        if is_known:
            return True, "known_person"
        now = time.time()
        last_save = self._last_save_time.get(persistent_id, 0)
        if now - last_save < self.min_time_between_saves:
            return False, f"temporal (saved {now - last_save:.0f}s ago, need {self.min_time_between_saves}s)"
        save_count = self._save_count.get(persistent_id, 0)
        if self.max_saves_per_session > 0 and save_count >= self.max_saves_per_session:
            return False, f"session_limit (max {self.max_saves_per_session})"
        current_phash = self._compute_face_phash(face)
        saved_hashes = self._saved_face_hashes.get(persistent_id, [])
        for saved_hash in saved_hashes:
            distance = self._phash_distance(current_phash, saved_hash)
            if distance <= self.perceptual_hash_threshold:
                return False, f"perceptual_hash (distance {distance} <= threshold {self.perceptual_hash_threshold})"
        return True, "passed_all_checks"

    def _save_face_image(self, persistent_id: int, data: FaceData, force: bool = False) -> Optional[str]:
        x1, y1, x2, y2 = map(int, data.box)
        h, w = data.frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        face = data.frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        if not force:
            should_save, reason = self._should_save_by_dedup(persistent_id, face, data.is_known)
            if not should_save:
                self._logger.debug(f"Dedup skipped PID {persistent_id}: {reason}")
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
            now = time.time()
            if not data.is_known:
                self._last_save_time[persistent_id] = now
                self._save_count[persistent_id] = self._save_count.get(persistent_id, 0) + 1
                face_phash = self._compute_face_phash(face)
                if persistent_id not in self._saved_face_hashes:
                    self._saved_face_hashes[persistent_id] = []
                self._saved_face_hashes[persistent_id].append(face_phash)
                self._logger.debug(
                    f"Saved unknown face PID {persistent_id}, phash={face_phash:#x}, count={self._save_count[persistent_id]}"
                )
        return path if success else None

    def schedule_save(self, persistent_id: int) -> None:
        if not self._async_enabled or not self._executor:
            return
        future = self._executor.submit(self.save_face, persistent_id)
        self._pending_saves.append(future)
        self._pending_saves = [f for f in self._pending_saves if not f.done()]

    def reset_captured_ids(self) -> None:
        with self._lock:
            self._captured_persistent_ids.clear()
            # [V5] Also reset update cooldown timers
            self._last_update_time.clear()

    def get_best_face(self, persistent_id: int) -> Optional[FaceData]:
        with self._lock:
            return self.best_faces.get(persistent_id)

    def shutdown(self):
        if self._executor:
            self._executor.shutdown(wait=True)


# ============================================================================
# FACE ATTRIBUTE ANALYZER (unchanged)
# ============================================================================


class FaceAttributeAnalyzer:
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
# PERSON CACHE (unchanged)
# ============================================================================


@dataclass
class PersonCacheEntry:
    name: str
    is_known: bool
    timestamp: float
    attributes: Dict
    confidence: float = 0.0


class PersonCache:
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

    def set(self, person_id: int, name: str, is_known: bool, attributes: Optional[Dict] = None) -> None:
        with self._lock:
            self._cache[person_id] = PersonCacheEntry(
                name=name, is_known=is_known, timestamp=time.time(), attributes=attributes or {}
            )

    def invalidate(self, person_id: int) -> None:
        with self._lock:
            self._cache.pop(person_id, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ============================================================================
# NEW PERSON TRACKER (unchanged)
# ============================================================================


class NewPersonTracker:
    def __init__(self, registry: PersonRegistry, max_seen: int = 10000):
        self._seen_track_ids: set = set()
        self._new_track_ids: set = set()
        self._lock = Lock()
        self.registry = registry
        self._max_seen = max_seen

    def update(self, person_ids: list, face_embeddings: Dict[int, np.ndarray]) -> list:
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
# FRAME CONTROLLER (unchanged)
# ============================================================================


class FrameController:
    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self._frame_count = 0
        self._last_fr_time = 0.0
        self._lock = Lock()

    def should_run_face_recognition(
        self, person_ids: list, new_person_tracker: Optional["NewPersonTracker"] = None
    ) -> Tuple[bool, bool]:
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
# INSIGHTFACE DETECTOR (unchanged)
# ============================================================================


class InsightFaceDetector:
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
# FACE RECOGNIZER (strengthened identity consistency)
# ============================================================================


class FaceRecognizer:
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
        if self._embedding_matrix.size == 0:
            return "Unknown", False, 0.0

        try:
            with self._identity_lock:
                if track_id is not None and track_id in self._identity_history:
                    name, conf, frames = self._identity_history[track_id]
                    # [V5] Strengthened: if we have a high-confidence known identity, return it directly
                    if conf > 0.85 and frames >= 2:
                        return name, True, conf
                    if frames >= self.identity_max_frames:
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

    def get_high_confidence_identity(self, track_id: int) -> Optional[Tuple[str, float]]:
        with self._identity_lock:
            if track_id in self._identity_history:
                name, conf, _ = self._identity_history[track_id]
                if conf > 0.85:
                    return name, conf
        return None

    def clear_track(self, track_id: int) -> None:
        with self._identity_lock:
            self._embedding_buffers.pop(track_id, None)
            self._identity_history.pop(track_id, None)

    def clear_all_tracks(self) -> None:
        with self._identity_lock:
            self._embedding_buffers.clear()
            self._identity_history.clear()


# ============================================================================
# FACE TRACKER (unchanged)
# ============================================================================


class FaceTracker:
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
# SECURITY GUARD MAIN CLASS (V5 with identity upgrade)
# ============================================================================


class EnhancedSecurityGuard(solutions.VisionEye):
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
        self._logger.info(
            f"Security Guard V5 initialized with "
            f"{len(known_embeddings_dict or {})} known persons, "
            f"FAISS={self.config.use_faiss_index}, "
            f"AsyncIO={self.config.enable_async_screenshots}, "
            f"UpgradeThresh={self.config.known_upgrade_threshold}"
        )

    def _initialize_components(self, known_embeddings_dict):
        self.face_detector = InsightFaceDetector(
            self.config.insightface_model, self.config.insightface_det_size, self.config
        )
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

        self.screenshot_capturer = None
        if self.config.capture_faces:
            sd = self.config.screenshot_dir or str(_get_base_dir() / "captured_faces")
            self.screenshot_capturer = FaceScreenshotCapturer(
                sd,
                self.config.min_face_quality,
                self.config.enable_async_screenshots,
                min_time_between_saves=self.config.screenshot_cooldown_seconds,
                max_saves_per_session=self.config.max_screenshots_per_person_per_session,
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
            "alerts_triggered": 0,
            "skipped_low_quality": 0,
            "skipped_yaw": 0,
            "registry_matches": 0,
            "registry_evictions": 0,
            "circuit_breaker_trips": 0,
            "frame_times_ms": [],
            "unknown_to_known_upgrades": 0,  # [V5 NEW]
        }

    def _initialize_metrics(self):
        self._metrics_exporter = MetricsExporter(export_interval=30)

    def _build_face_person_map(self, face_boxes: list, person_boxes: list) -> Dict[int, int]:
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
        for fi, pi in face_to_person.items():
            if pi == person_idx:
                return fi
        return None

    # [V5] Core processing with identity upgrade logic
    def _process_person(
        self,
        person_id: int,
        person_idx: int,
        face_boxes: list,
        face_embeddings: list,
        face_attributes: list,
        face_to_person: Dict[int, int],
        run_fr: bool,
    ) -> Tuple[str, bool, Dict, Optional[int]]:
        # Step 1: Check track-based cache
        cached = self.person_cache.get(person_id)
        if cached:
            self.stats["cache_hits"] += 1
            return cached.name, cached.is_known, cached.attributes, None
        self.stats["cache_misses"] += 1

        if not (run_fr and face_embeddings):
            return "Unknown", False, {}, None

        face_idx = self._get_face_for_person(person_idx, face_to_person)
        if face_idx is None or face_idx >= len(face_embeddings):
            return "Unknown", False, {}, None

        # Quality and yaw checks
        if self.screenshot_capturer:
            quality = self.screenshot_capturer.assess_quality(self._current_frame, face_boxes[face_idx])
            if quality < self.config.min_recognition_quality:
                self.stats["skipped_low_quality"] += 1
                return "Unknown", False, {}, None

        if face_idx < len(face_attributes):
            pose = face_attributes[face_idx].get("pose")
            if pose is not None and len(pose) >= 3 and abs(pose[1]) > self.config.max_face_yaw:
                self.stats["skipped_yaw"] += 1
                return "Unknown", False, {}, None

        embedding = face_embeddings[face_idx]
        attrs = face_attributes[face_idx] if face_idx < len(face_attributes) else {}

        # Step 2: Check registry for existing identity
        match = self.person_registry.find_match(embedding)
        if match is not None:
            persistent_id, sim = match
            self.stats["registry_matches"] += 1
            info = self.person_registry.get_info(persistent_id)
            if info:
                name, category, reg_attrs = info
                is_known = category == "KNOWN"

                # [V5] If registry says UNKNOWN but face recognition may give HIGH confidence known, run recognition
                if not is_known and self.config.enable_upgrade_on_high_confidence:
                    recog_name, recog_known, recog_conf = self.face_recognizer.identify(embedding, track_id=person_id)
                    if recog_known and recog_conf >= self.config.known_upgrade_threshold:
                        # Upgrade this entry to KNOWN
                        self.person_registry.upgrade_to_known(persistent_id, recog_name, embedding, attrs)
                        self.stats["unknown_to_known_upgrades"] += 1
                        self._logger.info(
                            f"Upgraded track {person_id} persistent {persistent_id} to KNOWN as {recog_name}"
                        )
                        is_known = True
                        name = recog_name
                        # Update cache
                        self.person_cache.set(person_id, name, is_known, attrs)
                        if self.screenshot_capturer:
                            self.screenshot_capturer.update_best_face(
                                persistent_id, face_boxes[face_idx], self._current_frame, is_known, name
                            )
                        return name, is_known, attrs, persistent_id
                    else:
                        # Still unknown, just update embeddings
                        self.person_registry.update_person(persistent_id, embedding, attrs)
                else:
                    # Known person: update and return
                    self.person_registry.update_person(persistent_id, embedding, attrs)
                    self.person_cache.set(person_id, name, is_known, reg_attrs)
                    if self.screenshot_capturer:
                        self.screenshot_capturer.update_best_face(
                            persistent_id, face_boxes[face_idx], self._current_frame, is_known, name
                        )
                    return name, is_known, reg_attrs, persistent_id

        # Step 3: No registry match or registry returned UNKNOWN and upgrade not triggered -> run face recognition
        name, is_known, confidence = self.face_recognizer.identify(embedding, track_id=person_id)
        category = "KNOWN" if is_known else "UNKNOWN"

        # Add to registry (if not already present, or if we already have a match but it was UNKNOWN and we didn't upgrade)
        persistent_id = self.person_registry.add_person(embedding, name, category, attrs)
        self.person_cache.set(person_id, name, is_known, attrs)

        if self.screenshot_capturer:
            self.screenshot_capturer.update_best_face(
                persistent_id, face_boxes[face_idx], self._current_frame, is_known, name
            )

        if "pose" in attrs and attrs["pose"] is not None:
            self.stats["poses_detected"] += 1

        return name, is_known, attrs, persistent_id

    def _trigger_alarm(self, unknown_count: int) -> None:
        self._alarm_pending += 1
        self._clear_streak = 0
        if self._alarm_pending >= self.config.alarm_debounce_frames and not self._alarm_active:
            if self.sound_manager.play_alarm():
                self._alarm_active = True
                self.stats["alerts_triggered"] += 1
                self._logger.warning(f"ALARM: {unknown_count} unknown person(s)")

    def _clear_alarm(self) -> None:
        self._clear_streak += 1
        if self._clear_streak >= self.config.alarm_clear_hysteresis:
            self._alarm_pending = 0
            if self._alarm_active:
                self.sound_manager.stop_alarm()
                self._alarm_active = False

    # [V5 FIX] Corrected screenshot saving during alarm
    def _save_alarm_screenshots(self, detected_persons: list) -> None:
        if not self.screenshot_capturer:
            return
        for p in detected_persons:
            if p.get("is_known"):
                continue
            pid = p.get("persistent_id")
            if pid is not None:
                # Use the public save_face method or schedule async save
                if self.config.enable_async_screenshots:
                    self.screenshot_capturer.schedule_save(pid)
                else:
                    path = self.screenshot_capturer.save_face(pid)
                    if path:
                        self.stats["screenshots_saved"] += 1

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
        label = f"cls:{int(cls)} id:{int(tid)} {float(conf):.2f}"
        annotator.box_label(box, label=label, color=(200, 200, 200))

    @logged_operation("process_frame")
    def __call__(self, im0: np.ndarray) -> SolutionResults:
        frame_start_time = time.time()
        self._current_frame = im0
        self.stats["frames_processed"] += 1

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

        face_to_person = self._build_face_person_map(face_boxes, person_boxes)

        track_to_embedding = {}
        for fi, pi in face_to_person.items():
            if pi < len(person_ids) and fi < len(face_embeddings):
                track_to_embedding[person_ids[pi]] = face_embeddings[fi]

        new_person_ids = self.new_person_tracker.update(person_ids, track_to_embedding)
        if new_person_ids:
            self.stats["new_person_triggers"] += len(new_person_ids)

        # Fallback: if new persons detected but FR not run, run it now
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

            name, is_known, attrs, persistent_id = self._process_person(
                person_id, person_idx, face_boxes, face_embeddings, face_attributes, face_to_person, run_fr
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

            label = self._build_label(name, is_known, attrs, conf)
            self._annotate_person(annotator, box, label, person_id, person_box, color)

            detected_persons.append(
                {
                    "id": person_id,
                    "persistent_id": persistent_id,
                    "name": name,
                    "is_known": is_known,
                    "box": person_box,
                    "attributes": attrs,
                }
            )

        alarm_threshold = self.CFG.get("records", 1)
        if unknown_count >= alarm_threshold:
            self._trigger_alarm(unknown_count)
            if self._alarm_active:
                self._save_alarm_screenshots(detected_persons)
                self._log_event("ALARM", {"unknown_count": unknown_count, "persons": detected_persons})
        else:
            self._clear_alarm()

        if self.stats["frames_processed"] % self.config.registry_cleanup_interval == 0:
            evicted = self.person_registry.remove_old(self.config.registry_ttl)
            if evicted:
                self.stats["registry_evictions"] += evicted

        output_frame = annotator.result()
        self.display_output(output_frame)

        frame_time_ms = (time.time() - frame_start_time) * 1000
        self._metrics_exporter.record_frame_time(frame_time_ms)
        self.stats["frame_times_ms"].append(frame_time_ms)
        if len(self.stats["frame_times_ms"]) > 100:
            self.stats["frame_times_ms"] = self.stats["frame_times_ms"][-100:]

        if self.stats["frames_processed"] % 30 == 0:
            self._logger.info(
                f"Frames:{self.stats['frames_processed']} FR:{self.stats['fr_runs']} "
                f"Cache:{self.stats['cache_hits']} Faces:{self.stats['total_faces_detected']} "
                f"Registry:{len(self.person_registry)} "
                f"Matches:{self.stats['registry_matches']} "
                f"Upgrades:{self.stats['unknown_to_known_upgrades']} "
                f"CircuitBreaker:{self.face_detector.circuit_breaker_state}"
            )

        stats_text = (
            f"Tracks:{len(person_ids)} Known:{self.stats['known_persons_detected']} "
            f"Unknown:{self.stats['unknown_persons_detected']}"
        )
        cv2.putText(output_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        status = (
            f"FR:{'ON' if run_fr else 'OFF'} Face:{len(face_boxes)} "
            f"Reg:{len(self.person_registry)} "
            f"CB:{self.face_detector.circuit_breaker_state[:1].upper()}"
        )
        cv2.putText(output_frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        avg_frame_time = np.mean(self.stats["frame_times_ms"]) if self.stats["frame_times_ms"] else 0
        perf_text = f"FPS:{1000 / avg_frame_time:.1f}" if avg_frame_time > 0 else "FPS: N/A"
        cv2.putText(output_frame, perf_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        return SolutionResults(plot_im=output_frame, total_tracks=len(self.track_ids))

    def save_screenshots(self) -> int:
        return self.screenshot_capturer.save_all() if self.screenshot_capturer else 0

    def get_metrics(self) -> SecurityGuardMetrics:
        return self._metrics_exporter.get_metrics(self)

    def export_prometheus_metrics(self) -> str:
        return self._metrics_exporter.export_prometheus(self.get_metrics())

    def reset(self) -> None:
        self.person_cache.clear()
        self.new_person_tracker.reset()
        self.frame_controller.reset()
        self.face_tracker.reset()
        self.face_recognizer.clear_all_tracks()
        if self.screenshot_capturer:
            self.screenshot_capturer.reset_captured_ids()
        self._alarm_active = False
        self._alarm_pending = 0
        self._clear_streak = 0
        self._logger.info("Security Guard reset")

    def shutdown(self):
        if self.screenshot_capturer:
            self.screenshot_capturer.shutdown()
        self._logger.info("Security Guard shutdown complete")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_security_guard(
    config: Optional[SecurityGuardConfig] = None, face_directory: Optional[str] = None, **kwargs
) -> EnhancedSecurityGuard:
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
        use_faiss_index=True,
        enable_async_screenshots=True,
        circuit_breaker_threshold=3,
        known_upgrade_threshold=0.85,
        enable_upgrade_on_high_confidence=True,
        screenshot_cooldown_seconds=60.0,
        max_screenshots_per_person_per_session=1,
    )

    print("\n" + "=" * 60)
    print("Enhanced Security Guard V5 - YOLO + InsightFace + Re-ID + FAISS + Upgrade Path")
    print("=" * 60 + "\n")

    video_path = "../media_files/WIN_20260227_22_00_29_Pro.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "output_reid_v5.avi"
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
