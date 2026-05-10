# Enhanced InsightFace Security Guard System - Refined V3
#
# CHANGELOG from Deep Seek Refinement V2.2.0:
#   [FIX] Dead-code branch in __call__ — new-person face detection re-run was unreachable
#   [FIX] _associate_face_to_person now returns a dict mapping face_idx -> person_idx
#   [FIX] Added missing _annotate_object method for non-person classes
#   [FIX] PersonRegistry.find_match threshold check moved inside lock
#   [FIX] Alarm screenshots now only saved for unknown persons
#   [FIX] Removed duplicate imports and unused imports (copy, Union, LOGGER, colors)
#   [FIX] Removed unused Emotion/Gender IntEnum classes (dict maps suffice)
#   [ADD] Registry TTL eviction called periodically from __call__
#   [ADD] Identity history expiration (max frames) in FaceRecognizer
#   [ADD] Alarm hysteresis (require N clear frames before resetting debounce)
#   [ADD] NewPersonTracker max_seen_ids cap to prevent unbounded growth
#   [ADD] Vectorized multi-embedding matching in FaceRecognizer.identify
#   [ADD] __all__ export list and __repr__ on key classes
#
# Version: 3.0.0

"""
Enhanced InsightFace Security Guard System — V3

Features:
1. Person Detection via YOLO (including pose keypoints)
2. Face Detection via InsightFace RetinaFace
3. Face Recognition (known vs unknown) with multi-embedding support
4. Face Attributes (age, gender, emotion, pose)
5. Person Re-Identification (persistent identity across track ID changes)
6. Alarm Debouncing with hysteresis
7. Deduplicated Screenshots using persistent ID
8. Quality/yaw pre-filtering to minimize recognition calls
9. Separate cache TTLs for known/unknown persons

Installation:
    pip install insightface onnxruntime opencv-python numpy pygame ultralytics
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
from pathlib import Path
from collections import OrderedDict
from types import MappingProxyType
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from threading import Lock
from contextlib import contextmanager

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
]

# ============================================================================
# CONSTANTS
# ============================================================================

"""
QUALITY ASSESSMENT WEIGHTS
===========================
These four weights control the composite face quality scoring algorithm.
Each weight determines how much its corresponding quality metric contributes
to the final quality score (0-100). The weights MUST sum to exactly 1.0 to
ensure proper normalization. The quality score is used to filter out blurry,
poorly lit, or low-resolution faces before recognition attempts.

Interaction: These weights are multiplied by their respective quality scores
in the quality assessment function and summed to produce a single quality
value that is compared against DEFAULT_MIN_FACE_QUALITY.

Trade-offs: Higher weights on specific metrics prioritize those aspects.
For example, increasing SIZE_WEIGHT favors larger faces (better for recognition
accuracy) but may reject distant faces. Increasing SHARPNESS_WEIGHT helps
filter motion blur but may reject faces with natural softness.
"""

SIZE_WEIGHT = 0.30
"""
Weight for face size relative to frame dimensions (30% of total score).
Larger faces generally yield better recognition accuracy. This weight is
applied to a normalized size score (0-100) based on face bounding box area.
Valid range: 0.0-1.0. Impact: Higher values prioritize face size over other
quality metrics, potentially rejecting smaller but otherwise clear faces.
"""

BRIGHTNESS_WEIGHT = 0.25
"""
Weight for face brightness/exposure quality (25% of total score).
Evaluates whether the face is properly lit - not too dark or overexposed.
Computed using mean pixel intensity of the grayscale face region.
Valid range: 0.0-1.0. Impact: Higher values reject faces in poor lighting
conditions, which improves recognition reliability but may increase false
negatives in variable lighting environments.
"""

SHARPNESS_WEIGHT = 0.25
"""
Weight for face sharpness/focus quality (25% of total score).
Measures image clarity using Laplacian variance to detect blur (motion blur,
out-of-focus). Higher variance indicates sharper edges and better focus.
Valid range: 0.0-1.0. Impact: Higher values filter out blurry faces that
would produce unreliable embeddings, but may reject faces with natural
softness or slight motion during capture.
"""

CONTRAST_WEIGHT = 0.20
"""
Weight for face contrast quality (20% of total score).
Evaluates tonal range using standard deviation of pixel intensities.
Good contrast ensures facial features are distinguishable for recognition.
Valid range: 0.0-1.0. Impact: Higher values reject washed-out or flat-looking
faces, improving feature extraction but potentially rejecting faces in
low-contrast environments (e.g., fog, uniform backgrounds).
"""

"""
QUALITY AND DETECTION THRESHOLDS
=================================
"""

# DEFAULT_MIN_FACE_QUALITY = 30.0
DEFAULT_MIN_FACE_QUALITY = 20.0
"""
Minimum acceptable face quality score (0-100 scale) for processing.
Faces scoring below this threshold are discarded before recognition attempts.
This value serves as the gatekeeper for the weighted quality metrics above.
Valid range: 0.0-100.0. Recommended: 20-40 for permissive, 50+ for strict.
Trade-offs: Lower values accept more faces (including poor quality) increasing
false positive risk. Higher values ensure only high-quality faces are processed
but may reject legitimate faces in challenging conditions (low light, motion).
Relationship: Compared against the weighted sum of quality metrics computed
using SIZE_WEIGHT, BRIGHTNESS_WEIGHT, SHARPNESS_WEIGHT, and CONTRAST_WEIGHT.
"""

DEFAULT_MIN_FACE_SIZE = 40
"""
Minimum face bounding box dimension in pixels (width and height).
Faces smaller than this are filtered out during detection to avoid processing
faces that are too distant for reliable recognition.
Valid range: 20-200 pixels. Typical: 40-80 for standard surveillance.
Impact: Smaller values allow detection of distant faces but increase false
positives and computational cost. Larger values focus on close-up faces with
better recognition accuracy but miss distant subjects.
Interaction: Used in both initial detection filtering and face tracking to
validate bounding box dimensions before quality assessment.
"""

DEFAULT_IOU_THRESHOLD = 0.8
"""
Intersection over Union (IoU) threshold for bounding box deduplication.
When multiple detections overlap with IoU above this value, they are merged
into a single detection (keeps the one with higher confidence).
Valid range: 0.1-0.9. Lower values (0.2-0.4) are more aggressive at merging;
higher values (0.5-0.8) preserve more distinct detections.
Trade-offs: Lower thresholds reduce duplicate detections but may incorrectly
merge nearby faces. Higher thresholds preserve separate face identities but
may allow duplicate tracking of the same face.
Relationship: Used by the face tracker to determine when detections should
be associated with existing tracks vs. creating new tracks.
"""

DEFAULT_FACE_TRACK_MAX_AGE = 30
"""
Maximum frames a face track persists without a matching detection.
After this many frames without a detection match, the track is considered
lost and removed from active tracking. Prevents ghost tracks from persisting
when faces leave the frame or are occluded.
Valid range: 10-100 frames. Lower values (10-20) make tracking more responsive
to exits; higher values (30-50) maintain tracks through brief occlusions.
Trade-offs: Lower values reduce ghost tracks but may lose legitimate tracks
during brief occlusions or detection failures. Higher values maintain
continuity but may keep stale tracks that slow down matching.
Interaction: Works with DEFAULT_IOU_THRESHOLD - tracks are extended when
a detection with IoU > threshold is found within this frame window.
"""

"""
NUMERICAL STABILITY AND CACHING
================================
"""

NORMALIZATION_EPS = 1e-5
"""
Epsilon value for numerical stability in vector normalization.
Added to the denominator during L2 normalization to prevent division by zero
when computing unit vectors for face embeddings.
Value: 1e-5 (0.00001) - small enough to not affect normal vectors but prevents
NaN/Inf errors when embedding magnitude is zero or near-zero.
Usage: Applied in embedding normalization for cosine similarity calculations
and when smoothing embeddings across multiple frames.
Do not modify unless experiencing numerical precision issues.
"""

DEFAULT_CACHE_TTL = 30.0
"""
Time-to-live (TTL) in seconds for person/face recognition cache entries.
Cached recognition results expire after this duration, forcing re-recognition.
This balances responsiveness (detecting new people) with performance (avoiding
redundant recognition calls for the same person).
Valid range: 10-300 seconds. Lower values (10-30) provide more responsive
updates but increase computational load. Higher values (60-120) reduce
processing but may miss identity changes or new person appearances.
Interaction: Used by the person registry to invalidate stale entries and
trigger fresh recognition when cache expires.
"""

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
        raise ValueError(f"Invalid gender index {index}. Must be in range {GENDER_VALID_INDICES}")
    return GENDER_MAP[index]


def get_emotion_index(label: str) -> Optional[int]:
    return EMOTION_TO_INDEX.get(label.lower())


def get_gender_index(label: str) -> Optional[int]:
    return GENDER_TO_INDEX.get(label.lower())


# ============================================================================
# LOGGING SETUP
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
# CONFIGURATION
# ============================================================================


@dataclass
class SecurityGuardConfig:
    """Configuration for the Security Guard system."""

    # Processing intervals
    # frame_interval: int = 10
    frame_interval: int = 5
    # face_recognition_interval: float = 0.0  # seconds; if >0, overrides frame_interval for recognition calls
    face_recognition_interval: float = 0.3  # seconds; if >0, overrides frame_interval for recognition calls

    # Detection settings
    enable_new_person_detection: bool = True
    # face_tolerance: float = 0.6
    face_tolerance: float = 0.2
    # embedding_smoothing_frames: int = 5
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
    known_color: Tuple[int, int, int] = (0, 255, 0)       # Green
    unknown_color: Tuple[int, int, int] = (0, 165, 255)   # Orange
    no_face_color: Tuple[int, int, int] = (128, 128, 128)  # Gray

    # Screenshot settings
    capture_faces: bool = True
    screenshot_dir: Optional[str] = None
    min_face_quality: float = DEFAULT_MIN_FACE_QUALITY

    # Feature flags
    enable_logging: bool = True
    enable_keypoints_extraction: bool = True
    enable_keypoints_display: bool = False

    # Performance optimizations
    min_recognition_quality: float = 30.0
    max_face_yaw: float = 30.0
    known_cache_ttl: float = 60.0
    unknown_cache_ttl: float = 5.0
    min_confidence: float = 0.6

    # [V3] Person re-identification settings
    reid_similarity_threshold: float = 0.7
    registry_max_embeddings_per_person: int = 5
    alarm_debounce_frames: int = 3
    alarm_clear_hysteresis: int = 2  # [V3] consecutive clear frames before reset
    registry_ttl: float = 300.0     # [V3] evict registry entries older than this (seconds)
    identity_history_max_frames: int = 50  # [V3] expire identity cache after N frames
    registry_cleanup_interval: int = 100   # [V3] run eviction every N frames
    max_seen_track_ids: int = 10000        # [V3] cap for NewPersonTracker._seen_track_ids

    def validate(self) -> "SecurityGuardConfig":
        if self.frame_interval < 0:
            raise ValueError("frame_interval must be non-negative")
        if not 0.3 <= self.face_tolerance <= 0.7:
            warnings.warn("face_tolerance should be between 0.3 and 0.7")
        if self.min_face_size[0] <= 0 or self.min_face_size[1] <= 0:
            raise ValueError("min_face_size must have positive dimensions")
        return self

    def __repr__(self) -> str:
        return (f"SecurityGuardConfig(tolerance={self.face_tolerance}, "
                f"reid_thresh={self.reid_similarity_threshold}, "
                f"debounce={self.alarm_debounce_frames})")


# ============================================================================
# PERSON REGISTRY — persistent identity across track ID changes
# [V3 FIX] threshold check moved inside lock; added __len__; auto-eviction
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
    Matches new face embeddings to previously seen persons even when
    YOLO track IDs change (e.g., person leaves and re-enters frame).
    """

    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self.entries: Dict[int, RegistryEntry] = {}
        self.next_id = 0
        self._lock = Lock()
        self._logger = _setup_logging("PersonRegistry")

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        return emb / (np.linalg.norm(emb) + NORMALIZATION_EPS)

    def find_match(self, embedding: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Find best matching persistent ID for embedding.
        Returns (persistent_id, similarity) or None.
        [V3 FIX] Entire check including threshold is inside lock.
        """
        embedding_norm = self._normalize(embedding)
        with self._lock:
            best_id = None
            best_sim = 0.0
            for pid, entry in self.entries.items():
                for stored_emb in entry.embeddings:
                    sim = float(np.dot(embedding_norm, stored_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_id = pid
            # [V3 FIX] threshold check inside lock to prevent TOCTOU race
            if best_sim >= self.config.reid_similarity_threshold:
                return best_id, best_sim
        return None

    def add_person(self, embedding: np.ndarray, name: str, category: str, attributes: Dict) -> int:
        embedding_norm = self._normalize(embedding)
        with self._lock:
            pid = self.next_id
            self.next_id += 1
            self.entries[pid] = RegistryEntry(
                persistent_id=pid, name=name, category=category,
                embeddings=[embedding_norm], last_seen=time.time(), attributes=attributes,
            )
        self._logger.debug(f"Added person to registry: {name} ({category}) ID={pid}")
        return pid

    def update_person(self, persistent_id: int, embedding: np.ndarray, attributes: Dict = None):
        embedding_norm = self._normalize(embedding)
        with self._lock:
            if persistent_id in self.entries:
                entry = self.entries[persistent_id]
                if len(entry.embeddings) >= self.config.registry_max_embeddings_per_person:
                    entry.embeddings.pop(0)
                entry.embeddings.append(embedding_norm)
                entry.last_seen = time.time()
                if attributes:
                    entry.attributes.update(attributes)

    def get_info(self, persistent_id: int) -> Optional[Tuple[str, str, Dict]]:
        with self._lock:
            entry = self.entries.get(persistent_id)
            if entry:
                return entry.name, entry.category, entry.attributes
        return None

    def remove_old(self, max_age: float) -> int:
        """[V3] Remove stale entries. Returns count removed."""
        now = time.time()
        with self._lock:
            to_remove = [pid for pid, e in self.entries.items() if now - e.last_seen > max_age]
            for pid in to_remove:
                del self.entries[pid]
        return len(to_remove)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"PersonRegistry(size={len(self.entries)}, next_id={self.next_id})"


# ============================================================================
# FACE SCREENSHOT CAPTURER — deduplication via persistent ID
# ============================================================================


@dataclass
class FaceData:
    """Data class for storing face information."""
    persistent_id: int
    box: list
    quality: float
    frame: np.ndarray
    is_known: bool
    name: str
    timestamp: float = field(default_factory=time.time)


class FaceScreenshotCapturer:
    """Captures and saves face screenshots with quality-based dedup using persistent ID."""

    def __init__(self, output_dir: str = "../captured_faces", min_quality: float = DEFAULT_MIN_FACE_QUALITY):
        self.output_dir = output_dir
        self.min_quality = min_quality
        self.known_dir = os.path.join(output_dir, "known")
        self.unknown_dir = os.path.join(output_dir, "unknown")
        for d in [self.known_dir, self.unknown_dir]:
            os.makedirs(d, exist_ok=True)
        self.best_faces: Dict[int, FaceData] = {}  # key = persistent_id
        self._unknown_count = 0
        self._captured_persistent_ids: set = set()
        self._lock = Lock()
        self._logger = _setup_logging("ScreenshotCapturer")

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
            return (size_score * SIZE_WEIGHT + bright_score * BRIGHTNESS_WEIGHT
                    + sharp_score * SHARPNESS_WEIGHT + contrast_score * CONTRAST_WEIGHT)
        except Exception:
            return 0.0

    def update_best_face(self, persistent_id: int, box: list, frame: np.ndarray,
                         is_known: bool, name: str = "Unknown") -> bool:
        quality = self.assess_quality(frame, box)
        if quality < self.min_quality:
            return False
        with self._lock:
            if persistent_id not in self.best_faces or quality > self.best_faces[persistent_id].quality:
                self.best_faces[persistent_id] = FaceData(
                    persistent_id=persistent_id, box=list(box), quality=quality,
                    frame=frame.copy(), is_known=is_known, name=name,
                )
                return True
        return False

    def save_all(self) -> int:
        saved = 0
        with self._lock:
            for pid, data in self.best_faces.items():
                try:
                    if self._save_face_image(pid, data):
                        saved += 1
                except Exception as e:
                    self._logger.error(f"Save failed for PID {pid}: {e}")
        return saved

    def _save_face_image(self, persistent_id: int, data: FaceData) -> Optional[str]:
        x1, y1, x2, y2 = map(int, data.box)
        h, w = data.frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        face = data.frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        timestamp = int(time.time())
        if data.is_known:
            filename = f"{data.name}_{data.quality:.0f}_{timestamp}.jpg"
            path = os.path.join(self.known_dir, filename)
        else:
            self._unknown_count += 1
            filename = f"unknown_{persistent_id}_{data.quality:.0f}_{timestamp}.jpg"
            path = os.path.join(self.unknown_dir, filename)
        return path if cv2.imwrite(path, face) else None

    def save_unknown_face(self, frame: np.ndarray, box: list,
                          persistent_id: Optional[int] = None,
                          is_known: bool = False, name: str = "Unknown") -> Optional[str]:
        """Save face with deduplication based on persistent ID."""
        if persistent_id is not None:
            with self._lock:
                if persistent_id in self._captured_persistent_ids:
                    return None
                self._captured_persistent_ids.add(persistent_id)
        try:
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return None
            timestamp = int(time.time())
            with self._lock:
                if is_known:
                    filename = f"{name}_{timestamp}.jpg"
                    path = os.path.join(self.known_dir, filename)
                else:
                    self._unknown_count += 1
                    pid_tag = f"_{persistent_id}" if persistent_id is not None else f"_{self._unknown_count}"
                    filename = f"unknown{pid_tag}_{timestamp}.jpg"
                    path = os.path.join(self.unknown_dir, filename)
            if cv2.imwrite(path, face):
                return path
        except Exception as e:
            self._logger.error(f"Save face failed: {e}")
        return None

    def reset_captured_ids(self) -> None:
        with self._lock:
            self._captured_persistent_ids.clear()

    def get_best_face(self, persistent_id: int) -> Optional[FaceData]:
        with self._lock:
            return self.best_faces.get(persistent_id)


# ============================================================================
# FACE ATTRIBUTE ANALYZER
# ============================================================================


class FaceAttributeAnalyzer:
    """Formats face attributes for display overlay."""
    EMOTION = EMOTION_MAP
    GENDER = GENDER_MAP

    @staticmethod
    def format(age: float, gender: float, emotion: Optional[np.ndarray] = None,
               pose: Optional[np.ndarray] = None) -> str:
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
# NEW PERSON TRACKER
# [V3] Uses registry to avoid re-triggering; caps _seen_track_ids size
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
        self._max_seen = max_seen  # [V3] prevent unbounded growth

    def update(self, person_ids: list, face_embeddings: Dict[int, np.ndarray]) -> list:
        """Return list of genuinely new track IDs (not in registry)."""
        current = set(person_ids)
        new_track_ids = []
        with self._lock:
            # [V3] cap seen IDs to prevent unbounded memory growth
            if len(self._seen_track_ids) > self._max_seen:
                self._seen_track_ids = set(sorted(self._seen_track_ids)[-self._max_seen // 2:])

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
# FRAME CONTROLLER
# ============================================================================


class FrameController:
    """Controls when to run face recognition based on configuration."""

    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self._frame_count = 0
        self._last_fr_time = 0.0
        self._lock = Lock()

    def should_run_face_recognition(self, new_person_ids: list) -> bool:
        with self._lock:
            self._frame_count += 1
            current_time = time.time()
            if self.config.enable_new_person_detection and new_person_ids:
                return True
            if self.config.face_recognition_interval > 0:
                if current_time - self._last_fr_time >= self.config.face_recognition_interval:
                    self._last_fr_time = current_time
                    return True
                return False
            if self.config.frame_interval > 0:
                return self._frame_count % self.config.frame_interval == 0
            return True

    def reset(self) -> None:
        with self._lock:
            self._frame_count = 0
            self._last_fr_time = 0.0


# ============================================================================
# INSIGHTFACE DETECTOR
# ============================================================================


class InsightFaceDetector:
    """Face detection and embedding extraction using InsightFace with GPU."""

    def __init__(self, model: str = "buffalo_l", detection_size: Tuple[int, int] = (640, 640)):
        self.model_name = model
        self.detection_size = detection_size
        self._logger = _setup_logging("InsightFaceDetector")
        try:
            self.app = FaceAnalysis(name=model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=detection_size)
            self._logger.info(f"InsightFace initialized: {model} with GPU support")
        except Exception as e:
            self._logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Tuple[list, list, list, list]:
        faces = self.app.get(frame)
        boxes, embeddings, landmarks, attributes = [], [], [], []
        for face in faces:
            bbox = face.bbox
            if (bbox[2] - bbox[0]) < DEFAULT_MIN_FACE_SIZE or (bbox[3] - bbox[1]) < DEFAULT_MIN_FACE_SIZE:
                continue
            boxes.append(bbox.tolist())
            embeddings.append(face.embedding)
            landmarks.append(face.kps)
            attributes.append({
                "age": face.age,
                "gender": face.gender,
                "emotion": getattr(face, "emotion", np.array([0.5])),
                "pose": getattr(face, "pose", np.array([0, 0, 0])),
            })
        return boxes, embeddings, landmarks, attributes


# ============================================================================
# FACE RECOGNIZER — multi-embedding support with vectorized matching
# [V3] Added identity history expiration; vectorized dot product
# ============================================================================


class FaceRecognizer:
    """Face recognition against known faces with multi-embedding + temporal smoothing."""

    def __init__(self, known_embeddings_dict: Dict[str, list],
                 tolerance: float = 0.5, embedding_smoothing_frames: int = 5,
                 min_confidence: float = 0.6, identity_max_frames: int = 50):
        self.tolerance = tolerance
        self.embedding_smoothing_frames = embedding_smoothing_frames
        self.min_confidence = min_confidence
        self.identity_max_frames = identity_max_frames  # [V3] expiration
        self._logger = _setup_logging("FaceRecognizer")

        # Build per-person normalized embedding lists
        self.known_dict: Dict[str, list] = {}
        # Also build a flat matrix + name index for vectorized matching
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

        # [V3] Pre-compute matrix for vectorized dot product
        if self._flat_embeddings:
            self._embedding_matrix = np.stack(self._flat_embeddings)  # (N, D)
            self._logger.info(f"Loaded {len(self.known_dict)} known persons "
                              f"({len(self._flat_embeddings)} total embeddings)")
        else:
            self._embedding_matrix = np.array([]).reshape(0, 0)
            self._logger.warning("No known faces loaded")

        self._embedding_buffers: Dict[int, list] = {}
        self._identity_history: Dict[int, Tuple[str, float, int]] = {}  # name, conf, frames_seen

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
            if track_id is not None:
                # [V3] Check identity history with expiration
                if track_id in self._identity_history:
                    name, conf, frames = self._identity_history[track_id]
                    if frames >= 3 and conf > 0.8 and frames <= self.identity_max_frames:
                        return name, True, conf
                    elif frames > self.identity_max_frames:
                        # [V3] Expired — re-evaluate
                        del self._identity_history[track_id]
                embedding = self._smooth_embedding(track_id, embedding)

            query_norm = embedding / (np.linalg.norm(embedding) + NORMALIZATION_EPS)

            # [V3] Vectorized dot product across all known embeddings
            similarities = self._embedding_matrix @ query_norm  # (N,)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            best_name = self._flat_names[best_idx]

            distance = 1 - best_score
            if distance < self.tolerance and best_score >= self.min_confidence:
                if track_id is not None:
                    frames_seen = len(self._embedding_buffers.get(track_id, []))
                    self._identity_history[track_id] = (best_name, best_score, frames_seen)
                return best_name, True, best_score

        except Exception as e:
            self._logger.error(f"Identification error: {e}")

        return "Unknown", False, 0.0

    def clear_track(self, track_id: int) -> None:
        self._embedding_buffers.pop(track_id, None)
        self._identity_history.pop(track_id, None)

    def clear_all_tracks(self) -> None:
        self._embedding_buffers.clear()
        self._identity_history.clear()

# ============================================================================
# FACE TRACKER (IoU-based, optional)
# ============================================================================


class FaceTracker:
    """IoU-based face tracking with thread safety (optional, for future use)."""

    def __init__(self, max_age: int = DEFAULT_FACE_TRACK_MAX_AGE,
                 iou_threshold: float = DEFAULT_IOU_THRESHOLD):
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
# SECURITY GUARD MAIN CLASS
# [V3 FIXES] dead-code branch, face-person mapping, _annotate_object,
#   alarm hysteresis, screenshot save only for unknowns, registry eviction
# ============================================================================


class EnhancedSecurityGuard(solutions.VisionEye):
    """Main security guard: YOLO + InsightFace + persistent re-identification."""

    def __init__(self, *args, config: Optional[SecurityGuardConfig] = None,
                 known_embeddings_dict: Optional[Dict[str, list]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or SecurityGuardConfig()
        self.config.validate()
        self._logger = _setup_logging("SecurityGuard")
        self._initialize_components(known_embeddings_dict)
        self._initialize_stats()
        self._logger.info(f"Security Guard V3 initialized with "
                          f"{len(known_embeddings_dict or {})} known persons")

    def _initialize_components(self, known_embeddings_dict):
        self.face_detector = InsightFaceDetector(
            self.config.insightface_model, self.config.insightface_det_size)
        self.face_recognizer = FaceRecognizer(
            known_embeddings_dict or {}, self.config.face_tolerance,
            self.config.embedding_smoothing_frames, self.config.min_confidence,
            self.config.identity_history_max_frames)
        self.face_tracker = FaceTracker(
            self.config.face_track_max_age, self.config.face_track_iou_threshold)
        self.person_cache = PersonCache(self.config)
        self.person_registry = PersonRegistry(self.config)
        self.new_person_tracker = NewPersonTracker(
            self.person_registry, self.config.max_seen_track_ids)
        self.frame_controller = FrameController(self.config)
        self.event_logger = EventLogger() if self.config.enable_logging else None

        self.screenshot_capturer = None
        if self.config.capture_faces:
            sd = self.config.screenshot_dir or str(_get_base_dir() / "captured_faces")
            self.screenshot_capturer = FaceScreenshotCapturer(sd, self.config.min_face_quality)

        self.sound_manager = SoundManager()
        self._alarm_active = False
        self._alarm_pending = 0
        self._clear_streak = 0  # [V3] hysteresis counter

    def _initialize_stats(self):
        self.stats = {
            "frames_processed": 0, "fr_runs": 0, "cache_hits": 0,
            "new_person_triggers": 0, "total_faces_detected": 0,
            "known_persons_detected": 0, "unknown_persons_detected": 0,
            "poses_detected": 0, "screenshots_saved": 0, "alerts_triggered": 0,
            "skipped_low_quality": 0, "skipped_yaw": 0,
            "registry_matches": 0, "registry_evictions": 0,
        }

    def _build_face_person_map(self, face_boxes: list, person_boxes: list) -> Dict[int, int]:
        """
        [V3 FIX] Build mapping from face_idx -> person_idx by checking
        if the face center falls inside a person bounding box.
        Returns dict mapping face index to person index.
        """
        face_to_person = {}
        for fi, fbox in enumerate(face_boxes):
            face_cx = (fbox[0] + fbox[2]) / 2
            face_cy = (fbox[1] + fbox[3]) / 2
            for pi, pbox in enumerate(person_boxes):
                if pbox[0] <= face_cx <= pbox[2] and pbox[1] <= face_cy <= pbox[3]:
                    face_to_person[fi] = pi
                    break
        return face_to_person

    def _get_face_for_person(self, person_idx: int,
                             face_to_person: Dict[int, int]) -> Optional[int]:
        """Return the face_idx associated with a given person_idx, or None."""
        for fi, pi in face_to_person.items():
            if pi == person_idx:
                return fi
        return None

    def _process_person(self, person_id: int, person_idx: int,
                        face_boxes: list, face_embeddings: list,
                        face_attributes: list, face_to_person: Dict[int, int],
                        run_fr: bool) -> Tuple[str, bool, Dict, Optional[int]]:
        """
        Returns: (name, is_known, attributes, persistent_id)
        [V3 FIX] Uses face_to_person mapping for correct association.
        """
        # Try track-based cache first
        cached = self.person_cache.get(person_id)
        if cached:
            self.stats["cache_hits"] += 1
            return cached.name, cached.is_known, cached.attributes, None

        if not (run_fr and face_embeddings):
            return "Unknown", False, {}, None

        # [V3 FIX] Find the face associated with this person using the mapping
        face_idx = self._get_face_for_person(person_idx, face_to_person)
        if face_idx is None or face_idx >= len(face_embeddings):
            return "Unknown", False, {}, None

        # Quality check
        if self.screenshot_capturer:
            quality = self.screenshot_capturer.assess_quality(
                self._current_frame, face_boxes[face_idx])
            if quality < self.config.min_recognition_quality:
                self.stats["skipped_low_quality"] += 1
                return "Unknown", False, {}, None

        # Yaw check
        if face_idx < len(face_attributes):
            pose = face_attributes[face_idx].get("pose")
            if pose is not None and len(pose) >= 3 and abs(pose[1]) > self.config.max_face_yaw:
                self.stats["skipped_yaw"] += 1
                return "Unknown", False, {}, None

        embedding = face_embeddings[face_idx]
        attrs = face_attributes[face_idx] if face_idx < len(face_attributes) else {}

        # First try persistent registry match
        match = self.person_registry.find_match(embedding)
        if match is not None:
            persistent_id, sim = match
            self.stats["registry_matches"] += 1
            info = self.person_registry.get_info(persistent_id)
            if info:
                name, category, reg_attrs = info
                is_known = category == "KNOWN"
                self.person_registry.update_person(persistent_id, embedding, attrs)
                self.person_cache.set(person_id, name, is_known, reg_attrs)
                if self.screenshot_capturer:
                    self.screenshot_capturer.update_best_face(
                        persistent_id, face_boxes[face_idx],
                        self._current_frame, is_known, name)
                return name, is_known, reg_attrs, persistent_id

        # No registry match: perform face recognition
        name, is_known, confidence = self.face_recognizer.identify(
            embedding, track_id=person_id)
        category = "KNOWN" if is_known else "UNKNOWN"

        # Add to registry
        persistent_id = self.person_registry.add_person(
            embedding, name, category, attrs)
        self.person_cache.set(person_id, name, is_known, attrs)

        if self.screenshot_capturer:
            self.screenshot_capturer.update_best_face(
                persistent_id, face_boxes[face_idx],
                self._current_frame, is_known, name)

        if "pose" in attrs and attrs["pose"] is not None:
            self.stats["poses_detected"] += 1

        return name, is_known, attrs, persistent_id

    def _trigger_alarm(self, unknown_count: int) -> None:
        """[V3] Alarm with debouncing — requires N consecutive frames."""
        self._alarm_pending += 1
        self._clear_streak = 0  # reset clear counter on any trigger
        if self._alarm_pending >= self.config.alarm_debounce_frames and not self._alarm_active:
            if self.sound_manager.play_alarm():
                self._alarm_active = True
                self.stats["alerts_triggered"] += 1
                self._logger.warning(f"ALARM: {unknown_count} unknown person(s)")

    def _clear_alarm(self) -> None:
        """[V3] Hysteresis: require N consecutive clear frames before reset."""
        self._clear_streak += 1
        if self._clear_streak >= self.config.alarm_clear_hysteresis:
            self._alarm_pending = 0
            if self._alarm_active:
                self.sound_manager.stop_alarm()
                self._alarm_active = False

    def _save_alarm_screenshots(self, detected_persons: list) -> None:
        """[V3 FIX] Only save screenshots for UNKNOWN persons."""
        if not self.screenshot_capturer:
            return
        for p in detected_persons:
            # [V3 FIX] Skip known persons
            if p.get("is_known"):
                continue
            pid = p.get("persistent_id")
            if pid is not None:
                best = self.screenshot_capturer.get_best_face(pid)
                if best:
                    path = self.screenshot_capturer.save_unknown_face(
                        self._current_frame, best.box,
                        persistent_id=pid, is_known=False, name=p["name"])
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
        if (self.config.enable_keypoints_extraction
                and hasattr(self.tracks, "keypoints")
                and self.tracks.keypoints is not None):
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
        if self.config.show_attributes and attributes:
            attr_str = FaceAttributeAnalyzer.format(
                attributes.get("age", 0), attributes.get("gender", 0),
                attributes.get("emotion"), attributes.get("pose"))
            label = f"{name}\n{attr_str}"
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
        """[V3 FIX] Added missing method for non-person class annotations."""
        label = f"cls:{int(cls)} id:{int(tid)} {float(conf):.2f}"
        annotator.box_label(box, label=label, color=(200, 200, 200))

    def __call__(self, im0: np.ndarray) -> SolutionResults:
        self._current_frame = im0
        self.stats["frames_processed"] += 1

        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)

        person_ids, person_boxes = self._extract_person_data()
        self._pose_keypoints = self._extract_pose_keypoints()

        # --- [V3 FIX] Restructured face detection + new-person logic ---
        # Step 1: Initial face detection (only if frame controller says so)
        initial_run_fr = self.frame_controller.should_run_face_recognition([])
        face_boxes, face_embeddings, face_attributes = [], [], []
        if initial_run_fr:
            self.stats["fr_runs"] += 1
            face_boxes, face_embeddings, _, face_attributes = self.face_detector.detect(im0)
            self.stats["total_faces_detected"] += len(face_boxes)

        # Step 2: Build face-to-person mapping for new person detection
        face_to_person = self._build_face_person_map(face_boxes, person_boxes)
        # Build track_id -> embedding for registry lookup
        track_to_embedding = {}
        for fi, pi in face_to_person.items():
            if pi < len(person_ids) and fi < len(face_embeddings):
                track_to_embedding[person_ids[pi]] = face_embeddings[fi]

        # Step 3: Detect new persons
        new_person_ids = self.new_person_tracker.update(person_ids, track_to_embedding)
        if new_person_ids:
            self.stats["new_person_triggers"] += len(new_person_ids)

        # Step 4: [V3 FIX] If new persons detected but FR was not run, run it now
        run_fr = initial_run_fr
        if self.config.enable_new_person_detection and new_person_ids and not initial_run_fr:
            face_boxes, face_embeddings, _, face_attributes = self.face_detector.detect(im0)
            self.stats["total_faces_detected"] += len(face_boxes)
            self.stats["fr_runs"] += 1
            run_fr = True
            # Rebuild mappings
            face_to_person = self._build_face_person_map(face_boxes, person_boxes)
            track_to_embedding.clear()
            for fi, pi in face_to_person.items():
                if pi < len(person_ids) and fi < len(face_embeddings):
                    track_to_embedding[person_ids[pi]] = face_embeddings[fi]

        # --- Process each person ---
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
                person_id, person_idx, face_boxes, face_embeddings,
                face_attributes, face_to_person, run_fr)

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

            detected_persons.append({
                "id": person_id, "persistent_id": persistent_id,
                "name": name, "is_known": is_known,
                "box": person_box, "attributes": attrs,
            })

        # Alarm handling with debouncing + hysteresis
        alarm_threshold = self.CFG.get("records", 1)
        if unknown_count >= alarm_threshold:
            self._trigger_alarm(unknown_count)
            if self._alarm_active:
                self._save_alarm_screenshots(detected_persons)
                self._log_event("ALARM", {"unknown_count": unknown_count,
                                          "persons": detected_persons})
        else:
            self._clear_alarm()

        # [V3] Periodic registry eviction
        if self.stats["frames_processed"] % self.config.registry_cleanup_interval == 0:
            evicted = self.person_registry.remove_old(self.config.registry_ttl)
            if evicted:
                self.stats["registry_evictions"] += evicted

        output_frame = annotator.result()
        self.display_output(output_frame)

        # Periodic stats logging
        if self.stats["frames_processed"] % 30 == 0:
            self._logger.info(
                f"Frames:{self.stats['frames_processed']} FR:{self.stats['fr_runs']} "
                f"Cache:{self.stats['cache_hits']} Faces:{self.stats['total_faces_detected']} "
                f"Registry:{len(self.person_registry)} "
                f"Matches:{self.stats['registry_matches']}")

        # HUD overlay
        stats_text = (f"Tracks:{len(person_ids)} Known:{self.stats['known_persons_detected']} "
                      f"Unknown:{self.stats['unknown_persons_detected']}")
        cv2.putText(output_frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status = (f"FR:{'ON' if run_fr else 'OFF'} Face:{len(face_boxes)} "
                  f"Reg:{len(self.person_registry)}")
        cv2.putText(output_frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return SolutionResults(plot_im=output_frame, total_tracks=len(self.track_ids))

    def save_screenshots(self) -> int:
        return self.screenshot_capturer.save_all() if self.screenshot_capturer else 0

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


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_security_guard(config: Optional[SecurityGuardConfig] = None,
                          face_directory: Optional[str] = None,
                          **kwargs) -> EnhancedSecurityGuard:
    """
    Create a Security Guard instance.
    Loads known faces from face_directory/person_name/*.jpg -> dict of embeddings.
    """
    cfg = config or SecurityGuardConfig()
    logger = _setup_logging("Factory")
    detector = InsightFaceDetector(cfg.insightface_model, cfg.insightface_det_size)

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

    return EnhancedSecurityGuard(config=cfg, known_embeddings_dict=embeddings_dict, **kwargs)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    config = SecurityGuardConfig(
        frame_interval=10,
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
    )

    print("\n" + "=" * 60)
    print("Enhanced Security Guard V3 - YOLO + InsightFace + Re-ID")
    print("=" * 60 + "\n")

    video_path = "../media_files/WIN_20260227_22_00_29_Pro.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "output_reid_v3.avi"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    guard = create_security_guard(
        config=config, 
        show=True, 
        model="yolo26n-pose.pt",
        classes=[0], 
        # vision_point=(width // 2 - 250, height - 10),
        vision_point=(width // 2 - 250, height - 350),
        # vision_point=(width // 2 + 200, height - 350),
        conf=0.6,
        iou=0.8, 
        records=1,
    )

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        result = guard(frame)
        writer.write(result.plot_im)

    print("\n[INFO] Saving screenshots...")
    saved = guard.save_screenshots()
    print(f"[INFO] Saved {saved} screenshots")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, value in guard.stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)
