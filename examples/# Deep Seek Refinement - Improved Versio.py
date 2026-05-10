# Deep Seek Refinement - Improved Version

"""
Enhanced InsightFace Security Guard System - Optimized Version with Person Re-Identification

This module provides an advanced security system using:
- YOLO for person detection, tracking, and pose estimation
- InsightFace for face detection, recognition, and attributes
- Face tracking, liveness detection, quality assessment
- Face attributes: age, gender, emotion, pose

Features:
1. Person Detection via YOLO (including pose keypoints)
2. Face Detection via InsightFace RetinaFace
3. Face Recognition (known vs unknown)
4. Face Attributes (age, gender, emotion, pose)
5. Face Tracking across frames
6. Liveness Detection
7. Quality Assessment
8. Pose Detection Integration
9. **Person Re-Identification** (face-based persistent identity across track ID changes)
10. **Multiple Embeddings per Person** for better matching
11. **Alarm Debouncing** to reduce false alarms
12. **Deduplicated Screenshots** using persistent ID

Installation:
    pip install insightface onnxruntime opencv-python numpy pygame

Version: 2.2.0
"""

# ============================================================================
# IMPORTS
# ============================================================================

import cv2
import numpy as np
import pygame
import os
import time
import json
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from threading import Lock
from contextlib import contextmanager
import copy
import warnings
import pickle

# Third-party imports
from insightface.app import FaceAnalysis
from ultralytics import solutions
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors
from ultralytics.utils import LOGGER

# ============================================================================
# CONSTANTS
# ============================================================================

# Quality assessment weights
SIZE_WEIGHT = 0.30
BRIGHTNESS_WEIGHT = 0.15
SHARPNESS_WEIGHT = 0.25
CONTRAST_WEIGHT = 0.20

# Quality thresholds
DEFAULT_MIN_FACE_QUALITY = 30.0
DEFAULT_MIN_FACE_SIZE = 40
DEFAULT_IOU_THRESHOLD = 0.8
DEFAULT_FACE_TRACK_MAX_AGE = 30

# Performance tuning
NORMALIZATION_EPS = 1e-5
DEFAULT_CACHE_TTL = 30.0

# ============================================================================
# ENHANCED FACE DETECTION CONFIGURATION (Addresses Issues 1, 2, 3)
# ============================================================================


@dataclass
class EnhancedFaceDetectionConfig:
    """
    Configuration for enhanced face detection pipeline.

    Addresses three main issues:
    1. FALSE POSITIVES: min_face_aspect_ratio, enable_landmark_validation, enable_nms
    2. DUPLICATES: enable_embedding_dedup, embedding_similarity_threshold
    3. UNKNOWN CONSISTENCY: enable_unknown_registry, unknown_similarity_threshold
    """

    # Basic detection
    min_face_size: Tuple[int, int] = (40, 40)

    # ISSUE 1: False positive reduction
    min_face_aspect_ratio: float = 0.6
    max_face_aspect_ratio: float = 1.0
    min_detection_confidence: float = 0.5
    enable_landmark_validation: bool = True
    landmark_distance_threshold: float = 0.3
    enable_nms: bool = True
    nms_iou_threshold: float = 0.3

    # ISSUE 2: Duplicate elimination
    enable_embedding_dedup: bool = True
    embedding_similarity_threshold: float = 0.9

    # ISSUE 3: Unknown consistency
    enable_unknown_registry: bool = True
    unknown_registry_path: str = "unknown_faces_registry.pkl"
    max_unknown_faces_stored: int = 1000
    unknown_similarity_threshold: float = 0.75
    unknown_face_ttl_days: int = 30


# ============================================================================
# UTILITY FUNCTIONS FOR ENHANCED DETECTION
# ============================================================================


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between embeddings."""
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
    return float(np.dot(emb1, emb2))


# ============================================================================
# ISSUE 1: FALSE POSITIVE REDUCTION - FACE GEOMETRY VALIDATOR
# ============================================================================


class FaceGeometryValidator:
    """
    Validates face detections using geometric constraints.
    Addresses FALSE POSITIVE DETECTIONS by:
    1. Checking aspect ratio (width/height between 0.6-1.0)
    2. Validating facial landmark positions
    """

    def __init__(self, config: EnhancedFaceDetectionConfig):
        self.config = config
        self.logger = _setup_logging("FaceGeometryValidator")

    def validate_aspect_ratio(self, box: List[float]) -> bool:
        """Validate face aspect ratio is within valid range."""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if height <= 0:
            return False
        aspect_ratio = width / height
        return self.config.min_face_aspect_ratio <= aspect_ratio <= self.config.max_face_aspect_ratio

    def validate_landmarks(self, kps: np.ndarray) -> bool:
        """Validate facial landmarks form valid structure."""
        if kps is None or len(kps) < 5:
            return False
        try:
            left_eye, right_eye = kps[0], kps[1]
            nose, left_mouth, right_mouth = kps[2], kps[3], kps[4]
            eye_distance = np.linalg.norm(right_eye - left_eye)

            # Eye level check
            eye_level_diff = abs(left_eye[1] - right_eye[1]) / eye_distance
            if eye_level_diff > self.config.landmark_distance_threshold * 2:
                return False

            # Nose below eyes
            if nose[1] < min(left_eye[1], right_eye[1]):
                return False

            # Mouth below nose
            mouth_center = (left_mouth + right_mouth) / 2
            if mouth_center[1] < nose[1]:
                return False

            return True
        except:
            return False

    def validate_detection(
        self, box: List[float], kps: Optional[np.ndarray] = None, confidence: Optional[float] = None
    ) -> bool:
        """Validate detection passes all checks."""
        # Size check
        x1, y1, x2, y2 = box
        if (x2 - x1) < self.config.min_face_size[0] or (y2 - y1) < self.config.min_face_size[1]:
            return False
        # Aspect ratio
        if not self.validate_aspect_ratio(box):
            return False
        # Confidence
        if confidence is not None and confidence < self.config.min_detection_confidence:
            return False
        # Landmark validation
        if self.config.enable_landmark_validation and kps is not None:
            if not self.validate_landmarks(kps):
                return False
        return True


# ============================================================================
# ISSUE 1 & 2: NMS DEDUPLICATION
# ============================================================================


class FaceNMSDeduplicator:
    """
    Applies Non-Maximum Suppression to remove overlapping detections.
    Addresses DUPLICATE CROPPING by removing spatially overlapping faces.
    """

    def __init__(self, config: EnhancedFaceDetectionConfig):
        self.config = config
        self.logger = _setup_logging("FaceNMS")

    def apply_nms(self, boxes: List[List[float]], confidences: Optional[List[float]] = None) -> List[int]:
        """Apply NMS and return indices of boxes to keep."""
        if not boxes:
            return []
        n = len(boxes)
        confidences = confidences or [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        indices = sorted(range(n), key=lambda i: confidences[i], reverse=True)
        keep = []
        while indices:
            current = indices[0]
            keep.append(current)
            if len(indices) == 1:
                break
            new_indices = []
            for idx in indices[1:]:
                iou = compute_iou(boxes[current], boxes[idx])
                if iou <= self.config.nms_iou_threshold:
                    new_indices.append(idx)
            indices = new_indices
        return keep


# ============================================================================
# ISSUE 2: EMBEDDING-BASED FACE DEDUPLICATION
# ============================================================================


class EmbeddingFaceDeduplicator:
    """
    Deduplicates faces using embedding similarity.
    Addresses DUPLICATE CROPPING by grouping similar embeddings.
    """

    def __init__(self, config: EnhancedFaceDetectionConfig):
        self.config = config
        self.logger = _setup_logging("EmbeddingDeduplicator")

    def find_duplicates(
        self, embeddings: List[np.ndarray], quality_scores: Optional[List[float]] = None
    ) -> List[List[int]]:
        """Find groups of duplicate faces based on embedding similarity."""
        if not embeddings or len(embeddings) <= 1:
            return []
        n = len(embeddings)
        quality_scores = quality_scores or [1.0] * n
        assigned = [False] * n
        duplicate_groups = []
        for i in range(n):
            if assigned[i]:
                continue
            current_group = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                similarity = compute_cosine_similarity(embeddings[i], embeddings[j])
                if similarity >= self.config.embedding_similarity_threshold:
                    current_group.append(j)
                    assigned[j] = True
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
        return duplicate_groups

    def get_best_face_index(self, group: List[int], quality_scores: List[float]) -> int:
        """Get best face from group based on quality."""
        if not group:
            return -1
        best_idx = group[0]
        best_score = quality_scores[group[0]]
        for idx in group:
            if quality_scores[idx] > best_score:
                best_score = quality_scores[idx]
                best_idx = idx
        return best_idx


# ============================================================================
# ISSUE 3: UNKNOWN FACE REGISTRY (PERSISTENT CROSS-SESSION)
# ============================================================================


@dataclass
class UnknownFaceEntry:
    """Entry for unknown face in persistent registry."""

    embedding: np.ndarray
    first_seen: float
    last_seen: float
    assigned_id: str = ""
    session_count: int = 1

    def __post_init__(self):
        if not self.assigned_id:
            self.assigned_id = f"unknown_{int(self.first_seen)}"


class UnknownFaceRegistry:
    """
    Persistent registry for unknown face embeddings.
    Addresses INCONSISTENT UNKNOWN HANDLING by:
    - Storing unknown face embeddings persistently
    - Matching new unknowns against stored ones
    - Cross-session identity for unknown persons
    """

    def __init__(self, config: EnhancedFaceDetectionConfig):
        self.config = config
        self.logger = _setup_logging("UnknownFaceRegistry")
        self._entries: Dict[str, UnknownFaceEntry] = {}
        self._session_unknown_ids: Dict[int, str] = {}
        self._lock = Lock()
        if config.enable_unknown_registry:
            self._load_registry()

    def _get_registry_path(self) -> Path:
        base_dir = _get_base_dir()
        return base_dir / self.config.unknown_registry_path

    def _load_registry(self) -> None:
        """Load registry from persistent storage."""
        path = self._get_registry_path()
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                entries_data = data.get("entries", {})
                ttl = self.config.unknown_face_ttl_days * 24 * 3600
                now = time.time()
                for eid, ed in entries_data.items():
                    if now - ed["last_seen"] <= ttl:
                        self._entries[eid] = UnknownFaceEntry(
                            embedding=np.array(ed["embedding"]),
                            first_seen=ed["first_seen"],
                            last_seen=ed["last_seen"],
                            assigned_id=ed["assigned_id"],
                            session_count=ed.get("session_count", 1),
                        )
                self.logger.info(f"Loaded {len(self._entries)} unknown faces")
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to persistent storage."""
        if not self.config.enable_unknown_registry:
            return
        path = self._get_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "entries": {
                    eid: {
                        "embedding": e.embedding.tolist(),
                        "first_seen": e.first_seen,
                        "last_seen": e.last_seen,
                        "assigned_id": e.assigned_id,
                        "session_count": e.session_count,
                    }
                    for eid, e in self._entries.items()
                }
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")

    def find_matching_unknown(
        self, embedding: np.ndarray, min_similarity: Optional[float] = None
    ) -> Optional[Tuple[str, float]]:
        """Find matching unknown face in registry."""
        threshold = min_similarity or self.config.unknown_similarity_threshold
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        best_match, best_sim = None, threshold
        for eid, entry in self._entries.items():
            sim = compute_cosine_similarity(emb_norm, entry.embedding)
            if sim > best_sim:
                best_sim, best_match = sim, eid
        return (best_match, best_sim) if best_match else None

    def register_unknown(self, embedding: np.ndarray, track_id: Optional[int] = None) -> str:
        """Register new unknown or update existing."""
        with self._lock:
            now = time.time()
            match = self.find_matching_unknown(embedding)
            if match:
                eid, sim = match
                entry = self._entries[eid]
                entry.last_seen = now
                entry.session_count += 1
                # Temporal smoothing
                entry.embedding = 0.7 * entry.embedding + 0.3 * (embedding / (np.linalg.norm(embedding) + 1e-8))
                unknown_id = entry.assigned_id
            else:
                if len(self._entries) >= self.config.max_unknown_faces_stored:
                    oldest = min(self._entries.keys(), key=lambda k: self._entries[k].last_seen)
                    del self._entries[oldest]
                emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
                entry = UnknownFaceEntry(embedding=emb_norm, first_seen=now, last_seen=now)
                self._entries[entry.assigned_id] = entry
                unknown_id = entry.assigned_id
            if track_id is not None:
                self._session_unknown_ids[track_id] = unknown_id
            self._save_registry()
            return unknown_id

    def get_session_unknown_id(self, track_id: int) -> Optional[str]:
        return self._session_unknown_ids.get(track_id)

    def clear_session(self) -> None:
        with self._lock:
            self._session_unknown_ids.clear()

    def get_statistics(self) -> Dict:
        return {"total_unknowns": len(self._entries), "session_tracked": len(self._session_unknown_ids)}


# ============================================================================
# EMOTION AND GENDER MAPPINGS (InsightFace Labels)
# ============================================================================

"""
Emotion and Gender mapping definitions for InsightFace face attributes.

InsightFace returns numeric indices for emotion and gender predictions:
- Emotion: 0=happy, 1=neutral, 2=sad, 3=angry, 4=surprise, 5=fear, 6=disgust
- Gender: 0=male, 1=female

This module provides type-safe mappings with validation and reverse lookup.
"""

import enum
from typing import Dict, Optional, Tuple
from types import MappingProxyType


class Emotion(enum.IntEnum):
    """InsightFace emotion labels as type-safe enum."""

    HAPPY = 0
    NEUTRAL = 1
    SAD = 2
    ANGRY = 3
    SURPRISE = 4
    FEAR = 5
    DISGUST = 6


class Gender(enum.IntEnum):
    """InsightFace gender labels as type-safe enum."""

    MALE = 0
    FEMALE = 1


# Immutable emotion mapping (index -> label)
EMOTION_MAP: Dict[int, str] = MappingProxyType(
    {0: "happy", 1: "neutral", 2: "sad", 3: "angry", 4: "surprise", 5: "fear", 6: "disgust"}
)

# Immutable gender mapping (index -> label)
GENDER_MAP: Dict[int, str] = MappingProxyType({0: "male", 1: "female"})

# Reverse mappings for lookup (label -> index)
EMOTION_TO_INDEX: Dict[str, int] = MappingProxyType({v: k for k, v in EMOTION_MAP.items()})
GENDER_TO_INDEX: Dict[str, int] = MappingProxyType({v: k for k, v in GENDER_MAP.items()})

# Valid index ranges for validation
EMOTION_VALID_INDICES: Tuple[int, int] = (0, 6)
GENDER_VALID_INDICES: Tuple[int, int] = (0, 1)


def get_emotion_label(index: int) -> str:
    """
    Get emotion label from numeric index with validation.

    Args:
        index: Emotion index from InsightFace (0-6)

    Returns:
        Emotion label string (e.g., "happy", "sad", "angry")

    Raises:
        ValueError: If index is outside valid range [0, 6]
    """
    if not EMOTION_VALID_INDICES[0] <= index <= EMOTION_VALID_INDICES[1]:
        raise ValueError(f"Invalid emotion index {index}. Must be in range {EMOTION_VALID_INDICES}")
    return EMOTION_MAP[index]


def get_gender_label(index: int) -> str:
    """
    Get gender label from numeric index with validation.

    Args:
        index: Gender index from InsightFace (0=male, 1=female)

    Returns:
        Gender label string ("male" or "female")

    Raises:
        ValueError: If index is outside valid range [0, 1]
    """
    if not GENDER_VALID_INDICES[0] <= index <= GENDER_VALID_INDICES[1]:
        raise ValueError(f"Invalid gender index {index}. Must be in range {GENDER_VALID_INDICES}")
    return GENDER_MAP[index]


def get_emotion_index(label: str) -> Optional[int]:
    """
    Reverse lookup: Get emotion index from label string.

    Args:
        label: Emotion label string (e.g., "happy", "sad")

    Returns:
        Emotion index or None if not found
    """
    return EMOTION_TO_INDEX.get(label.lower())


def get_gender_index(label: str) -> Optional[int]:
    """
    Reverse lookup: Get gender index from label string.

    Args:
        label: Gender label string ("male" or "female")

    Returns:
        Gender index or None if not found
    """
    return GENDER_TO_INDEX.get(label.lower())


def validate_emotion_index(index: int) -> bool:
    """Check if emotion index is valid."""
    return EMOTION_VALID_INDICES[0] <= index <= EMOTION_VALID_INDICES[1]


def validate_gender_index(index: int) -> bool:
    """Check if gender index is valid."""
    return GENDER_VALID_INDICES[0] <= index <= GENDER_VALID_INDICES[1]


# ============================================================================
# LOGGING SETUP
# ============================================================================


def _setup_logging(name: str = "SecurityGuard") -> logging.Logger:
    """Setup module logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ============================================================================
# BASE DIRECTORY HELPER
# ============================================================================


def _get_base_dir() -> Path:
    """Get the base directory of the project, handling both script and notebook modes."""
    try:
        return Path(__file__).parent.parent
    except NameError:
        return Path.cwd()


# ============================================================================
# SOUND MANAGER
# ============================================================================


class SoundManager:
    """Thread-safe sound playback manager for security alerts."""

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
            self._initialized_pygame = False
            self._current_sound = None
            self._logger = _setup_logging("SoundManager")
            self._sound_lock = Lock()
            self._load_alarm_sound()

    def _load_alarm_sound(self) -> None:
        """Load alarm sound file with error handling."""
        try:
            base_dir = _get_base_dir()
            alarm_file = base_dir / "../media_files/Alarm-sound-samples/humordome-security-alert-sound-453297.mp3"

            if alarm_file.exists():
                pygame.mixer.init()
                self._initialized_pygame = True
                pygame.mixer.music.load(str(alarm_file))
                self._alarm_loaded = True
                self._logger.info(f"Alarm sound loaded: {alarm_file}")
            else:
                self._logger.warning(f"Alarm file not found: {alarm_file}")
        except Exception as e:
            self._logger.warning(f"Failed to load alarm: {e}")

    @contextmanager
    def _pygame_context(self):
        """Context manager for pygame operations."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            yield
        except Exception as e:
            self._logger.error(f"Pygame error: {e}")

    def play_alarm(self) -> bool:
        """Play alarm sound if not already playing."""
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
        """Stop currently playing alarm."""
        with self._sound_lock:
            try:
                if self._alarm_loaded and pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
            except Exception as e:
                self._logger.debug(f"Stop alarm error: {e}")

    def is_loaded(self) -> bool:
        """Check if alarm sound is loaded."""
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

        # Ensure log file directory exists
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else ".", exist_ok=True)

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event with timestamp."""
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
    frame_interval: int = 3
    face_recognition_interval: float = 0.0

    # Detection settings
    enable_new_person_detection: bool = True
    person_cache_ttl: float = DEFAULT_CACHE_TTL  # legacy, kept for compatibility
    face_tolerance: float = 0.4
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
    enable_liveness: bool = True
    show_attributes: bool = True

    # Display colors (BGR format for OpenCV)
    known_color: Tuple[int, int, int] = (0, 255, 0)  # Green for known persons
    unknown_color: Tuple[int, int, int] = (0, 165, 255)  # Orange for unknown persons
    no_face_color: Tuple[int, int, int] = (128, 128, 128)  # Gray when no face detected

    # Screenshot settings
    capture_faces: bool = True
    screenshot_dir: Optional[str] = None
    min_face_quality: float = DEFAULT_MIN_FACE_QUALITY

    # Feature flags
    enable_logging: bool = True
    enable_keypoints_extraction: bool = False
    enable_keypoints_display: bool = False

    # Performance optimizations
    min_recognition_quality: float = 30.0  # skip FR if face quality below this
    max_face_yaw: float = 45.0  # skip FR if head turned beyond this angle (degrees)
    known_cache_ttl: float = 60.0  # cache duration for known persons
    unknown_cache_ttl: float = 5.0  # cache duration for unknown persons
    min_confidence: float = 0.2  # minimum similarity to accept a match

    # New: Person re-identification settings
    reid_similarity_threshold: float = 0.2  # threshold for matching against registry
    registry_max_embeddings_per_person: int = 5  # number of recent embeddings to keep
    alarm_debounce_frames: int = 3  # consecutive frames with unknown >= threshold to trigger alarm

    def validate(self) -> "SecurityGuardConfig":
        """Validate configuration parameters."""
        if self.frame_interval < 0:
            raise ValueError("frame_interval must be non-negative")

        if not 0.3 <= self.face_tolerance <= 0.7:
            warnings.warn("face_tolerance should be between 0.3 and 0.7")

        if self.person_cache_ttl <= 0:
            raise ValueError("person_cache_ttl must be positive")

        if self.min_face_size[0] <= 0 or self.min_face_size[1] <= 0:
            raise ValueError("min_face_size must have positive dimensions")

        return self


# ============================================================================
# PERSON REGISTRY (for face-based persistent identity)
# ============================================================================


@dataclass
class RegistryEntry:
    """Entry in the persistent person registry."""

    persistent_id: int
    name: str
    category: str  # "KNOWN" or "UNKNOWN"
    embeddings: List[np.ndarray]  # list of recent embeddings (normalized)
    last_seen: float
    attributes: Dict  # last known attributes


class PersonRegistry:
    """
    Maintains a persistent identity database based on face embeddings.
    Allows matching a new face embedding to a previously seen person,
    even if the YOLO track ID changes.
    """

    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self.entries: Dict[int, RegistryEntry] = {}  # persistent_id -> entry
        self.next_id = 0
        self._lock = Lock()
        self._logger = _setup_logging("PersonRegistry")

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        return emb / (np.linalg.norm(emb) + NORMALIZATION_EPS)

    def find_match(self, embedding: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Find the best matching persistent ID for the given embedding.
        Returns (persistent_id, similarity) or None.
        """
        embedding_norm = self._normalize(embedding)
        best_id = None
        best_sim = 0.0
        with self._lock:
            for pid, entry in self.entries.items():
                for stored_emb in entry.embeddings:
                    sim = np.dot(embedding_norm, stored_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = pid
        if best_sim >= self.config.reid_similarity_threshold:
            return best_id, best_sim
        return None

    def add_person(self, embedding: np.ndarray, name: str, category: str, attributes: Dict) -> int:
        """Add a new person to the registry and return persistent ID."""
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
        self._logger.debug(f"Added new person to registry: {name} ({category}) with ID {pid}")
        return pid

    def update_person(self, persistent_id: int, embedding: np.ndarray, attributes: Dict = None):
        """Update an existing person with a new embedding and refresh last_seen."""
        embedding_norm = self._normalize(embedding)
        with self._lock:
            if persistent_id in self.entries:
                entry = self.entries[persistent_id]
                # Keep a rolling window of embeddings
                if len(entry.embeddings) >= self.config.registry_max_embeddings_per_person:
                    entry.embeddings.pop(0)
                entry.embeddings.append(embedding_norm)
                entry.last_seen = time.time()
                if attributes:
                    entry.attributes.update(attributes)
                self._logger.debug(f"Updated person {persistent_id} ({entry.name})")

    def get_info(self, persistent_id: int) -> Optional[Tuple[str, str, Dict]]:
        """Retrieve (name, category, attributes) for a persistent ID."""
        with self._lock:
            entry = self.entries.get(persistent_id)
            if entry:
                return entry.name, entry.category, entry.attributes
        return None

    def remove_old(self, max_age: float):
        """Remove entries not seen for longer than max_age (seconds)."""
        now = time.time()
        with self._lock:
            to_remove = [pid for pid, e in self.entries.items() if now - e.last_seen > max_age]
            for pid in to_remove:
                del self.entries[pid]
                self._logger.debug(f"Removed old person {pid} from registry")


# ============================================================================
# FACE SCREENSHOT CAPTURER (modified to use persistent ID)
# ============================================================================


@dataclass
class FaceData:
    """Data class for storing face information."""

    persistent_id: int  # now store persistent ID instead of track ID
    box: List[float]
    quality: float
    frame: np.ndarray
    is_known: bool
    name: str
    timestamp: float = field(default_factory=time.time)


class FaceScreenshotCapturer:
    """Captures and saves face screenshots with quality assessment and deduplication based on persistent ID."""

    def __init__(self, output_dir: str = "../captured_faces", min_quality: float = DEFAULT_MIN_FACE_QUALITY):
        self.output_dir = output_dir
        self.min_quality = min_quality

        self.known_dir = os.path.join(output_dir, "known")
        self.unknown_dir = os.path.join(output_dir, "unknown")

        # Create directories
        for d in [self.known_dir, self.unknown_dir]:
            os.makedirs(d, exist_ok=True)

        # Thread-safe data structures
        self.best_faces: Dict[int, FaceData] = {}  # key = persistent_id
        self._unknown_count = 0
        self._captured_persistent_ids: set = set()  # persistent IDs already captured
        self._lock = Lock()

        self._logger = _setup_logging("ScreenshotCapturer")
        self._logger.info(f"Screenshots directory: {output_dir}")

    def assess_quality(self, frame: np.ndarray, box: List[float]) -> float:
        """Assess face quality based on multiple factors."""
        try:
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]

            # Clamp coordinates to frame bounds
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return 0.0

            # Calculate size score
            face_area = (x2 - x1) * (y2 - y1)
            frame_area = w * h
            size_score = min(100, (face_area / frame_area) * 1000)

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Calculate brightness score
            brightness = np.mean(gray)
            bright_score = max(0, 100 - abs(brightness - 128) * 0.8)

            # Calculate sharpness using Laplacian variance
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            sharp_score = min(100, lap.var() / 10)

            # Calculate contrast score
            contrast_score = min(100, np.std(gray) / 30 * 100)

            # Weighted combination
            return (
                size_score * SIZE_WEIGHT
                + bright_score * BRIGHTNESS_WEIGHT
                + sharp_score * SHARPNESS_WEIGHT
                + contrast_score * CONTRAST_WEIGHT
            )

        except Exception:
            return 0.0

    def update_best_face(
        self, persistent_id: int, box: List[float], frame: np.ndarray, is_known: bool, name: str = "Unknown"
    ) -> bool:
        """
        Update the best face for a persistent person ID.
        Returns True if face was updated, False if quality too low.
        """
        quality = self.assess_quality(frame, box)

        if quality < self.min_quality:
            return False

        with self._lock:
            if persistent_id not in self.best_faces or quality > self.best_faces[persistent_id].quality:
                self.best_faces[persistent_id] = FaceData(
                    persistent_id=persistent_id,
                    box=box.copy(),
                    quality=quality,
                    frame=frame.copy(),
                    is_known=is_known,
                    name=name,
                )
                return True
        return False

    def save_all(self) -> int:
        """Save all captured faces to disk."""
        saved_count = 0
        with self._lock:
            for pid, data in self.best_faces.items():
                try:
                    path = self._save_face_image(pid, data)
                    if path:
                        saved_count += 1
                        self._logger.info(f"Saved face: {path}")
                except Exception as e:
                    self._logger.error(f"Save failed for persistent ID {pid}: {e}")

        return saved_count

    def _save_face_image(self, persistent_id: int, data: FaceData) -> Optional[str]:
        """Internal method to save a face image."""
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
            filename = f"unknown_{self._unknown_count}_{data.quality:.0f}_{timestamp}.jpg"
            path = os.path.join(self.unknown_dir, filename)

        success = cv2.imwrite(path, face)
        return path if success else None

    def save_unknown_face(
        self,
        frame: np.ndarray,
        box: List[float],
        persistent_id: Optional[int] = None,
        is_known: bool = False,
        name: str = "Unknown",
    ) -> Optional[str]:
        """
        Save a face screenshot with deduplication based on persistent ID.
        """
        # Deduplication check
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
                    if persistent_id is not None:
                        filename = f"unknown_{persistent_id}_{timestamp}.jpg"
                    else:
                        filename = f"unknown_{self._unknown_count}_{timestamp}.jpg"
                    path = os.path.join(self.unknown_dir, filename)

            success = cv2.imwrite(path, face)
            if success:
                self._logger.info(f"Saved {'known' if is_known else 'unknown'} face: {path}")
                return path

        except Exception as e:
            self._logger.error(f"Save face failed: {e}")

        return None

    def reset_captured_ids(self) -> None:
        """Reset captured persistent IDs to allow new captures."""
        with self._lock:
            self._captured_persistent_ids.clear()

    def get_captured_ids(self) -> set:
        """Get copy of captured persistent IDs."""
        with self._lock:
            return self._captured_persistent_ids.copy()

    def get_best_face(self, persistent_id: int) -> Optional[FaceData]:
        """Get the best face data for a persistent ID."""
        with self._lock:
            return self.best_faces.get(persistent_id)


# ============================================================================
# FACE ATTRIBUTE ANALYZER
# ============================================================================


class FaceAttributeAnalyzer:
    """Analyzes and formats face attributes for display."""

    EMOTION = EMOTION_MAP
    GENDER = GENDER_MAP

    @staticmethod
    def format(
        age: float,
        gender: float,
        emotion: Optional[np.ndarray] = None,
        pose: Optional[np.ndarray] = None,
        face_width: float = 0,
        face_height: float = 0,
    ) -> str:
        """
        Format face attributes into a display string.
        Includes: age, gender, emotion, pose, face box size.
        """
        age_str = f"{int(age)}y" if age is not None else "?y"
        # FIXED: Handle both integer (0=male, 1=female) and float probability
        # InsightFace returns gender as integer, but we handle both cases
        if gender is None:
            gender_str = "?"
        elif isinstance(gender, (int, np.integer)):
            # Integer: 0=male, 1=female
            gender_str = FaceAttributeAnalyzer.GENDER.get(int(gender), "?")
        elif isinstance(gender, (float, np.floating)):
            # Float probability: >0.5 means female
            gender_idx = int(gender > 0.5)
            gender_str = FaceAttributeAnalyzer.GENDER.get(gender_idx, "?")
        else:
            gender_str = "?"
        emo_idx = 1
        if emotion is not None and len(emotion) > 0:
            try:
                emo_idx = int(np.argmax(emotion))
            except Exception:
                pass
        emo_str = FaceAttributeAnalyzer.EMOTION.get(emo_idx, "😐")
        pose_str = ""
        if pose is not None and len(pose) >= 3:
            try:
                pitch, yaw, roll = pose[0], pose[1], pose[2]
                pose_str = f" | P:{int(pitch)}° Y:{int(yaw)}° R:{int(roll)}°"
            except Exception:
                pass
        # Add face box dimensions
        size_str = ""
        if face_width > 0 and face_height > 0:
            size_str = f" | Face:{int(face_width)}x{int(face_height)}"
        return f"{age_str} {gender_str} {emo_str}{pose_str}{size_str}"


# ============================================================================
# FACE TRACKER (unchanged, but currently unused)
# ============================================================================


class FaceTracker:
    """IoU-based face tracking with thread safety."""

    def __init__(self, max_age: int = DEFAULT_FACE_TRACK_MAX_AGE, iou_threshold: float = DEFAULT_IOU_THRESHOLD):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        self._lock = Lock()
        self._logger = _setup_logging("FaceTracker")

    def _compute_iou(self, b1: List[float], b2: List[float]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, boxes: List[List[float]], embeddings: List[np.ndarray], attributes: List[Dict]) -> Dict[int, Dict]:
        with self._lock:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["age"] += 1
            self.tracks = {k: v for k, v in self.tracks.items() if v["age"] < self.max_age}
            matched = set()
            for box, embed, attr in zip(boxes, embeddings, attributes):
                best_iou, best_id = 0, None
                for track_id, track_data in self.tracks.items():
                    if track_id in matched:
                        continue
                    iou = self._compute_iou(box, track_data["box"])
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou, best_id = iou, track_id
                if best_id is not None:
                    self.tracks[best_id] = {"box": box, "embedding": embed, "attributes": attr, "age": 0}
                    matched.add(best_id)
                else:
                    self.tracks[self.next_id] = {"box": box, "embedding": embed, "attributes": attr, "age": 0}
                    matched.add(self.next_id)
                    self.next_id += 1
            return {tid: data for tid, data in self.tracks.items() if tid in matched}

    def get_track(self, track_id: int) -> Optional[Dict]:
        with self._lock:
            return self.tracks.get(track_id)

    def reset(self) -> None:
        with self._lock:
            self.tracks.clear()
            self.next_id = 0


# ============================================================================
# PERSON CACHE (track ID based, now used as fallback)
# ============================================================================


@dataclass
class PersonCacheEntry:
    """Data class for cached person data (by track ID)."""

    name: str
    is_known: bool
    timestamp: float
    attributes: Dict
    confidence: float = 0.0


class PersonCache:
    """LRU cache for person recognition results with separate TTLs for known/unknown."""

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
# NEW PERSON TRACKER (modified to use registry)
# ============================================================================


class NewPersonTracker:
    """
    Tracks newly detected persons, but now uses the registry to avoid
    re-triggering for returning persons with new track IDs.
    """

    def __init__(self, registry: PersonRegistry):
        self._seen_track_ids: set = set()  # track IDs seen
        self._new_track_ids: set = set()  # track IDs considered new this session
        self._lock = Lock()
        self.registry = registry

    def update(self, person_ids: List[int], face_embeddings: Dict[int, np.ndarray]) -> List[int]:
        """
        Update with current person IDs and return new ones, but skip if the
        face matches an existing registry entry.
        """
        current = set(person_ids)
        new_track_ids = []
        with self._lock:
            for pid in current:
                if pid in self._seen_track_ids:
                    continue
                # Check if this person's face (if available) matches any in registry
                if pid in face_embeddings:
                    match = self.registry.find_match(face_embeddings[pid])
                    if match is not None:
                        # It's a returning person, so don't mark as new
                        self._seen_track_ids.add(pid)
                        continue
                # No match, it's a genuinely new person
                new_track_ids.append(pid)
                self._new_track_ids.add(pid)
            self._seen_track_ids.update(current)
            return new_track_ids

    def mark_processed(self, person_id: int) -> None:
        """Mark a person as processed (removes from new set)."""
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

    def should_run_face_recognition(self, new_person_ids: List[int]) -> bool:
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
# INSIGHTFACE DETECTOR (with GPU support)
# ============================================================================


class InsightFaceDetector:
    """Face detection and recognition using InsightFace."""

    def __init__(self, model: str = "buffalo_l", detection_size: Tuple[int, int] = (640, 640)):
        self.model_name = model
        self.detection_size = detection_size
        self._logger = _setup_logging("InsightFaceDetector")

        try:
            # Use GPU if available
            self.app = FaceAnalysis(name=model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=detection_size)
            self._logger.info(f"InsightFace initialized: {model} with GPU support")
        except Exception as e:
            self._logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Tuple[List[List[float]], List[np.ndarray], List, List[Dict]]:
        faces = self.app.get(frame)
        boxes, embeddings, landmarks, attributes = [], [], [], []
        for face in faces:
            bbox = face.bbox
            if (bbox[2] - bbox[0]) < DEFAULT_MIN_FACE_SIZE or (bbox[3] - bbox[1]) < DEFAULT_MIN_FACE_SIZE:
                continue
            boxes.append(bbox.tolist())
            embeddings.append(face.embedding)
            landmarks.append(face.kps)
            # Calculate face box dimensions
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]

            # Extract attributes with logging for debugging
            face_age = getattr(face, "age", None)
            face_gender = getattr(face, "gender", None)
            face_emotion = getattr(face, "emotion", None)
            face_pose = getattr(face, "pose", None)

            # Log attribute extraction for debugging
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    f"Face attributes - age: {face_age} (type: {type(face_age)}), "
                    f"gender: {face_gender} (type: {type(face_gender)}), "
                    f"emotion shape: {face_emotion.shape if face_emotion is not None else None}, "
                    f"pose: {face_pose}"
                )

            attributes.append(
                {
                    # Use getattr with defaults for all attributes including age and gender
                    # This fixes the issue where buffalo_l may return None for these attributes
                    "age": getattr(face, "age", None),
                    "gender": getattr(face, "gender", None),
                    "emotion": getattr(face, "emotion", np.array([0.5])),
                    "pose": getattr(face, "pose", np.array([0, 0, 0])),
                    "face_width": face_width,
                    "face_height": face_height,
                }
            )
        return boxes, embeddings, landmarks, attributes


# ============================================================================
# FACE RECOGNIZER (now supports multiple embeddings per person)
# ============================================================================


class FaceRecognizer:
    """Handles face recognition against known faces with multiple embeddings per person."""

    def __init__(
        self,
        known_embeddings_dict: Dict[str, List[np.ndarray]],  # name -> list of embeddings
        tolerance: float = 0.5,
        embedding_smoothing_frames: int = 5,
        min_confidence: float = 0.6,
    ):
        self.tolerance = tolerance
        self.embedding_smoothing_frames = embedding_smoothing_frames
        self.min_confidence = min_confidence
        self._logger = _setup_logging("FaceRecognizer")

        # Store normalized embeddings per person
        self.known_dict = {}
        for name, emb_list in known_embeddings_dict.items():
            norm_list = []
            for emb in emb_list:
                norm = emb / (np.linalg.norm(emb) + NORMALIZATION_EPS)
                norm_list.append(norm)
            self.known_dict[name] = norm_list

        if self.known_dict:
            self._logger.info(f"Loaded {len(self.known_dict)} known persons with multiple embeddings")
        else:
            self._logger.warning("No known faces loaded")

        # Temporal smoothing buffers for each track
        self._embedding_buffers: Dict[int, List[np.ndarray]] = {}
        self._identity_history: Dict[int, Tuple[str, float, int]] = {}

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
        """
        Returns: (name, is_known, confidence)
        """
        if not self.known_dict:
            return "Unknown", False, 0.0

        try:
            if track_id is not None:
                if track_id in self._identity_history:
                    name, conf, frames = self._identity_history[track_id]
                    if frames >= 3 and conf > 0.8:
                        return name, True, conf
                embedding = self._smooth_embedding(track_id, embedding)

            query_norm = embedding / (np.linalg.norm(embedding) + NORMALIZATION_EPS)

            best_name = "Unknown"
            best_score = 0.0
            for name, emb_list in self.known_dict.items():
                for stored_norm in emb_list:
                    sim = np.dot(query_norm, stored_norm)
                    if sim > best_score:
                        best_score = sim
                        best_name = name

            distance = 1 - best_score
            confidence = float(best_score)

            if distance < self.tolerance and confidence >= self.min_confidence:
                if track_id is not None:
                    frames_seen = len(self._embedding_buffers.get(track_id, []))
                    self._identity_history[track_id] = (best_name, confidence, frames_seen)
                return best_name, True, confidence

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
# SECURITY GUARD MAIN CLASS
# ============================================================================


class EnhancedSecurityGuard(solutions.VisionEye):
    """
    Main security guard class combining YOLO and InsightFace with persistent re-identification.
    """

    def __init__(
        self,
        *args,
        config: Optional[SecurityGuardConfig] = None,
        known_embeddings_dict: Optional[Dict[str, List[np.ndarray]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.config = config or SecurityGuardConfig()
        self.config.validate()
        self._logger = _setup_logging("SecurityGuard")

        self._initialize_components(known_embeddings_dict)
        self._initialize_stats()

        self._logger.info(f"Security Guard initialized with {len(known_embeddings_dict or {})} known persons")

    def _initialize_components(self, known_embeddings_dict: Optional[Dict[str, List[np.ndarray]]] = None):
        # Face detection
        self.face_detector = InsightFaceDetector(self.config.insightface_model, self.config.insightface_det_size)

        # Face recognition (with multiple embeddings per person)
        self.face_recognizer = FaceRecognizer(
            known_embeddings_dict or {},
            self.config.face_tolerance,
            self.config.embedding_smoothing_frames,
            self.config.min_confidence,
        )

        # Face tracking (optional, currently unused)
        self.face_tracker = FaceTracker(self.config.face_track_max_age, self.config.face_track_iou_threshold)

        # Person cache (track ID based, now secondary)
        self.person_cache = PersonCache(self.config)

        # Person registry (persistent identity)
        self.person_registry = PersonRegistry(self.config)

        # New person detection (uses registry)
        self.new_person_tracker = NewPersonTracker(self.person_registry)

        # Frame controller
        self.frame_controller = FrameController(self.config)

        # Event logging
        self.event_logger = EventLogger() if self.config.enable_logging else None

        # Screenshot capture
        self.screenshot_capturer = None
        if self.config.capture_faces:
            sd = self.config.screenshot_dir or str(_get_base_dir() / "captured_faces")
            self.screenshot_capturer = FaceScreenshotCapturer(sd, self.config.min_face_quality)

        # Sound manager
        self.sound_manager = SoundManager()
        self._alarm_active = False
        self._alarm_pending = 0  # for debouncing

    def _initialize_stats(self):
        self.stats = {
            "frames_processed": 0,
            "fr_runs": 0,
            "cache_hits": 0,
            "new_person_triggers": 0,
            "total_faces_detected": 0,
            "known_persons_detected": 0,
            "unknown_persons_detected": 0,
            "poses_detected": 0,
            "screenshots_saved": 0,
            "alerts_triggered": 0,
            "skipped_low_quality": 0,
            "skipped_yaw": 0,
            "registry_matches": 0,  # new stat
        }

    def _associate_face_to_person(self, face_box: List[float], person_boxes: List[List[float]]) -> Optional[int]:
        face_cx = (face_box[0] + face_box[2]) / 2
        face_cy = (face_box[1] + face_box[3]) / 2
        for idx, pbox in enumerate(person_boxes):
            if pbox[0] <= face_cx <= pbox[2] and pbox[1] <= face_cy <= pbox[3]:
                return idx
        return None

    def _process_person(
        self,
        person_id: int,
        person_box: List[float],
        face_boxes: List[List[float]],
        face_embeddings: List[np.ndarray],
        face_attributes: List[Dict],
        run_fr: bool,
    ) -> Tuple[str, bool, Dict, Optional[int]]:
        """
        Returns: (name, is_known, attributes, persistent_id)
        """
        # Try track-based cache first (fast)
        cached = self.person_cache.get(person_id)
        if cached:
            self.stats["cache_hits"] += 1
            return cached.name, cached.is_known, cached.attributes, None

        if not (run_fr and face_embeddings):
            return "Unknown", False, {}, None

        # Find associated face
        face_idx = self._associate_face_to_person(person_box, face_boxes)
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

        # First, try to match against persistent registry
        match = self.person_registry.find_match(embedding)
        if match is not None:
            persistent_id, sim = match
            self.stats["registry_matches"] += 1
            name, category, reg_attrs = self.person_registry.get_info(persistent_id)
            is_known = category == "KNOWN"
            # Update registry with new embedding and attributes
            self.person_registry.update_person(persistent_id, embedding, attrs)
            # Update track cache
            self.person_cache.set(person_id, name, is_known, reg_attrs)
            # Update screenshot if enabled
            if self.screenshot_capturer:
                self.screenshot_capturer.update_best_face(
                    persistent_id, face_boxes[face_idx], self._current_frame, is_known, name
                )
            return name, is_known, reg_attrs, persistent_id

        # No registry match, perform recognition
        name, is_known, confidence = self.face_recognizer.identify(embedding, track_id=person_id)
        if is_known:
            category = "KNOWN"
        else:
            category = "UNKNOWN"

        # Add to registry
        persistent_id = self.person_registry.add_person(embedding, name, category, attrs)

        # Update track cache
        self.person_cache.set(person_id, name, is_known, attrs)

        # Update screenshot
        if self.screenshot_capturer:
            self.screenshot_capturer.update_best_face(
                persistent_id, face_boxes[face_idx], self._current_frame, is_known, name
            )

        # Track pose detection
        if "pose" in attrs and attrs["pose"] is not None:
            self.stats["poses_detected"] += 1

        return name, is_known, attrs, persistent_id

    def _trigger_alarm(self, unknown_count: int) -> None:
        """Trigger alarm with debouncing."""
        self._alarm_pending += 1
        if self._alarm_pending >= self.config.alarm_debounce_frames and not self._alarm_active:
            if self.sound_manager.play_alarm():
                self._alarm_active = True
                self.stats["alerts_triggered"] += 1
                self._logger.warning(f"ALARM: {unknown_count} unknown person(s) detected")
        # If alarm already active, keep it active (don't reset pending)

    def _clear_alarm(self) -> None:
        self._alarm_pending = 0
        if self._alarm_active:
            self.sound_manager.stop_alarm()
            self._alarm_active = False

    def _save_alarm_screenshots(self, detected_persons: List[Dict]) -> None:
        if not self.screenshot_capturer:
            return
        for p in detected_persons:
            if p.get("persistent_id") is not None:
                best = self.screenshot_capturer.get_best_face(p["persistent_id"])
                if best:
                    path = self.screenshot_capturer.save_unknown_face(
                        self._current_frame,
                        best.box,
                        persistent_id=p["persistent_id"],
                        is_known=p["is_known"],
                        name=p["name"],
                    )
                    if path:
                        self.stats["screenshots_saved"] += 1

    def _log_event(self, event_type: str, data: Dict) -> None:
        if self.event_logger:
            self.event_logger.log(event_type, data)

    def _extract_person_data(self) -> Tuple[List[int], List[List[float]]]:
        person_ids, person_boxes = [], []
        for cls, tid, box in zip(self.clss, self.track_ids, self.boxes):
            if int(cls) == 0:
                person_ids.append(int(tid))
                person_boxes.append(box.tolist())
        return person_ids, person_boxes

    def _extract_pose_keypoints(self) -> Dict[int, List]:
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
        # Add YOLO confidence score to the label
        conf_str = f"{float(confidence) * 100:.1f}%" if confidence is not None else "N/A"
        label = name if is_known else "Unknown"
        if self.config.show_attributes and attributes:
            attr_str = FaceAttributeAnalyzer.format(
                attributes.get("age", 0),
                attributes.get("gender", 0),
                attributes.get("emotion"),
                attributes.get("pose"),
                attributes.get("face_width", 0),
                attributes.get("face_height", 0),
            )
            label = f"{name}\n{attr_str}\nConf:{conf_str}" if is_known else f"Unknown\n{attr_str}\nConf:{conf_str}"
        else:
            # Even without attributes, show confidence
            label = f"{label}\nConf:{conf_str}"
        return label

    def _annotate_person(
        self, annotator: SolutionAnnotator, box, label: str, person_id: int, person_box: List[float], color
    ) -> None:
        # Use the label directly (which now includes confidence and attributes from _build_label)
        # Add track ID to the label
        final_label = f"ID {person_id}: {label}" if label else f"ID {person_id}"
        annotator.box_label(box, label=final_label, color=color)
        annotator.visioneye(box, self.vision_point)
        if self.config.enable_keypoints_display and person_id in self._pose_keypoints:
            kpts = self._pose_keypoints[person_id]
            if kpts:
                kpts_array = np.array(kpts, dtype=np.float32)
                annotator.kpts(kpts_array, shape=self._current_frame.shape[:2], kpt_line=True)

    def __call__(self, im0: np.ndarray) -> SolutionResults:
        self._current_frame = im0
        self.stats["frames_processed"] += 1

        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)

        person_ids, person_boxes = self._extract_person_data()
        self._pose_keypoints = self._extract_pose_keypoints()

        # Determine if face recognition should run
        run_fr = self.frame_controller.should_run_face_recognition([])  # we'll handle new persons later

        face_boxes, face_embeddings, _, face_attributes = [], [], [], []
        if run_fr:
            self.stats["fr_runs"] += 1
            face_boxes, face_embeddings, _, face_attributes = self.face_detector.detect(im0)
            self.stats["total_faces_detected"] += len(face_boxes)

        # Build a mapping from track ID to its associated face embedding (if any)
        track_to_embedding = {}
        for box, emb in zip(face_boxes, face_embeddings):
            face_idx = self._associate_face_to_person(box, person_boxes)
            if face_idx is not None and face_idx < len(person_ids):
                track_to_embedding[person_ids[face_idx]] = emb

        # Detect new persons using registry
        new_person_ids = self.new_person_tracker.update(person_ids, track_to_embedding)
        if new_person_ids:
            self.stats["new_person_triggers"] += len(new_person_ids)

        # Update run_fr based on new persons (if enabled)
        if self.config.enable_new_person_detection and new_person_ids:
            run_fr = True
            # Need to re-run face detection if we didn't already? We'll assume we did if run_fr was True.
            # If run_fr was False but we have new persons, we should force face detection.
            if not run_fr and new_person_ids:
                # Force face detection now
                face_boxes, face_embeddings, _, face_attributes = self.face_detector.detect(im0)
                self.stats["total_faces_detected"] += len(face_boxes)
                # Rebuild track_to_embedding
                track_to_embedding.clear()
                for box, emb in zip(face_boxes, face_embeddings):
                    face_idx = self._associate_face_to_person(box, person_boxes)
                    if face_idx is not None and face_idx < len(person_ids):
                        track_to_embedding[person_ids[face_idx]] = emb
                run_fr = True
                self.stats["fr_runs"] += 1

        unknown_count = 0
        detected_persons = []

        for cls, tid, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            if int(cls) != 0:
                self._annotate_object(annotator, cls, tid, box, conf)
                continue

            person_id = int(tid)
            person_box = box.tolist()

            # Process person for recognition
            name, is_known, attrs, persistent_id = self._process_person(
                person_id, person_box, face_boxes, face_embeddings, face_attributes, run_fr
            )

            # Mark new person as processed if it was considered new
            if person_id in new_person_ids:
                self.new_person_tracker.mark_processed(person_id)

            # Determine color based on status
            if is_known:
                color = self.config.known_color
                self.stats["known_persons_detected"] += 1
            else:
                # Check if a face was actually associated
                face_idx = self._associate_face_to_person(person_box, face_boxes)
                if face_idx is not None:
                    color = self.config.unknown_color  # face present but unknown
                else:
                    color = self.config.no_face_color  # no face detected
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

        # Alarm handling with debouncing
        alarm_threshold = self.CFG.get("records", 1)
        if unknown_count >= alarm_threshold:
            self._trigger_alarm(unknown_count)
            if self._alarm_active:
                self._save_alarm_screenshots(detected_persons)
                self._log_event("ALARM", {"unknown_count": unknown_count, "persons": detected_persons})
        else:
            self._clear_alarm()

        output_frame = annotator.result()
        self.display_output(output_frame)

        # Periodic stats logging
        if self.stats["frames_processed"] % 30 == 0:
            self._logger.info(
                f"Frames:{self.stats['frames_processed']} FR:{self.stats['fr_runs']} "
                f"Cache:{self.stats['cache_hits']} Faces:{self.stats['total_faces_detected']} "
                f"Poses:{self.stats['poses_detected']} Skipped(Q:{self.stats['skipped_low_quality']} "
                f"Y:{self.stats['skipped_yaw']}) RegistryMatches:{self.stats['registry_matches']}"
            )

        # Overlay
        stats_text = (
            f"Tracks:{len(person_ids)} Known:{self.stats['known_persons_detected']} "
            f"Unknown:{self.stats['unknown_persons_detected']}"
        )
        cv2.putText(output_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status = f"FR:{'ON' if run_fr else 'OFF'} Face:{len(face_boxes)} Pose:{self.stats['poses_detected']}"
        cv2.putText(output_frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
        self._logger.info("Security Guard reset")


# ============================================================================
# FACTORY FUNCTION (loads known faces from family_member folder)
# ============================================================================


def create_security_guard(
    config: Optional[SecurityGuardConfig] = None, face_directory: Optional[str] = None, **kwargs
) -> EnhancedSecurityGuard:
    """
    Factory function to create a Security Guard instance.
    Loads known faces from family_member/person_name/*.jpg and builds a dictionary
    of person names to list of embeddings.
    """
    cfg = config or SecurityGuardConfig()
    logger = _setup_logging("Factory")

    # Use GPU for embedding extraction during loading
    detector = InsightFaceDetector(cfg.insightface_model, cfg.insightface_det_size)

    embeddings_dict: Dict[str, List[np.ndarray]] = {}  # name -> list of embeddings

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
                        logger.debug(f"Loaded: {person_name}/{img_file}")
                except Exception as e:
                    logger.warning(f"Failed loading {person_name}/{img_file}: {e}")
            if person_embeddings:
                embeddings_dict[person_name] = person_embeddings
                logger.info(f"Loaded {len(person_embeddings)} embeddings for {person_name}")
            else:
                logger.warning(f"No valid faces found for {person_name}")
    else:
        logger.warning(f"Face directory not found: {face_dir}")

    # Create security guard
    return EnhancedSecurityGuard(config=cfg, known_embeddings_dict=embeddings_dict, **kwargs)


# ============================================================================
# MAIN ENTRY POINT``
# ============================================================================

if __name__ == "__main__":
    config = SecurityGuardConfig(
        frame_interval=10,
        enable_new_person_detection=True,
        person_cache_ttl=60.0,
        face_tolerance=0.5,
        enable_face_tracking=True,
        show_attributes=True,
        capture_faces=True,
        enable_logging=True,
        # Optimizations
        min_recognition_quality=30.0,
        max_face_yaw=30.0,
        known_cache_ttl=60.0,
        unknown_cache_ttl=5.0,
        min_confidence=0.6,
        # Re-identification settings
        reid_similarity_threshold=0.7,
        registry_max_embeddings_per_person=5,
        alarm_debounce_frames=3,
    )

    print("\n" + "=" * 60)
    print("Enhanced Security Guard - YOLO + InsightFace + Re-ID")
    print("=" * 60 + "\n")

    video_path = "../media_files/istockphoto-1450610976-640_adpp_is.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "output_reid.avi"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    guard = create_security_guard(
        config=config,
        show=True,
        model="yolo26m-pose.pt",
        # classes=[0],  # person only
        # vision_point=(width // 2 - 150, height - 350),
        conf=0.18,
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
