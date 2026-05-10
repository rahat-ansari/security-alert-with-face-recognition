"""
Enhanced InsightFace Security Guard System - Optimized Version

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

Installation:
    pip install insightface onnxruntime opencv-python numpy pygame

Version: 2.2.0 (Refined)
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

# Third-party imports
from insightface.app import FaceAnalysis
from ultralytics import solutions
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors
from ultralytics.utils import LOGGER as ultralytics_logger

# Suppress Ultralytics "No tracks found" warning (only show errors)
ultralytics_logger.setLevel(logging.ERROR)

# ============================================================================
# CONSTANTS
# ============================================================================

# Quality assessment weights
SIZE_WEIGHT = 0.30
BRIGHTNESS_WEIGHT = 0.25
SHARPNESS_WEIGHT = 0.25
CONTRAST_WEIGHT = 0.20

# Quality thresholds
DEFAULT_MIN_FACE_QUALITY = 10.1
DEFAULT_MIN_FACE_SIZE = 40
DEFAULT_IOU_THRESHOLD = 0.8
DEFAULT_FACE_TRACK_MAX_AGE = 50

# Performance tuning
NORMALIZATION_EPS = 1e-5
DEFAULT_CACHE_TTL = 30.0

# Emotion mapping
EMOTION_MAP = {0: "😊", 1: "😐", 2: "😢", 3: "😠", 4: "😲", 5: "😨", 6: "😞"}
GENDER_MAP = {0: "♂", 1: "♀"}

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
    frame_interval: int = 10
    face_recognition_interval: float = 0.0

    # Detection settings
    enable_new_person_detection: bool = True
    person_cache_ttl: float = DEFAULT_CACHE_TTL
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

    # Screenshot settings
    capture_faces: bool = True
    screenshot_dir: Optional[str] = None
    min_face_quality: float = DEFAULT_MIN_FACE_QUALITY

    # Feature flags
    enable_logging: bool = True
    enable_keypoints_extraction: bool = False
    enable_keypoints_display: bool = False

    # New performance optimizations
    min_recognition_quality: float = 30.0
    max_face_yaw: float = 45.0
    known_cache_ttl: float = 60.0
    unknown_cache_ttl: float = 5.0
    min_confidence: float = 0.6

    # Alarm settings
    alarm_cooldown: float = 5.0  # seconds before allowing another alarm

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
# FACE SCREENSHOT CAPTURER
# ============================================================================


@dataclass
class FaceData:
    """Data class for storing face information."""

    box: List[float]
    quality: float
    frame: np.ndarray
    is_known: bool
    name: str
    timestamp: float = field(default_factory=time.time)


class FaceScreenshotCapturer:
    """Captures and saves face screenshots with quality assessment and deduplication."""

    def __init__(self, output_dir: str = "../captured_faces", min_quality: float = DEFAULT_MIN_FACE_QUALITY):
        self.output_dir = output_dir
        self.min_quality = min_quality

        self.known_dir = os.path.join(output_dir, "known")
        self.unknown_dir = os.path.join(output_dir, "unknown")

        # Create directories
        for d in [self.known_dir, self.unknown_dir]:
            os.makedirs(d, exist_ok=True)

        # Thread-safe data structures
        self.best_faces: Dict[int, FaceData] = {}
        self._unknown_count = 0
        self._captured_person_ids: set = set()
        self._lock = Lock()

        self._logger = _setup_logging("ScreenshotCapturer")
        self._logger.info(f"Screenshots directory: {output_dir}")

    def assess_quality(self, frame: np.ndarray, box: List[float]) -> float:
        """
        Assess face quality based on multiple factors.

        Args:
            frame: Video frame
            box: Face bounding box [x1, y1, x2, y2]

        Returns:
            Quality score (0-100)
        """
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
        self, track_id: int, box: List[float], frame: np.ndarray, is_known: bool, name: str = "Unknown"
    ) -> bool:
        """
        Update the best face for a tracked person.

        Args:
            track_id: Unique person ID
            box: Face bounding box
            frame: Video frame
            is_known: Whether the person is known
            name: Person's name

        Returns:
            True if face was updated, False if quality too low
        """
        quality = self.assess_quality(frame, box)

        if quality < self.min_quality:
            return False

        with self._lock:
            if track_id not in self.best_faces or quality > self.best_faces[track_id].quality:
                self.best_faces[track_id] = FaceData(
                    box=box.copy(), quality=quality, frame=frame.copy(), is_known=is_known, name=name
                )
                return True
        return False

    def save_all(self) -> int:
        """
        Save all captured faces to disk.

        Returns:
            Number of faces saved
        """
        saved_count = 0
        with self._lock:
            for tid, data in self.best_faces.items():
                try:
                    path = self._save_face_image(tid, data)
                    if path:
                        saved_count += 1
                        self._logger.info(f"Saved face: {path}")
                except Exception as e:
                    self._logger.error(f"Save failed for track {tid}: {e}")

        return saved_count

    def _save_face_image(self, tid: int, data: FaceData) -> Optional[str]:
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
        track_id: Optional[int] = None,
        is_known: bool = False,
        name: str = "Unknown",
    ) -> Optional[str]:
        """
        Save a face screenshot with deduplication.

        Args:
            frame: Video frame
            box: Face bounding box
            track_id: Unique person ID for deduplication
            is_known: Whether person is known
            name: Person's name

        Returns:
            Path to saved file or None
        """
        # Deduplication check
        if track_id is not None:
            with self._lock:
                if track_id in self._captured_person_ids:
                    return None
                self._captured_person_ids.add(track_id)

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
                    if track_id is not None:
                        filename = f"unknown_{track_id}_{timestamp}.jpg"
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
        """Reset captured person IDs to allow new captures."""
        with self._lock:
            self._captured_person_ids.clear()

    def get_captured_ids(self) -> set:
        """Get copy of captured person IDs."""
        with self._lock:
            return self._captured_person_ids.copy()

    def get_best_face(self, track_id: int) -> Optional[FaceData]:
        """Get the best face data for a track ID."""
        with self._lock:
            return self.best_faces.get(track_id)


# ============================================================================
# FACE ATTRIBUTE ANALYZER
# ============================================================================


class FaceAttributeAnalyzer:
    """Analyzes and formats face attributes for display."""

    EMOTION = EMOTION_MAP
    GENDER = GENDER_MAP

    @staticmethod
    def format(
        age: float, gender: float, emotion: Optional[np.ndarray] = None, pose: Optional[np.ndarray] = None
    ) -> str:
        """
        Format face attributes into a display string.

        Args:
            age: Estimated age
            gender: Gender probability (0=male, 1=female)
            emotion: Emotion array from InsightFace
            pose: Pose array [pitch, yaw, roll]

        Returns:
            Formatted attribute string
        """
        # Format age
        age_str = f"{int(age)}y"

        # Format gender
        gender_idx = int(gender > 0.5)
        gender_str = FaceAttributeAnalyzer.GENDER.get(gender_idx, "?")

        # Format emotion
        emo_idx = 1  # Default to neutral
        if emotion is not None and len(emotion) > 0:
            try:
                emo_idx = int(np.argmax(emotion))
            except Exception:
                pass
        emo_str = FaceAttributeAnalyzer.EMOTION.get(emo_idx, "😐")

        # Format pose if available
        pose_str = ""
        if pose is not None and len(pose) >= 3:
            try:
                pitch, yaw, roll = pose[0], pose[1], pose[2]
                pose_str = f" | P:{int(pitch)}° Y:{int(yaw)}°"
            except Exception:
                pass

        return f"{age_str} {gender_str} {emo_str}{pose_str}"


# ============================================================================
# FACE TRACKER
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
        """Compute Intersection over Union between two boxes."""
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
        """
        Update face tracks with new detections.

        Args:
            boxes: List of face bounding boxes
            embeddings: List of face embeddings
            attributes: List of face attribute dicts

        Returns:
            Dict of active tracks
        """
        with self._lock:
            # Age out old tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["age"] += 1

            self.tracks = {k: v for k, v in self.tracks.items() if v["age"] < self.max_age}

            matched = set()

            # Match new detections to existing tracks
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

            return {track_id: data for track_id, data in self.tracks.items() if track_id in matched}

    def get_track(self, track_id: int) -> Optional[Dict]:
        """Get track data by ID."""
        with self._lock:
            return self.tracks.get(track_id)

    def reset(self) -> None:
        """Clear all tracks."""
        with self._lock:
            self.tracks.clear()
            self.next_id = 0


# ============================================================================
# PERSON CACHE
# ============================================================================


@dataclass
class PersonCacheEntry:
    """Data class for cached person data."""

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
        """Get cached entry if not expired using TTL based on known status."""
        with self._lock:
            if person_id in self._cache:
                entry = self._cache[person_id]
                # Select TTL based on whether the person was known
                ttl = self.config.known_cache_ttl if entry.is_known else self.config.unknown_cache_ttl
                if time.time() - entry.timestamp < ttl:
                    self._cache.move_to_end(person_id)
                    return entry
                del self._cache[person_id]
            return None

    def set(self, person_id: int, name: str, is_known: bool, attributes: Optional[Dict] = None) -> None:
        """Set cache entry."""
        with self._lock:
            self._cache[person_id] = PersonCacheEntry(
                name=name, is_known=is_known, timestamp=time.time(), attributes=attributes or {}
            )

    def invalidate(self, person_id: int) -> None:
        """Remove specific entry from cache."""
        with self._lock:
            self._cache.pop(person_id, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()


# ============================================================================
# NEW PERSON TRACKER
# ============================================================================


class NewPersonTracker:
    """Tracks newly detected persons (simplified version)."""

    def __init__(self):
        self._seen: set = set()
        self._new: set = set()
        self._lock = Lock()

    def update(self, person_ids: List[int]) -> List[int]:
        """
        Update with current person IDs and return new ones.

        Args:
            person_ids: Current person IDs in frame

        Returns:
            List of new person IDs
        """
        current = set(person_ids)
        with self._lock:
            new_ids = current - self._seen
            self._seen.update(current)
            self._new.update(new_ids)
            return list(new_ids)

    def mark_processed(self, person_id: int) -> None:
        """Mark a person as processed."""
        with self._lock:
            self._new.discard(person_id)

    def is_new(self, person_id: int) -> bool:
        """Check if person is new."""
        with self._lock:
            return person_id in self._new

    def reset(self) -> None:
        """Reset all tracking."""
        with self._lock:
            self._seen.clear()
            self._new.clear()


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
        """
        Determine if face recognition should run this frame.

        Args:
            new_person_ids: List of new person IDs

        Returns:
            True if FR should run
        """
        with self._lock:
            self._frame_count += 1
            current_time = time.time()

            # Run if new persons detected and enabled
            if self.config.enable_new_person_detection and new_person_ids:
                return True

            # Run based on time interval
            if self.config.face_recognition_interval > 0:
                if current_time - self._last_fr_time >= self.config.face_recognition_interval:
                    self._last_fr_time = current_time
                    return True
                return False

            # Run based on frame interval
            if self.config.frame_interval > 0:
                return self._frame_count % self.config.frame_interval == 0

            return True

    def reset(self) -> None:
        """Reset frame counter."""
        with self._lock:
            self._frame_count = 0
            self._last_fr_time = 0.0


# ============================================================================
# INSIGHTFACE DETECTOR
# ============================================================================


class InsightFaceDetector:
    """Face detection and recognition using InsightFace."""

    def __init__(self, model: str = "buffalo_l", detection_size: Tuple[int, int] = (640, 640)):
        self.model_name = model
        self.detection_size = detection_size
        self._logger = _setup_logging("InsightFaceDetector")

        # Initialize InsightFace
        try:
            self.app = FaceAnalysis(name=model, providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=detection_size)
            self._logger.info(f"InsightFace initialized: {model}")
        except Exception as e:
            self._logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Tuple[List[List[float]], List[np.ndarray], List, List[Dict]]:
        """
        Detect faces in a frame.

        Args:
            frame: Input video frame

        Returns:
            Tuple of (boxes, embeddings, landmarks, attributes)
        """
        faces = self.app.get(frame)

        boxes = []
        embeddings = []
        landmarks = []
        attributes = []

        for face in faces:
            bbox = face.bbox

            # Filter small faces
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width < DEFAULT_MIN_FACE_SIZE or height < DEFAULT_MIN_FACE_SIZE:
                continue

            boxes.append(bbox.tolist())
            embeddings.append(face.embedding)
            landmarks.append(face.kps)

            # Extract attributes safely
            attributes.append(
                {
                    "age": face.age,
                    "gender": face.gender,
                    "emotion": getattr(face, "emotion", np.array([0.5])),
                    "pose": getattr(face, "pose", np.array([0, 0, 0])),
                }
            )

        return boxes, embeddings, landmarks, attributes

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get detection input size."""
        return self.detection_size


# ============================================================================
# FACE RECOGNIZER
# ============================================================================


class FaceRecognizer:
    """Handles face recognition against known faces with temporal smoothing."""

    def __init__(
        self,
        known_embeddings: List[np.ndarray],
        known_names: List[str],
        tolerance: float = 0.5,
        embedding_smoothing_frames: int = 5,
        min_confidence: float = 0.65,
    ):
        self.tolerance = tolerance
        self.embedding_smoothing_frames = embedding_smoothing_frames
        self.min_confidence = min_confidence
        self._logger = _setup_logging("FaceRecognizer")

        # Store known embeddings
        if known_embeddings and known_names:
            self._known_embeddings = np.array(known_embeddings)
            self._known_names = list(known_names)

            # Pre-compute normalized embeddings for performance
            norms = np.linalg.norm(self._known_embeddings, axis=1, keepdims=True)
            self._normalized_embeddings = self._known_embeddings / (norms + NORMALIZATION_EPS)

            self._logger.info(f"Loaded {len(known_names)} known faces")
        else:
            self._known_embeddings = np.array([])
            self._known_names = []
            self._normalized_embeddings = np.array([])
            self._logger.warning("No known faces loaded")

        # Temporal smoothing buffers for each track
        self._embedding_buffers: Dict[int, List[np.ndarray]] = {}
        self._identity_history: Dict[int, Tuple[str, float, int]] = {}  # track_id -> (name, confidence, frames_seen)

    def _smooth_embedding(self, track_id: int, embedding: np.ndarray) -> np.ndarray:
        """Smooth face embeddings over multiple frames for more stable recognition."""
        if track_id not in self._embedding_buffers:
            self._embedding_buffers[track_id] = []

        # Add new embedding to buffer
        self._embedding_buffers[track_id].append(embedding)

        # Keep only recent frames
        if len(self._embedding_buffers[track_id]) > self.embedding_smoothing_frames:
            self._embedding_buffers[track_id].pop(0)

        # Compute moving average
        buffer = self._embedding_buffers[track_id]
        if len(buffer) == 1:
            return buffer[0]

        # Weighted average (more recent frames have higher weight)
        weights = np.linspace(0.5, 1.0, len(buffer))
        weights = weights / weights.sum()

        smoothed = np.average(buffer, axis=0, weights=weights)
        return smoothed / np.linalg.norm(smoothed)

    def identify(self, embedding: np.ndarray, track_id: Optional[int] = None) -> Tuple[str, bool, float]:
        """
        Identify a face against known faces with temporal consistency.

        Args:
            embedding: Face embedding to identify
            track_id: Optional track ID for temporal smoothing

        Returns:
            Tuple of (name, is_known, confidence)
        """
        if len(self._known_embeddings) == 0:
            return "Unknown", False, 0.0

        try:
            # Apply temporal smoothing if track_id provided
            if track_id is not None:
                # Check identity history first
                if track_id in self._identity_history:
                    name, conf, frames_seen = self._identity_history[track_id]
                    # Only trust history if we've seen this person enough times with high confidence
                    if frames_seen >= 3 and conf > 0.8:
                        return name, True, conf

                # Apply smoothing
                embedding = self._smooth_embedding(track_id, embedding)

            # Normalize query embedding
            query_norm = embedding / (np.linalg.norm(embedding) + NORMALIZATION_EPS)
            query_norm = query_norm.reshape(1, -1)

            # Compute cosine similarity
            similarities = np.dot(query_norm, self._normalized_embeddings.T)[0]

            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]

            # Compute distance (1 - similarity for cosine)
            distance = 1 - best_similarity
            confidence = float(best_similarity)

            # Accept match only if within tolerance and above min_confidence
            if distance < self.tolerance and confidence >= self.min_confidence:
                # Update identity history
                if track_id is not None:
                    frames_seen = len(self._embedding_buffers.get(track_id, []))
                    self._identity_history[track_id] = (self._known_names[best_idx], confidence, frames_seen)

                return self._known_names[best_idx], True, confidence

        except Exception as e:
            self._logger.error(f"Identification error: {e}")

        return "Unknown", False, 0.0

    def clear_track(self, track_id: int) -> None:
        """Clear smoothing buffer for a specific track."""
        self._embedding_buffers.pop(track_id, None)
        self._identity_history.pop(track_id, None)

    def clear_all_tracks(self) -> None:
        """Clear all smoothing buffers."""
        self._embedding_buffers.clear()
        self._identity_history.clear()

    def add_known_face(self, embedding: np.ndarray, name: str) -> None:
        """Add a new known face."""
        if len(self._known_embeddings) == 0:
            self._known_embeddings = np.array([embedding])
            self._known_names = [name]
            self._normalized_embeddings = embedding.reshape(1, -1)
            self._normalized_embeddings /= (
                np.linalg.norm(self._normalized_embeddings, axis=1, keepdims=True) + NORMALIZATION_EPS
            )
        else:
            self._known_embeddings = np.vstack([self._known_embeddings, embedding])
            self._known_names.append(name)

            # Recompute normalized embeddings
            norms = np.linalg.norm(self._known_embeddings, axis=1, keepdims=True)
            self._normalized_embeddings = self._known_embeddings / (norms + NORMALIZATION_EPS)

    @property
    def known_count(self) -> int:
        """Number of known faces."""
        return len(self._known_names)


# ============================================================================
# SECURITY GUARD MAIN CLASS
# ============================================================================


class EnhancedSecurityGuard(solutions.VisionEye):
    """
    Main security guard class combining YOLO and InsightFace.

    Provides real-time person detection, face recognition,
    and security alerting.
    """

    def __init__(
        self,
        *args,
        config: Optional[SecurityGuardConfig] = None,
        known_embeddings: Optional[List[np.ndarray]] = None,
        known_names: Optional[List[str]] = None,
        **kwargs,
    ):
        # Initialize parent class
        super().__init__(*args, **kwargs)

        # Setup configuration
        self.config = config or SecurityGuardConfig()
        self.config.validate()

        # Setup logging
        self._logger = _setup_logging("SecurityGuard")

        # Initialize components
        self._initialize_components(known_embeddings, known_names)

        # Initialize statistics
        self._initialize_stats()

        # Initialize alarm state variables
        self._alarm_pending = 0
        self._alarm_triggered_this_frame = False
        self._last_alarm_time = 0.0  # for cooldown

        self._logger.info(f"Security Guard initialized with {len(known_names or [])} known faces")

    def _initialize_components(
        self, known_embeddings: Optional[List[np.ndarray]], known_names: Optional[List[str]]
    ) -> None:
        """Initialize all system components."""
        # Face detection
        self.face_detector = InsightFaceDetector(self.config.insightface_model, self.config.insightface_det_size)

        # Face recognition
        self.face_recognizer = FaceRecognizer(
            known_embeddings or [],
            known_names or [],
            self.config.face_tolerance,
            self.config.embedding_smoothing_frames,
            self.config.min_confidence,
        )

        # Face tracking
        self.face_tracker = FaceTracker(self.config.face_track_max_age, self.config.face_track_iou_threshold)

        # Person cache
        self.person_cache = PersonCache(self.config)

        # New person detection
        self.new_person_tracker = NewPersonTracker()

        # Frame controller
        self.frame_controller = FrameController(self.config)

        # Event logging
        self.event_logger = EventLogger() if self.config.enable_logging else None

        # Screenshot capture
        self.screenshot_capturer = None
        if self.config.capture_faces:
            screenshot_dir = self.config.screenshot_dir
            if screenshot_dir is None:
                screenshot_dir = str(_get_base_dir() / "captured_faces")
            self.screenshot_capturer = FaceScreenshotCapturer(screenshot_dir, self.config.min_face_quality)

        # Sound manager
        self.sound_manager = SoundManager()
        self._alarm_active = False

    def _initialize_stats(self) -> None:
        """Initialize statistics tracking."""
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
        }

    def _associate_face_to_person(self, face_box: List[float], person_boxes: List[List[float]]) -> Optional[int]:
        """Associate a face box with a person box."""
        # Calculate face center
        face_cx = (face_box[0] + face_box[2]) / 2
        face_cy = (face_box[1] + face_box[3]) / 2

        # Find containing person box
        for idx, person_box in enumerate(person_boxes):
            if person_box[0] <= face_cx <= person_box[2] and person_box[1] <= face_cy <= person_box[3]:
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
    ) -> Tuple[str, bool, Dict]:
        """
        Process a single person for recognition.

        Returns:
            Tuple of (name, is_known, attributes)
        """
        # Try cache first
        cached = self.person_cache.get(person_id)
        if cached:
            self.stats["cache_hits"] += 1
            return cached.name, cached.is_known, cached.attributes

        # Run face recognition if needed
        if run_fr and face_embeddings:
            # Find associated face
            face_idx = self._associate_face_to_person(person_box, face_boxes)

            if face_idx is not None and face_idx < len(face_embeddings) and face_idx < len(face_attributes):
                # --- Quality check before recognition ---
                if self.screenshot_capturer:
                    quality = self.screenshot_capturer.assess_quality(self._current_frame, face_boxes[face_idx])
                    if quality < self.config.min_recognition_quality:
                        self.stats["skipped_low_quality"] += 1
                        return "Unknown", False, {}

                # --- Yaw (head pose) check ---
                attrs = face_attributes[face_idx]
                if "pose" in attrs and attrs["pose"] is not None:
                    yaw = abs(attrs["pose"][1])  # yaw is second element
                    if yaw > self.config.max_face_yaw:
                        self.stats["skipped_yaw"] += 1
                        return "Unknown", False, {}

                # Identify face
                name, is_known, confidence = self.face_recognizer.identify(
                    face_embeddings[face_idx], track_id=person_id
                )
                attrs = face_attributes[face_idx]

                # Update screenshot if enabled
                if self.screenshot_capturer and face_idx < len(face_boxes):
                    self.screenshot_capturer.update_best_face(
                        person_id, face_boxes[face_idx], self._current_frame, is_known, name
                    )

                # Track pose detection
                if "pose" in attrs and attrs["pose"] is not None:
                    self.stats["poses_detected"] += 1

                # Cache result
                self.person_cache.set(person_id, name, is_known, attrs)

                return name, is_known, attrs

        # Unknown person
        return "Unknown", False, {}

    def _trigger_alarm(self, unknown_count: int) -> None:
        """Trigger security alarm with cooldown."""
        current_time = time.time()
        if current_time - self._last_alarm_time > self.config.alarm_cooldown:
            if self.sound_manager.play_alarm():
                self._alarm_active = True
                self._last_alarm_time = current_time
                self.stats["alerts_triggered"] += 1
                self._logger.warning(f"ALARM: {unknown_count} unknown person(s) detected")

    def _clear_alarm(self) -> None:
        """Clear security alarm."""
        if self._alarm_active:
            self.sound_manager.stop_alarm()
            self._alarm_active = False

    def _save_alarm_screenshots(self, detected_persons: List[Dict]) -> None:
        """Save screenshots during alarm."""
        if not self.screenshot_capturer:
            return

        for person_info in detected_persons:
            pid = person_info["id"]
            name = person_info["name"]
            is_known = person_info["is_known"]

            # Get best face for this person
            best_face = self.screenshot_capturer.get_best_face(pid)
            if best_face:
                path = self.screenshot_capturer.save_unknown_face(
                    self._current_frame, best_face.box, track_id=pid, is_known=is_known, name=name
                )
                if path:
                    self.stats["screenshots_saved"] += 1

    def _log_event(self, event_type: str, data: Dict) -> None:
        """Log security event."""
        if self.event_logger:
            self.event_logger.log(event_type, data)

    def _format_status_line(self, run_fr: bool, face_count: int) -> str:
        """Format status display line."""
        parts = [f"FR: {'ON' if run_fr else 'OFF'}", f"Face: {face_count}", f"Pose: {self.stats['poses_detected']}"]

        if self.config.enable_keypoints_display:
            parts.append(f"Keypoints: {len(self._pose_keypoints)}")

        return " | ".join(parts)

    def __call__(self, im0: np.ndarray) -> SolutionResults:
        """
        Process a frame and return annotated results.

        Args:
            im0: Input video frame

        Returns:
            Annotated frame results
        """
        self._current_frame = im0
        self.stats["frames_processed"] += 1

        # Extract YOLO tracks
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)

        # Extract person data
        person_ids, person_boxes = self._extract_person_data()

        # Extract pose keypoints if enabled
        self._pose_keypoints = self._extract_pose_keypoints()

        # Detect new persons
        new_person_ids = self.new_person_tracker.update(person_ids)

        # Determine if face recognition should run
        run_fr = self.frame_controller.should_run_face_recognition(new_person_ids)

        # Run face detection if needed
        face_boxes, face_embeddings, _, face_attributes = [], [], [], []

        if run_fr:
            self.stats["fr_runs"] += 1
            if new_person_ids:
                self.stats["new_person_triggers"] += len(new_person_ids)

            # Detect faces
            (face_boxes, face_embeddings, _, face_attributes) = self.face_detector.detect(im0)
            self.stats["total_faces_detected"] += len(face_boxes)

        # Process each detected person
        unknown_count = 0
        detected_persons = []

        for cls, tid, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            # Skip non-person classes
            if int(cls) != 0:
                self._annotate_object(annotator, cls, tid, box, conf)
                continue

            person_id = int(tid)
            person_box = box.tolist()

            # Process person for recognition
            name, is_known, attrs = self._process_person(
                person_id, person_box, face_boxes, face_embeddings, face_attributes, run_fr
            )

            # Mark new person as processed
            if person_id in new_person_ids:
                self.new_person_tracker.mark_processed(person_id)

            # Update statistics
            if is_known:
                self.stats["known_persons_detected"] += 1
            else:
                unknown_count += 1
                self.stats["unknown_persons_detected"] += 1

            # Build display label
            label = self._build_label(name, is_known, attrs, conf)

            # Annotate person
            self._annotate_person(annotator, box, label, person_id, person_box)

            # Store for logging
            detected_persons.append(
                {"id": person_id, "name": name, "is_known": is_known, "box": person_box, "attributes": attrs}
            )

        # Handle alarm with debouncing and cooldown
        alarm_threshold = self.CFG.get("records", 1)
        debounce_threshold = 3  # Number of consecutive frames before triggering alarm

        alarm_condition_met = unknown_count >= alarm_threshold

        if alarm_condition_met:
            self._alarm_pending += 1
            if self._alarm_pending >= debounce_threshold and not self._alarm_triggered_this_frame:
                self._trigger_alarm(unknown_count)
                self._save_alarm_screenshots(detected_persons)
                self._log_event("ALARM", {"unknown_count": unknown_count, "persons": detected_persons})
                self._alarm_triggered_this_frame = True
                self._logger.debug(
                    f"Alarm triggered: {unknown_count} unknown persons (debounce: {self._alarm_pending}/{debounce_threshold})"
                )
        else:
            self._alarm_pending = 0
            self._clear_alarm()

        # Reset frame flag for next iteration
        self._alarm_triggered_this_frame = False

        # Generate output
        output_frame = annotator.result()
        self.display_output(output_frame)

        # Log statistics periodically
        self._log_stats()

        # Draw overlay information
        self._draw_overlay(output_frame, len(person_ids), run_fr, len(face_boxes))

        return SolutionResults(plot_im=output_frame, total_tracks=len(self.track_ids))

    def _extract_person_data(self) -> Tuple[List[int], List[List[float]]]:
        """Extract person IDs and boxes from YOLO tracks."""
        person_ids = []
        person_boxes = []

        for cls, tid, box in zip(self.clss, self.track_ids, self.boxes):
            if int(cls) == 0:  # Person class
                person_ids.append(int(tid))
                person_boxes.append(box.tolist())

        return person_ids, person_boxes

    def _extract_pose_keypoints(self) -> Dict[int, List]:
        """Extract pose keypoints if enabled."""
        pose_keypoints = {}

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
                            pose_keypoints[int(tid)] = kpts_data[i]
            except Exception as e:
                self._logger.debug(f"Pose extraction error: {e}")

        return pose_keypoints

    def _build_label(self, name: str, is_known: bool, attributes: Dict, confidence) -> str:
        """Build display label for person."""
        if is_known:
            label = name
        else:
            label = "Unknown"

        # Add attributes if enabled
        if self.config.show_attributes and attributes:
            attr_str = FaceAttributeAnalyzer.format(
                attributes.get("age", 0), attributes.get("gender", 0), attributes.get("emotion"), attributes.get("pose")
            )
            label = f"{name}\n{attr_str}"

        return label

    def _annotate_object(self, annotator: SolutionAnnotator, cls, tid, box, conf) -> None:
        """Annotate non-person objects."""
        label = self.adjust_box_label(cls, float(conf) if conf else 0.0, tid)
        annotator.box_label(box, label=label, color=colors(int(tid), True))
        annotator.visioneye(box, self.vision_point)

    def _annotate_person(
        self, annotator: SolutionAnnotator, box, label: str, person_id: int, person_box: List[float]
    ) -> None:
        """Annotate detected person."""
        # Get base label
        base = self.adjust_box_label(0, 0.0, person_id)
        prefix = str(self.CFG.get("person_label_prefix", label))
        final_label = f"{prefix}: {base}" if base else prefix

        # Draw annotations
        annotator.box_label(box, label=final_label, color=colors(int(person_id), True))
        annotator.visioneye(box, self.vision_point)

        # Draw pose keypoints if enabled
        if self.config.enable_keypoints_display and person_id in self._pose_keypoints:
            kpts = self._pose_keypoints[person_id]
            if kpts:
                kpts_array = np.array(kpts, dtype=np.float32)
                annotator.kpts(kpts_array, shape=self._current_frame.shape[:2], kpt_line=True)

    def _log_stats(self) -> None:
        """Log statistics every 30 frames."""
        if self.stats["frames_processed"] % 30 == 0:
            self._logger.info(
                f"Frames: {self.stats['frames_processed']} | "
                f"FR: {self.stats['fr_runs']} | "
                f"Cache: {self.stats['cache_hits']} | "
                f"Faces: {self.stats['total_faces_detected']} | "
                f"Poses: {self.stats['poses_detected']} | "
                f"Skipped: Q={self.stats['skipped_low_quality']} Y={self.stats['skipped_yaw']}"
            )

    def _draw_overlay(self, frame: np.ndarray, person_count: int, run_fr: bool, face_count: int) -> None:
        """Draw status overlay on frame."""
        # Detection stats
        stats_text = (
            f"Tracks: {person_count} | "
            f"Known: {self.stats['known_persons_detected']} | "
            f"Unknown: {self.stats['unknown_persons_detected']}"
        )

        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Status line
        status = self._format_status_line(run_fr, face_count)
        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def save_screenshots(self) -> int:
        """Save all captured screenshots."""
        if self.screenshot_capturer:
            return self.screenshot_capturer.save_all()
        return 0

    def reset(self) -> None:
        """Reset all tracking and caches."""
        self.person_cache.clear()
        self.new_person_tracker.reset()
        self.frame_controller.reset()
        self.face_tracker.reset()
        self.face_recognizer.clear_all_tracks()

        if self.screenshot_capturer:
            self.screenshot_capturer.reset_captured_ids()

        self._alarm_active = False
        self._alarm_pending = 0
        self._alarm_triggered_this_frame = False
        self._last_alarm_time = 0.0
        self._logger.info("Security Guard reset")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_security_guard(
    config: Optional[SecurityGuardConfig] = None, face_directory: Optional[str] = None, **kwargs
) -> EnhancedSecurityGuard:
    """
    Factory function to create a Security Guard instance.

    Args:
        config: Security configuration
        face_directory: Path to known faces directory
        **kwargs: Additional arguments for VisionEye

    Returns:
        Configured EnhancedSecurityGuard instance
    """
    cfg = config or SecurityGuardConfig()
    logger = _setup_logging("Factory")

    # Initialize detector
    detector = InsightFaceDetector(cfg.insightface_model, cfg.insightface_det_size)

    # Load known faces
    embeddings = []
    names = []

    face_dir = face_directory
    if face_dir is None:
        face_dir = str(_get_base_dir() / "family_members")

    if os.path.exists(face_dir):
        logger.info(f"Loading known faces from: {face_dir}")

        for person_name in os.listdir(face_dir):
            person_path = os.path.join(face_dir, person_name)

            if not os.path.isdir(person_path):
                continue

            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)

                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue

                    _, face_embs, _, _ = detector.detect(image)

                    if face_embs:
                        embeddings.append(face_embs[0])
                        names.append(person_name)
                        logger.info(f"Loaded: {person_name}/{image_file}")

                except Exception as e:
                    logger.warning(f"Failed loading {person_name}/{image_file}: {e}")
    else:
        logger.warning(f"Face directory not found: {face_dir}")

    # Create security guard
    return EnhancedSecurityGuard(config=cfg, known_embeddings=embeddings, known_names=names, **kwargs)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = SecurityGuardConfig(
        frame_interval=10,
        enable_new_person_detection=True,
        person_cache_ttl=60.0,
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
        alarm_cooldown=5.0,
    )

    # Banner
    print("\n" + "=" * 60)
    print("Enhanced Security Guard - YOLO + InsightFace + Pose")
    print("=" * 60 + "\n")

    # Open video
    video_path = "../media_files/WIN_20260227_22_00_29_Pro.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer
    output_path = "output.avi"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Use a standard YOLO pose model (adjust if you have a different version)
    model_name = "yolo26n-pose.pt"  # FIXED: was "yolo26n-pose.pt" which is invalid

    # Check if model exists (optional)
    if not os.path.exists(model_name):
        print(f"Warning: Model file {model_name} not found locally. Ultralytics will download it.")

    # Create security guard
    guard = create_security_guard(
        config=config,
        show=True,
        model=model_name,
        classes=[0, 2],
        vision_point=(width // 2 - 250, height - 10),
        conf=0.3,
        records=1,
    )

    # Process video
    frame_count = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        result = guard(frame)
        writer.write(result.plot_im)

    # Save screenshots
    print("\n[INFO] Saving screenshots...")
    saved = guard.save_screenshots()
    print(f"[INFO] Saved {saved} screenshots")

    # Cleanup
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for key, value in guard.stats.items():
        print(f"  {key}: {value}")

    print("=" * 60)