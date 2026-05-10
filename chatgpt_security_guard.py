"""
Enhanced Security Guard System - Optimized V2

- YOLO26 via VisionEye for person detection, tracking, and pose estimation
- InsightFace for face detection, recognition, and attributes (age, gender, emotion, pose)
- Frame‑skipping and quality/yaw pre‑filtering to minimise recognition calls
- Separate cache TTLs for known and unknown persons
- Bounding boxes always displayed (green = known, red = unknown with face, blue = no face)
- Alarm sound, event logging, screenshot saving
- GPU acceleration enabled (falls back to CPU if CUDA unavailable)
"""

import cv2
import numpy as np
import pygame
import os
import time
import json
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from threading import Lock
from contextlib import contextmanager
import warnings

from insightface.app import FaceAnalysis
from ultralytics import solutions
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors

# ============================================================================
# CONSTANTS & CONFIG
# ============================================================================

# Quality assessment weights (same as original)
SIZE_WEIGHT = 0.30
BRIGHTNESS_WEIGHT = 0.25
SHARPNESS_WEIGHT = 0.25
CONTRAST_WEIGHT = 0.20

# Emotion / gender maps
EMOTION_MAP = {0: "😊", 1: "😐", 2: "😢", 3: "😠", 4: "😲", 5: "😨", 6: "😞"}
GENDER_MAP = {0: "♂", 1: "♀"}

NORMALIZATION_EPS = 1e-5


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
# CONFIGURATION DATACLASS (extended with optimizations)
# ============================================================================


@dataclass
class SecurityGuardConfig:
    # Paths
    family_member_dir: str = "../family_members"
    yolo_model: str = "yolo26n-pose.pt"
    screenshot_dir: Optional[str] = None

    # Processing intervals
    frame_interval: int = 10  # legacy, kept for compatibility
    face_recognition_interval: float = 0.0
    frame_skip: int = 5  # run face recognition every N frames (new)

    # Detection settings
    enable_new_person_detection: bool = True
    face_tolerance: float = 0.5
    embedding_smoothing_frames: int = 5
    min_face_size: Tuple[int, int] = (40, 40)
    iou_threshold: float = 0.3

    # InsightFace settings
    insightface_model: str = "buffalo_l"
    insightface_det_size: Tuple[int, int] = (640, 640)

    # Face tracking settings (optional)
    enable_face_tracking: bool = False  # disabled by default (unused)

    # Display settings
    enable_liveness: bool = True
    show_attributes: bool = True
    enable_keypoints_display: bool = False

    # Screenshot settings
    capture_faces: bool = True
    min_face_quality: float = 30.0

    # Feature flags
    enable_logging: bool = True
    enable_keypoints_extraction: bool = False

    # New performance optimizations
    min_recognition_quality: float = 30.0  # skip FR if face quality below this
    max_face_yaw: float = 30.0  # skip FR if head turned beyond this (degrees)
    known_cache_ttl: float = 60.0
    unknown_cache_ttl: float = 5.0
    min_confidence: float = 0.6  # minimum similarity to accept a match

    # Visualization colors
    known_color: Tuple[int, int, int] = (0, 255, 0)  # green
    unknown_color: Tuple[int, int, int] = (0, 0, 255)  # red
    no_face_color: Tuple[int, int, int] = (255, 0, 0)  # blue

    def validate(self) -> "SecurityGuardConfig":
        if self.frame_interval < 0:
            raise ValueError("frame_interval must be non-negative")
        if self.min_face_size[0] <= 0 or self.min_face_size[1] <= 0:
            raise ValueError("min_face_size must have positive dimensions")
        return self


# ============================================================================
# SOUND MANAGER (unchanged from original)
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
            self._initialized_pygame = False
            self._logger = _setup_logging("SoundManager")
            self._sound_lock = Lock()
            self._load_alarm_sound()

    def _load_alarm_sound(self) -> None:
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
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)

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
# FACE SCREENSHOT CAPTURER (simplified, quality assessment kept)
# ============================================================================


@dataclass
class FaceData:
    box: List[float]
    quality: float
    frame: np.ndarray
    is_known: bool
    name: str
    timestamp: float = field(default_factory=time.time)


class FaceScreenshotCapturer:
    def __init__(self, output_dir: str = "../captured_faces", min_quality: float = 30.0):
        self.output_dir = output_dir
        self.min_quality = min_quality
        self.known_dir = os.path.join(output_dir, "known")
        self.unknown_dir = os.path.join(output_dir, "unknown")
        for d in [self.known_dir, self.unknown_dir]:
            os.makedirs(d, exist_ok=True)
        self.best_faces: Dict[int, FaceData] = {}
        self._unknown_count = 0
        self._captured_person_ids: set = set()
        self._lock = Lock()
        self._logger = _setup_logging("ScreenshotCapturer")
        self._logger.info(f"Screenshots directory: {output_dir}")

    def assess_quality(self, frame: np.ndarray, box: List[float]) -> float:
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
            brightness = np.mean(gray)
            bright_score = max(0, 100 - abs(brightness - 128) * 0.8)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            sharp_score = min(100, lap.var() / 10)
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
        self, track_id: int, box: List[float], frame: np.ndarray, is_known: bool, name: str = "Unknown"
    ) -> bool:
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
        saved = 0
        with self._lock:
            for tid, data in self.best_faces.items():
                try:
                    if self._save_face_image(tid, data):
                        saved += 1
                except Exception as e:
                    self._logger.error(f"Save failed for track {tid}: {e}")
        return saved

    def _save_face_image(self, tid: int, data: FaceData) -> Optional[str]:
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
        return path if cv2.imwrite(path, face) else None

    def get_best_face(self, track_id: int) -> Optional[FaceData]:
        with self._lock:
            return self.best_faces.get(track_id)

    def reset_captured_ids(self) -> None:
        with self._lock:
            self._captured_person_ids.clear()


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
        gender_idx = int(gender > 0.5)
        gender_str = FaceAttributeAnalyzer.GENDER.get(gender_idx, "?")
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
                pose_str = f" | P:{int(pitch)}° Y:{int(yaw)}°"
            except Exception:
                pass
        return f"{age_str} {gender_str} {emo_str}{pose_str}"


# ============================================================================
# FACE RECOGNIZER with embedding smoothing & identity history
# ============================================================================


class FaceRecognizer:
    def __init__(
        self,
        known_embeddings: List[np.ndarray],
        known_names: List[str],
        tolerance: float = 0.5,
        smoothing_frames: int = 5,
        min_confidence: float = 0.6,
    ):
        self.tolerance = tolerance
        self.smoothing_frames = smoothing_frames
        self.min_confidence = min_confidence
        self._logger = _setup_logging("FaceRecognizer")

        if known_embeddings and known_names:
            self._known_embeddings = np.array(known_embeddings)
            self._known_names = list(known_names)
            norms = np.linalg.norm(self._known_embeddings, axis=1, keepdims=True)
            self._normalized_embeddings = self._known_embeddings / (norms + NORMALIZATION_EPS)
            self._logger.info(f"Loaded {len(known_names)} known faces")
        else:
            self._known_embeddings = np.array([])
            self._known_names = []
            self._normalized_embeddings = np.array([])
            self._logger.warning("No known faces loaded")

        self._embedding_buffers: Dict[int, List[np.ndarray]] = {}
        self._identity_history: Dict[int, Tuple[str, float, int]] = {}

    def _smooth_embedding(self, track_id: int, embedding: np.ndarray) -> np.ndarray:
        if track_id not in self._embedding_buffers:
            self._embedding_buffers[track_id] = []
        buf = self._embedding_buffers[track_id]
        buf.append(embedding)
        if len(buf) > self.smoothing_frames:
            buf.pop(0)
        if len(buf) == 1:
            return buf[0]
        weights = np.linspace(0.5, 1.0, len(buf))
        weights /= weights.sum()
        smoothed = np.average(buf, axis=0, weights=weights)
        return smoothed / (np.linalg.norm(smoothed) + NORMALIZATION_EPS)

    def identify(self, embedding: np.ndarray, track_id: Optional[int] = None) -> Tuple[str, bool, float]:
        if len(self._known_embeddings) == 0:
            return "Unknown", False, 0.0

        try:
            if track_id is not None:
                if track_id in self._identity_history:
                    name, conf, frames = self._identity_history[track_id]
                    if frames >= 3 and conf > 0.8:
                        return name, True, conf
                embedding = self._smooth_embedding(track_id, embedding)

            query_norm = embedding / (np.linalg.norm(embedding) + NORMALIZATION_EPS)
            query_norm = query_norm.reshape(1, -1)
            similarities = np.dot(query_norm, self._normalized_embeddings.T)[0]
            best_idx = np.argmax(similarities)
            best_sim = similarities[best_idx]
            distance = 1 - best_sim
            confidence = float(best_sim)

            if distance < self.tolerance and confidence >= self.min_confidence:
                if track_id is not None:
                    frames_seen = len(self._embedding_buffers.get(track_id, []))
                    self._identity_history[track_id] = (self._known_names[best_idx], confidence, frames_seen)
                return self._known_names[best_idx], True, confidence
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
# PERSON CACHE with separate TTLs
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

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ============================================================================
# NEW PERSON TRACKER (unchanged)
# ============================================================================


class NewPersonTracker:
    def __init__(self):
        self._seen: set = set()
        self._new: set = set()
        self._lock = Lock()

    def update(self, person_ids: List[int]) -> List[int]:
        current = set(person_ids)
        with self._lock:
            new_ids = current - self._seen
            self._seen.update(current)
            self._new.update(new_ids)
            return list(new_ids)

    def mark_processed(self, person_id: int) -> None:
        with self._lock:
            self._new.discard(person_id)

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()
            self._new.clear()


# ============================================================================
# FRAME CONTROLLER (extended with frame_skip)
# ============================================================================


class FrameController:
    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self._frame_count = 0
        self._last_fr_time = 0.0
        self._lock = Lock()

    def should_run_face_recognition(self, new_person_ids: List[int]) -> bool:
        with self._lock:
            self._frame_count += 1
            current_time = time.time()

            # New person detection overrides everything
            if self.config.enable_new_person_detection and new_person_ids:
                return True

            # Use frame_skip if set (new optimization)
            if self.config.frame_skip > 0:
                return self._frame_count % self.config.frame_skip == 0

            # Fallback to legacy intervals
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
# INSIGHTFACE DETECTOR (unchanged)
# ============================================================================


class InsightFaceDetector:
    def __init__(self, model: str = "buffalo_l", detection_size: Tuple[int, int] = (640, 640)):
        self.model_name = model
        self.detection_size = detection_size
        self._logger = _setup_logging("InsightFaceDetector")
        try:
            # Automatically use GPU if available (providers order)
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
            if (bbox[2] - bbox[0]) < 40 or (bbox[3] - bbox[1]) < 40:
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
# MAIN SECURITY GUARD (inherits from VisionEye)
# ============================================================================


class EnhancedSecurityGuard(solutions.VisionEye):
    def __init__(
        self,
        *args,
        config: Optional[SecurityGuardConfig] = None,
        known_embeddings: Optional[List[np.ndarray]] = None,
        known_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config or SecurityGuardConfig()
        self.config.validate()
        self._logger = _setup_logging("SecurityGuard")
        self._initialize_components(known_embeddings, known_names)
        self._initialize_stats()
        self._logger.info(f"Security Guard initialized with {len(known_names or [])} known faces")

    def _initialize_components(self, known_embeddings, known_names):
        self.face_detector = InsightFaceDetector(self.config.insightface_model, self.config.insightface_det_size)
        self.face_recognizer = FaceRecognizer(
            known_embeddings or [],
            known_names or [],
            self.config.face_tolerance,
            self.config.embedding_smoothing_frames,
            self.config.min_confidence,
        )
        self.person_cache = PersonCache(self.config)
        self.new_person_tracker = NewPersonTracker()
        self.frame_controller = FrameController(self.config)
        self.event_logger = EventLogger() if self.config.enable_logging else None
        self.screenshot_capturer = None
        if self.config.capture_faces:
            sd = self.config.screenshot_dir or str(_get_base_dir() / "captured_faces")
            self.screenshot_capturer = FaceScreenshotCapturer(sd, self.config.min_face_quality)
        self.sound_manager = SoundManager()
        self._alarm_active = False

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
        }

    def _associate_face_to_person(self, face_box, person_boxes):
        face_cx = (face_box[0] + face_box[2]) / 2
        face_cy = (face_box[1] + face_box[3]) / 2
        for idx, pbox in enumerate(person_boxes):
            if pbox[0] <= face_cx <= pbox[2] and pbox[1] <= face_cy <= pbox[3]:
                return idx
        return None

    def _process_person(
        self, person_id, person_box, face_boxes, face_embeddings, face_attributes, run_fr
    ) -> Tuple[str, bool, Dict]:
        # Cache first
        cached = self.person_cache.get(person_id)
        if cached:
            self.stats["cache_hits"] += 1
            return cached.name, cached.is_known, cached.attributes

        if run_fr and face_embeddings:
            face_idx = self._associate_face_to_person(person_box, face_boxes)
            if face_idx is not None and face_idx < len(face_embeddings):
                # Quality check
                if self.screenshot_capturer:
                    q = self.screenshot_capturer.assess_quality(self._current_frame, face_boxes[face_idx])
                    if q < self.config.min_recognition_quality:
                        self.stats["skipped_low_quality"] += 1
                        return "Unknown", False, {}
                # Yaw check
                if face_idx < len(face_attributes):
                    pose = face_attributes[face_idx].get("pose")
                    if pose is not None and len(pose) >= 3:
                        if abs(pose[1]) > self.config.max_face_yaw:
                            self.stats["skipped_yaw"] += 1
                            return "Unknown", False, {}

                name, is_known, conf = self.face_recognizer.identify(face_embeddings[face_idx], track_id=person_id)
                attrs = face_attributes[face_idx] if face_idx < len(face_attributes) else {}

                if self.screenshot_capturer and face_idx < len(face_boxes):
                    self.screenshot_capturer.update_best_face(
                        person_id, face_boxes[face_idx], self._current_frame, is_known, name
                    )

                if "pose" in attrs and attrs["pose"] is not None:
                    self.stats["poses_detected"] += 1

                self.person_cache.set(person_id, name, is_known, attrs)
                return name, is_known, attrs

        return "Unknown", False, {}

    def _trigger_alarm(self, unknown_count):
        if not self._alarm_active and self.sound_manager.play_alarm():
            self._alarm_active = True
            self.stats["alerts_triggered"] += 1
            self._logger.warning(f"ALARM: {unknown_count} unknown person(s) detected")

    def _clear_alarm(self):
        if self._alarm_active:
            self.sound_manager.stop_alarm()
            self._alarm_active = False

    def _save_alarm_screenshots(self, detected_persons):
        if not self.screenshot_capturer:
            return
        for p in detected_persons:
            best = self.screenshot_capturer.get_best_face(p["id"])
            if best:
                path = self.screenshot_capturer.save_unknown_face(
                    self._current_frame, best.box, track_id=p["id"], is_known=p["is_known"], name=p["name"]
                )
                if path:
                    self.stats["screenshots_saved"] += 1

    def _log_event(self, event_type, data):
        if self.event_logger:
            self.event_logger.log(event_type, data)

    def _extract_person_data(self):
        person_ids, person_boxes = [], []
        for cls, tid, box in zip(self.clss, self.track_ids, self.boxes):
            if int(cls) == 0:
                person_ids.append(int(tid))
                person_boxes.append(box.tolist())
        return person_ids, person_boxes

    def _extract_pose_keypoints(self):
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

    def _build_label(self, name, is_known, attributes, confidence):
        label = name if is_known else "Unknown"
        if self.config.show_attributes and attributes:
            attr_str = FaceAttributeAnalyzer.format(
                attributes.get("age", 0), attributes.get("gender", 0), attributes.get("emotion"), attributes.get("pose")
            )
            label = f"{name}\n{attr_str}"
        return label

    def _annotate_person(self, annotator, box, label, person_id, person_box, color):
        # Use the color passed from processing logic
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

    def __call__(self, im0: np.ndarray) -> SolutionResults:
        self._current_frame = im0
        self.stats["frames_processed"] += 1

        # YOLO tracking via VisionEye
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)

        # Extract person data
        person_ids, person_boxes = self._extract_person_data()

        # Pose keypoints (if enabled)
        self._pose_keypoints = self._extract_pose_keypoints()

        # New person detection
        new_person_ids = self.new_person_tracker.update(person_ids)

        # Should we run face recognition this frame?
        run_fr = self.frame_controller.should_run_face_recognition(new_person_ids)

        face_boxes, face_embeddings, _, face_attributes = [], [], [], []
        if run_fr:
            self.stats["fr_runs"] += 1
            if new_person_ids:
                self.stats["new_person_triggers"] += len(new_person_ids)
            face_boxes, face_embeddings, _, face_attributes = self.face_detector.detect(im0)
            self.stats["total_faces_detected"] += len(face_boxes)

        unknown_count = 0
        detected_persons = []

        for cls, tid, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            if int(cls) != 0:  # non-person object
                label = self.adjust_box_label(cls, float(conf) if conf else 0.0, tid)
                annotator.box_label(box, label=label, color=colors(int(tid), True))
                annotator.visioneye(box, self.vision_point)
                continue

            person_id = int(tid)
            person_box = box.tolist()

            # Determine color and label
            cached = self.person_cache.get(person_id)
            if cached:
                name = cached.name
                is_known = cached.is_known
                attrs = cached.attributes
                color = self.config.known_color if is_known else self.config.unknown_color
            else:
                if run_fr:
                    # Try to recognize now
                    name, is_known, attrs = self._process_person(
                        person_id, person_box, face_boxes, face_embeddings, face_attributes, run_fr
                    )
                    if is_known:
                        color = self.config.known_color
                    else:
                        # Check if we actually had a face
                        face_idx = self._associate_face_to_person(person_box, face_boxes)
                        if face_idx is not None:
                            color = self.config.unknown_color  # face present but unknown
                        else:
                            color = self.config.no_face_color  # no face detected
                else:
                    # Frame skip: no recognition, default to no_face_color
                    name = "Unknown"
                    is_known = False
                    attrs = {}
                    color = self.config.no_face_color

            if person_id in new_person_ids:
                self.new_person_tracker.mark_processed(person_id)

            if is_known:
                self.stats["known_persons_detected"] += 1
            else:
                unknown_count += 1
                self.stats["unknown_persons_detected"] += 1

            label = self._build_label(name, is_known, attrs, conf)
            self._annotate_person(annotator, box, label, person_id, person_box, color)

            detected_persons.append(
                {"id": person_id, "name": name, "is_known": is_known, "box": person_box, "attributes": attrs}
            )

        # Alarm handling
        alarm_threshold = self.CFG.get("records", 1)
        if unknown_count >= alarm_threshold:
            self._trigger_alarm(unknown_count)
            self._save_alarm_screenshots(detected_persons)
            self._log_event("ALARM", {"unknown_count": unknown_count, "persons": detected_persons})
        else:
            self._clear_alarm()

        output_frame = annotator.result()
        self.display_output(output_frame)

        # Statistics logging
        if self.stats["frames_processed"] % 30 == 0:
            self._logger.info(
                f"Frames:{self.stats['frames_processed']} FR:{self.stats['fr_runs']} "
                f"Cache:{self.stats['cache_hits']} Faces:{self.stats['total_faces_detected']} "
                f"Poses:{self.stats['poses_detected']} Skipped(Q:{self.stats['skipped_low_quality']} "
                f"Y:{self.stats['skipped_yaw']})"
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
        self.face_recognizer.clear_all_tracks()
        if self.screenshot_capturer:
            self.screenshot_capturer.reset_captured_ids()
        self._alarm_active = False
        self._logger.info("Security Guard reset")


# ============================================================================
# FACTORY FUNCTION (loads known faces from family_member folder)
# ============================================================================


def create_security_guard(
    config: Optional[SecurityGuardConfig] = None, face_directory: Optional[str] = None, **kwargs
) -> EnhancedSecurityGuard:
    cfg = config or SecurityGuardConfig()
    logger = _setup_logging("Factory")

    # Use GPU for embedding extraction during loading
    detector = InsightFaceDetector(cfg.insightface_model, cfg.insightface_det_size)

    embeddings, names = [], []
    face_dir = face_directory or cfg.family_member_dir
    if os.path.exists(face_dir):
        logger.info(f"Loading known faces from: {face_dir}")
        for person_name in os.listdir(face_dir):
            person_path = os.path.join(face_dir, person_name)
            if not os.path.isdir(person_path):
                continue
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
                        embeddings.append(face_embs[0])
                        names.append(person_name)
                        logger.info(f"Loaded: {person_name}/{img_file}")
                except Exception as e:
                    logger.warning(f"Failed loading {person_name}/{img_file}: {e}")
    else:
        logger.warning(f"Face directory not found: {face_dir}")

    return EnhancedSecurityGuard(config=cfg, known_embeddings=embeddings, known_names=names, **kwargs)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    config = SecurityGuardConfig(
        family_member_dir="../family_members",
        yolo_model="yolo26n-pose.pt",
        frame_skip=5,
        known_cache_ttl=60,
        unknown_cache_ttl=5,
        face_tolerance=0.55,
        min_face_quality=35,
        min_recognition_quality=30,
        max_face_yaw=30,
        min_confidence=0.6,
        capture_faces=True,
        enable_logging=True,
        show_attributes=True,
        enable_keypoints_display=False,
    )

    print("\n" + "=" * 60)
    print("Enhanced Security Guard - YOLO26 + InsightFace + Optimized")
    print("=" * 60 + "\n")

    video_path = "../media_files/WIN_20260227_22_00_29_Pro.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "output_optimized.avi"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    guard = create_security_guard(
        config=config,
        show=True,
        model=config.yolo_model,
        classes=[0],  # person only
        vision_point=(width // 2 - 250, height - 10),
        conf=0.3,
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
