# ============================================================================
# Improved Security Guard - Persistent Re‑ID, Best Face Cropping, Optimized Tracking
# ============================================================================

import cv2
import numpy as np
import pygame
import os
import time
import json
import logging
from pathlib import Path
from collections import OrderedDict, deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from threading import Lock
from contextlib import contextmanager
import warnings
import pickle

from insightface.app import FaceAnalysis
from ultralytics import solutions
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER

# ------------------------------
# Constants & Helpers
# ------------------------------
SIZE_WEIGHT = 0.30
BRIGHTNESS_WEIGHT = 0.15
SHARPNESS_WEIGHT = 0.25
CONTRAST_WEIGHT = 0.20
NORMALIZATION_EPS = 1e-5
DEFAULT_CACHE_TTL = 30.0
DEFAULT_MIN_FACE_SIZE = 40
DEFAULT_IOU_THRESHOLD = 0.8
DEFAULT_FACE_TRACK_MAX_AGE = 30
DEFAULT_MIN_FACE_QUALITY = 30.0

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

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1 = emb1 / (np.linalg.norm(emb1) + NORMALIZATION_EPS)
    emb2 = emb2 / (np.linalg.norm(emb2) + NORMALIZATION_EPS)
    return float(np.dot(emb1, emb2))

def align_face(img: np.ndarray, landmarks: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """Align face using eye centers and crop to target size."""
    if landmarks is None or len(landmarks) < 5:
        return None
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    eye_center = (left_eye + right_eye) / 2
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale=1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    # Estimate bounding box after rotation
    h, w = rotated.shape[:2]
    # Expand face region
    margin = 0.2
    face_w = (right_eye[0] - left_eye[0]) * (1 + 2 * margin)
    face_h = face_w  # approximate square
    x = eye_center[0] - face_w / 2
    y = eye_center[1] - face_h / 2
    x, y, x2, y2 = int(x), int(y), int(x + face_w), int(y + face_h)
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x or y2 <= y:
        return None
    face = rotated[y:y2, x:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, target_size)

# ------------------------------
# Quality Assessment with Pose
# ------------------------------
def assess_face_quality(frame: np.ndarray, box: List[float], pose: Optional[np.ndarray] = None) -> float:
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

    score = (size_score * SIZE_WEIGHT +
             bright_score * BRIGHTNESS_WEIGHT +
             sharp_score * SHARPNESS_WEIGHT +
             contrast_score * CONTRAST_WEIGHT)

    if pose is not None and len(pose) >= 3:
        yaw = abs(pose[1])
        pitch = abs(pose[0])
        # Penalize strong yaw or pitch
        if yaw > 30:
            score *= (1 - (yaw - 30) / 60)
        if pitch > 20:
            score *= (1 - (pitch - 20) / 40)
        score = max(0, score)
    return score

# ------------------------------
# Person Registry (with best face)
# ------------------------------
@dataclass
class RegistryEntry:
    persistent_id: int
    name: str
    category: str   # "KNOWN" or "UNKNOWN"
    embeddings: List[np.ndarray]          # recent embeddings (normalized)
    best_embedding: np.ndarray            # best quality embedding
    best_face: np.ndarray                 # aligned best face image (optional)
    last_seen: float
    attributes: Dict
    track_id_mapping: Dict[int, float]    # map YOLO track ID -> last seen time (for short-term re-identification)

class PersonRegistry:
    def __init__(self, similarity_threshold: float = 0.7, max_embeddings: int = 5,
                 track_timeout: float = 2.0):
        self.similarity_threshold = similarity_threshold
        self.max_embeddings = max_embeddings
        self.track_timeout = track_timeout
        self.entries: Dict[int, RegistryEntry] = {}
        self.next_id = 0
        self._lock = Lock()
        self._logger = _setup_logging("PersonRegistry")

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        return emb / (np.linalg.norm(emb) + NORMALIZATION_EPS)

    def find_match(self, embedding: np.ndarray, quality: float = 0.0) -> Optional[Tuple[int, float]]:
        embedding_norm = self._normalize(embedding)
        best_id = None
        best_sim = self.similarity_threshold
        with self._lock:
            for pid, entry in self.entries.items():
                # Use best embedding for matching
                sim = compute_cosine_similarity(embedding_norm, entry.best_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid
                # Also check recent embeddings
                for emb in entry.embeddings:
                    sim = compute_cosine_similarity(embedding_norm, emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = pid
        if best_id is not None:
            return best_id, best_sim
        return None

    def add_person(self, embedding: np.ndarray, name: str, category: str,
                   attributes: Dict, face_img: Optional[np.ndarray] = None,
                   quality: float = 0.0) -> int:
        embedding_norm = self._normalize(embedding)
        with self._lock:
            pid = self.next_id
            self.next_id += 1
            entry = RegistryEntry(
                persistent_id=pid,
                name=name,
                category=category,
                embeddings=[embedding_norm],
                best_embedding=embedding_norm,
                best_face=face_img,
                last_seen=time.time(),
                attributes=attributes,
                track_id_mapping={}
            )
            self.entries[pid] = entry
        self._logger.debug(f"Added {category} {name} with ID {pid}")
        return pid

    def update_person(self, persistent_id: int, embedding: np.ndarray,
                      attributes: Dict, face_img: Optional[np.ndarray] = None,
                      quality: float = 0.0) -> None:
        embedding_norm = self._normalize(embedding)
        with self._lock:
            entry = self.entries.get(persistent_id)
            if entry is None:
                return
            # Update last seen
            entry.last_seen = time.time()
            # Update recent embeddings (rolling window)
            if len(entry.embeddings) >= self.max_embeddings:
                entry.embeddings.pop(0)
            entry.embeddings.append(embedding_norm)
            # Update best embedding if quality is higher
            if quality > 0 and quality > getattr(entry, '_best_quality', 0):
                entry.best_embedding = embedding_norm
                if face_img is not None:
                    entry.best_face = face_img
                entry._best_quality = quality
            # Update attributes
            entry.attributes.update(attributes)

    def get_info(self, persistent_id: int) -> Optional[Tuple[str, str, Dict, np.ndarray]]:
        with self._lock:
            entry = self.entries.get(persistent_id)
            if entry:
                return entry.name, entry.category, entry.attributes, entry.best_face
        return None

    def associate_track(self, persistent_id: int, track_id: int) -> None:
        """Associate a YOLO track ID with this persistent ID (for short-term re‑entry)."""
        with self._lock:
            entry = self.entries.get(persistent_id)
            if entry:
                entry.track_id_mapping[track_id] = time.time()

    def get_persistent_for_track(self, track_id: int) -> Optional[int]:
        """If a track ID was recently associated with a persistent ID, return it."""
        with self._lock:
            now = time.time()
            for pid, entry in self.entries.items():
                if track_id in entry.track_id_mapping:
                    last_seen = entry.track_id_mapping[track_id]
                    if now - last_seen <= self.track_timeout:
                        return pid
                    else:
                        del entry.track_id_mapping[track_id]
        return None

    def remove_old(self, max_age: float) -> None:
        now = time.time()
        with self._lock:
            to_remove = [pid for pid, e in self.entries.items() if now - e.last_seen > max_age]
            for pid in to_remove:
                del self.entries[pid]
                self._logger.debug(f"Removed old person {pid}")

# ------------------------------
# Face Screenshot Capturer (with alignment)
# ------------------------------
@dataclass
class FaceData:
    persistent_id: int
    box: List[float]
    quality: float
    frame: np.ndarray
    is_known: bool
    name: str
    aligned_face: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)

class FaceScreenshotCapturer:
    def __init__(self, output_dir: str = "../captured_faces", min_quality: float = DEFAULT_MIN_FACE_QUALITY):
        self.output_dir = output_dir
        self.min_quality = min_quality
        self.known_dir = os.path.join(output_dir, "known")
        self.unknown_dir = os.path.join(output_dir, "unknown")
        for d in [self.known_dir, self.unknown_dir]:
            os.makedirs(d, exist_ok=True)

        self.best_faces: Dict[int, FaceData] = {}
        self._captured_persistent_ids: set = set()
        self._lock = Lock()
        self._logger = _setup_logging("ScreenshotCapturer")
        self._logger.info(f"Screenshots dir: {output_dir}")

    def update_best_face(self, persistent_id: int, box: List[float], frame: np.ndarray,
                         is_known: bool, name: str, aligned_face: Optional[np.ndarray] = None,
                         pose: Optional[np.ndarray] = None) -> bool:
        quality = assess_face_quality(frame, box, pose)
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
                    aligned_face=aligned_face,
                )
                return True
        return False

    def save_unknown_face(self, frame: np.ndarray, box: List[float],
                          persistent_id: Optional[int] = None,
                          is_known: bool = False, name: str = "Unknown",
                          aligned_face: Optional[np.ndarray] = None) -> Optional[str]:
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
                    if persistent_id is not None:
                        filename = f"unknown_{persistent_id}_{timestamp}.jpg"
                    else:
                        self._unknown_count = getattr(self, '_unknown_count', 0) + 1
                        filename = f"unknown_{self._unknown_count}_{timestamp}.jpg"
                    path = os.path.join(self.unknown_dir, filename)
            success = cv2.imwrite(path, face)
            if success:
                self._logger.info(f"Saved {'known' if is_known else 'unknown'} face: {path}")
                return path
        except Exception as e:
            self._logger.error(f"Save face failed: {e}")
        return None

    def save_all(self) -> int:
        saved = 0
        with self._lock:
            for pid, data in self.best_faces.items():
                try:
                    path = self._save_face_image(pid, data)
                    if path:
                        saved += 1
                except Exception as e:
                    self._logger.error(f"Save failed for {pid}: {e}")
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
            filename = f"unknown_{persistent_id}_{data.quality:.0f}_{timestamp}.jpg"
            path = os.path.join(self.unknown_dir, filename)
        success = cv2.imwrite(path, face)
        return path if success else None

    def get_best_face(self, persistent_id: int) -> Optional[FaceData]:
        with self._lock:
            return self.best_faces.get(persistent_id)

    def reset_captured_ids(self) -> None:
        with self._lock:
            self._captured_persistent_ids.clear()

# ------------------------------
# Face Recognizer (with multiple embeddings per person)
# ------------------------------
class FaceRecognizer:
    def __init__(self, known_embeddings_dict: Dict[str, List[np.ndarray]],
                 tolerance: float = 0.5, smoothing_frames: int = 5, min_confidence: float = 0.6):
        self.tolerance = tolerance
        self.smoothing_frames = smoothing_frames
        self.min_confidence = min_confidence
        self._logger = _setup_logging("FaceRecognizer")
        self.known_dict = {}
        for name, emb_list in known_embeddings_dict.items():
            norm_list = [e / (np.linalg.norm(e) + NORMALIZATION_EPS) for e in emb_list]
            self.known_dict[name] = norm_list
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
        if not self.known_dict:
            return "Unknown", False, 0.0
        try:
            if track_id is not None and track_id in self._identity_history:
                name, conf, frames = self._identity_history[track_id]
                if frames >= 3 and conf > 0.8:
                    return name, True, conf
            if track_id is not None:
                embedding = self._smooth_embedding(track_id, embedding)

            query_norm = embedding / (np.linalg.norm(embedding) + NORMALIZATION_EPS)
            best_name = "Unknown"
            best_score = 0.0
            for name, emb_list in self.known_dict.items():
                for stored in emb_list:
                    sim = np.dot(query_norm, stored)
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

    def clear_all(self) -> None:
        self._embedding_buffers.clear()
        self._identity_history.clear()

# ------------------------------
# InsightFace Detector (GPU support)
# ------------------------------
class InsightFaceDetector:
    def __init__(self, model: str = "buffalo_l", det_size: Tuple[int, int] = (640, 640)):
        self.model_name = model
        self.det_size = det_size
        self._logger = _setup_logging("InsightFaceDetector")
        try:
            self.app = FaceAnalysis(name=model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=det_size)
            self._logger.info(f"InsightFace initialized: {model}")
        except Exception as e:
            self._logger.error(f"Failed to initialize: {e}")
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
            attrs = {
                "age": getattr(face, "age", None),
                "gender": getattr(face, "gender", None),
                "emotion": getattr(face, "emotion", np.array([0.5])),
                "pose": getattr(face, "pose", np.array([0, 0, 0])),
                "face_width": bbox[2] - bbox[0],
                "face_height": bbox[3] - bbox[1],
            }
            attributes.append(attrs)
        return boxes, embeddings, landmarks, attributes

# ------------------------------
# Configuration
# ------------------------------
@dataclass
class SecurityGuardConfig:
    frame_interval: int = 3
    face_recognition_interval: float = 0.0
    enable_new_person_detection: bool = True
    person_cache_ttl: float = DEFAULT_CACHE_TTL
    face_tolerance: float = 0.4
    embedding_smoothing_frames: int = 5
    min_face_size: Tuple[int, int] = (DEFAULT_MIN_FACE_SIZE, DEFAULT_MIN_FACE_SIZE)
    insightface_model: str = "buffalo_l"
    insightface_det_size: Tuple[int, int] = (640, 640)
    enable_face_tracking: bool = True
    face_track_max_age: int = DEFAULT_FACE_TRACK_MAX_AGE
    face_track_iou_threshold: float = DEFAULT_IOU_THRESHOLD
    enable_liveness: bool = True
    show_attributes: bool = True
    known_color: Tuple[int, int, int] = (0, 255, 0)
    unknown_color: Tuple[int, int, int] = (0, 165, 255)
    no_face_color: Tuple[int, int, int] = (128, 128, 128)
    capture_faces: bool = True
    screenshot_dir: Optional[str] = None
    min_face_quality: float = DEFAULT_MIN_FACE_QUALITY
    enable_logging: bool = True
    enable_keypoints_extraction: bool = False
    enable_keypoints_display: bool = False
    min_recognition_quality: float = 30.0
    max_face_yaw: float = 45.0
    known_cache_ttl: float = 60.0
    unknown_cache_ttl: float = 5.0
    min_confidence: float = 0.2
    reid_similarity_threshold: float = 0.7
    registry_max_embeddings_per_person: int = 5
    alarm_debounce_frames: int = 3

    def validate(self):
        # (kept as before)
        return self

# ------------------------------
# Security Guard Main Class
# ------------------------------
class EnhancedSecurityGuard(solutions.VisionEye):
    def __init__(self, *args, config: Optional[SecurityGuardConfig] = None,
                 known_embeddings_dict: Optional[Dict[str, List[np.ndarray]]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or SecurityGuardConfig()
        self.config.validate()
        self._logger = _setup_logging("SecurityGuard")
        self._initialize_components(known_embeddings_dict)
        self._initialize_stats()
        self._logger.info(f"Initialized with {len(known_embeddings_dict or {})} known persons")

    def _initialize_components(self, known_embeddings_dict):
        self.face_detector = InsightFaceDetector(self.config.insightface_model, self.config.insightface_det_size)
        self.face_recognizer = FaceRecognizer(known_embeddings_dict or {},
                                              self.config.face_tolerance,
                                              self.config.embedding_smoothing_frames,
                                              self.config.min_confidence)
        self.person_registry = PersonRegistry(similarity_threshold=self.config.reid_similarity_threshold,
                                              max_embeddings=self.config.registry_max_embeddings_per_person,
                                              track_timeout=2.0)
        self.screenshot_capturer = None
        if self.config.capture_faces:
            sd = self.config.screenshot_dir or str(_get_base_dir() / "captured_faces")
            self.screenshot_capturer = FaceScreenshotCapturer(sd, self.config.min_face_quality)
        self.sound_manager = SoundManager()  # defined earlier, but we need to include it
        self._alarm_active = False
        self._alarm_pending = 0
        self._track_to_persistent: Dict[int, int] = {}  # current track -> persistent ID
        self._lost_tracks: Dict[int, Dict] = {}         # track ID -> last box, last seen
        self._lost_track_timeout = 30  # frames
        self._frame_counter = 0

    def _initialize_stats(self):
        self.stats = {
            "frames_processed": 0, "fr_runs": 0, "cache_hits": 0, "new_person_triggers": 0,
            "total_faces_detected": 0, "known_persons_detected": 0, "unknown_persons_detected": 0,
            "poses_detected": 0, "screenshots_saved": 0, "alerts_triggered": 0,
            "skipped_low_quality": 0, "skipped_yaw": 0, "registry_matches": 0,
        }

    def _associate_face_to_person(self, face_box: List[float], person_boxes: List[List[float]]) -> Optional[int]:
        face_cx = (face_box[0] + face_box[2]) / 2
        face_cy = (face_box[1] + face_box[3]) / 2
        best_idx = None
        best_iou = 0
        for idx, pbox in enumerate(person_boxes):
            if pbox[0] <= face_cx <= pbox[2] and pbox[1] <= face_cy <= pbox[3]:
                # If multiple persons contain the same face center, pick the one with larger area
                area = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
                if area > best_iou:
                    best_iou = area
                    best_idx = idx
        return best_idx

    def _process_person(self, person_id: int, person_box: List[float],
                        face_boxes: List[List[float]], face_embeddings: List[np.ndarray],
                        face_landmarks: List[np.ndarray], face_attributes: List[Dict],
                        run_fr: bool) -> Tuple[str, bool, Dict, Optional[int], Optional[np.ndarray]]:
        # Check if we already have a persistent ID for this track (short-term)
        persistent_id = self._track_to_persistent.get(person_id)
        if persistent_id is not None:
            # Refresh last seen time in registry
            self.person_registry.associate_track(persistent_id, person_id)
            info = self.person_registry.get_info(persistent_id)
            if info:
                name, category, attrs, best_face = info
                is_known = (category == "KNOWN")
                return name, is_known, attrs, persistent_id, best_face
            else:
                # Registry entry disappeared
                persistent_id = None

        # Try to find a matching lost track by IoU
        if not run_fr:
            # Even if FR is off, we can try to restore identity from lost tracks
            best_lost = None
            best_iou = 0
            for lost_id, lost in self._lost_tracks.items():
                iou = self._compute_iou(person_box, lost["box"])
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_lost = lost_id
            if best_lost is not None:
                pid = self._lost_tracks[best_lost].get("persistent_id")
                if pid is not None:
                    self._track_to_persistent[person_id] = pid
                    self.person_registry.associate_track(pid, person_id)
                    info = self.person_registry.get_info(pid)
                    if info:
                        name, category, attrs, best_face = info
                        is_known = (category == "KNOWN")
                        return name, is_known, attrs, pid, best_face

        if not run_fr:
            return "Unknown", False, {}, None, None

        # Find associated face
        face_idx = self._associate_face_to_person(person_box, face_boxes)
        if face_idx is None or face_idx >= len(face_embeddings):
            return "Unknown", False, {}, None, None

        # Quality and yaw checks
        quality = assess_face_quality(self._current_frame, face_boxes[face_idx], face_attributes[face_idx].get("pose"))
        if quality < self.config.min_recognition_quality:
            self.stats["skipped_low_quality"] += 1
            return "Unknown", False, {}, None, None

        pose = face_attributes[face_idx].get("pose")
        if pose is not None and len(pose) >= 3 and abs(pose[1]) > self.config.max_face_yaw:
            self.stats["skipped_yaw"] += 1
            return "Unknown", False, {}, None, None

        embedding = face_embeddings[face_idx]
        attrs = face_attributes[face_idx]

        # Align face for best storage
        landmarks = face_landmarks[face_idx] if face_landmarks else None
        aligned_face = align_face(self._current_frame, landmarks) if landmarks is not None else None

        # Try registry match
        match = self.person_registry.find_match(embedding, quality)
        if match is not None:
            persistent_id, sim = match
            self.stats["registry_matches"] += 1
            name, category, reg_attrs, best_face = self.person_registry.get_info(persistent_id)
            is_known = (category == "KNOWN")
            # Update registry with new embedding and attributes
            self.person_registry.update_person(persistent_id, embedding, attrs, aligned_face, quality)
            self._track_to_persistent[person_id] = persistent_id
            self.person_registry.associate_track(persistent_id, person_id)
            # Update screenshot best face
            if self.screenshot_capturer:
                self.screenshot_capturer.update_best_face(persistent_id, face_boxes[face_idx], self._current_frame,
                                                          is_known, name, aligned_face, pose)
            return name, is_known, reg_attrs, persistent_id, best_face

        # No registry match, do face recognition
        name, is_known, confidence = self.face_recognizer.identify(embedding, track_id=person_id)
        category = "KNOWN" if is_known else "UNKNOWN"
        persistent_id = self.person_registry.add_person(embedding, name, category, attrs, aligned_face, quality)
        self._track_to_persistent[person_id] = persistent_id
        self.person_registry.associate_track(persistent_id, person_id)
        if self.screenshot_capturer:
            self.screenshot_capturer.update_best_face(persistent_id, face_boxes[face_idx], self._current_frame,
                                                      is_known, name, aligned_face, pose)
        if "pose" in attrs and attrs["pose"] is not None:
            self.stats["poses_detected"] += 1
        return name, is_known, attrs, persistent_id, aligned_face

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _trigger_alarm(self, unknown_count: int) -> None:
        self._alarm_pending += 1
        if self._alarm_pending >= self.config.alarm_debounce_frames and not self._alarm_active:
            if self.sound_manager.play_alarm():
                self._alarm_active = True
                self.stats["alerts_triggered"] += 1
                self._logger.warning(f"ALARM: {unknown_count} unknown(s)")

    def _clear_alarm(self) -> None:
        self._alarm_pending = 0
        if self._alarm_active:
            self.sound_manager.stop_alarm()
            self._alarm_active = False

    def _save_alarm_screenshots(self, detected_persons):
        if not self.screenshot_capturer:
            return
        for p in detected_persons:
            if p.get("persistent_id") is not None:
                best = self.screenshot_capturer.get_best_face(p["persistent_id"])
                if best:
                    self.screenshot_capturer.save_unknown_face(
                        self._current_frame, best.box, persistent_id=p["persistent_id"],
                        is_known=p["is_known"], name=p["name"], aligned_face=best.aligned_face)

    def __call__(self, im0: np.ndarray) -> SolutionResults:
        self._current_frame = im0
        self.stats["frames_processed"] += 1
        self._frame_counter += 1

        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)

        person_ids, person_boxes = [], []
        for cls, tid, box in zip(self.clss, self.track_ids, self.boxes):
            if int(cls) == 0:
                person_ids.append(int(tid))
                person_boxes.append(box.tolist())

        # Update lost tracks
        current_ids = set(person_ids)
        for tid in list(self._lost_tracks.keys()):
            self._lost_tracks[tid]["age"] += 1
            if self._lost_tracks[tid]["age"] > self._lost_track_timeout:
                del self._lost_tracks[tid]
        for tid, box in zip(person_ids, person_boxes):
            if tid in self._lost_tracks:
                del self._lost_tracks[tid]

        # Determine if FR should run
        run_fr = (self._frame_counter % max(1, self.config.frame_interval) == 0)

        # Face detection
        face_boxes, face_embeddings, face_landmarks, face_attributes = [], [], [], []
        if run_fr:
            self.stats["fr_runs"] += 1
            face_boxes, face_embeddings, face_landmarks, face_attributes = self.face_detector.detect(im0)
            self.stats["total_faces_detected"] += len(face_boxes)

        # Process each person
        unknown_count = 0
        detected_persons = []
        for cls, tid, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            if int(cls) != 0:
                self._annotate_object(annotator, cls, tid, box, conf)
                continue
            person_id = int(tid)
            person_box = box.tolist()
            name, is_known, attrs, persistent_id, best_face = self._process_person(
                person_id, person_box, face_boxes, face_embeddings, face_landmarks, face_attributes, run_fr)

            if not is_known and face_boxes:
                unknown_count += 1
                self.stats["unknown_persons_detected"] += 1
            elif is_known:
                self.stats["known_persons_detected"] += 1

            # Determine color
            if is_known:
                color = self.config.known_color
            else:
                if self._associate_face_to_person(person_box, face_boxes) is not None:
                    color = self.config.unknown_color
                else:
                    color = self.config.no_face_color

            # Build label
            label = name if is_known else "Unknown"
            if self.config.show_attributes and attrs:
                # Format attributes (using helper from original)
                attr_str = self._format_attributes(attrs)
                label = f"{label}\n{attr_str}"
            # Add confidence
            if is_known:
                conf_str = f"{conf*100:.1f}%" if conf is not None else "N/A"
                label = f"{label}\nConf:{conf_str}"

            final_label = f"ID {person_id}: {label}" if label else f"ID {person_id}"
            annotator.box_label(box, label=final_label, color=color)
            annotator.visioneye(box, self.vision_point)

            # Display best face thumbnail
            if best_face is not None and is_known:
                # Resize thumbnail and place near box
                thumb = cv2.resize(best_face, (60, 60))
                x1, y1 = int(box[0]), int(box[1]) - 70
                if y1 < 0: y1 = 0
                h, w = im0.shape[:2]
                if x1 + 60 < w and y1 + 60 < h:
                    im0[y1:y1+60, x1:x1+60] = thumb

            # Optional pose keypoints
            if self.config.enable_keypoints_display and hasattr(self.tracks, 'keypoints') and self.tracks.keypoints is not None:
                try:
                    kpts = self.tracks.keypoints.xy.cpu().tolist()
                    if len(kpts) > 0:
                        kpts_array = np.array(kpts[0], dtype=np.float32)
                        annotator.kpts(kpts_array, shape=im0.shape[:2], kpt_line=True)
                except:
                    pass

            detected_persons.append({
                "id": person_id, "persistent_id": persistent_id,
                "name": name, "is_known": is_known, "box": person_box, "attributes": attrs
            })

        # Alarm with debouncing
        alarm_threshold = self.CFG.get("records", 1)
        if unknown_count >= alarm_threshold:
            self._trigger_alarm(unknown_count)
            if self._alarm_active:
                self._save_alarm_screenshots(detected_persons)
        else:
            self._clear_alarm()

        # Update lost tracks after processing
        for tid in person_ids:
            if tid not in current_ids:
                # Track disappeared: store it
                idx = person_ids.index(tid) if tid in person_ids else None
                if idx is not None:
                    self._lost_tracks[tid] = {
                        "box": person_boxes[idx],
                        "age": 0,
                        "persistent_id": self._track_to_persistent.get(tid)
                    }

        output_frame = annotator.result()
        self.display_output(output_frame)

        # Overlay stats
        cv2.putText(output_frame, f"Tracks:{len(person_ids)} Known:{self.stats['known_persons_detected']} Unknown:{self.stats['unknown_persons_detected']}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(output_frame, f"FR:{'ON' if run_fr else 'OFF'} Faces:{len(face_boxes)}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        return SolutionResults(plot_im=output_frame, total_tracks=len(self.track_ids))

    def _format_attributes(self, attrs):
        # Simplified version; you can reuse FaceAttributeAnalyzer from original
        age = attrs.get("age", 0)
        gender = attrs.get("gender", 0)
        if isinstance(gender, (int, np.integer)):
            gender_str = "male" if gender == 0 else "female"
        else:
            gender_str = "?"
        return f"{int(age)}y {gender_str}"

    def save_screenshots(self) -> int:
        return self.screenshot_capturer.save_all() if self.screenshot_capturer else 0

    def reset(self) -> None:
        self.person_registry.remove_old(60)  # optional cleanup
        self._track_to_persistent.clear()
        self._lost_tracks.clear()
        self._alarm_active = False
        self._alarm_pending = 0
        if self.screenshot_capturer:
            self.screenshot_capturer.reset_captured_ids()
        self._logger.info("Reset")

# ------------------------------
# SoundManager (simplified)
# ------------------------------
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
            self._load_alarm_sound()

    def _load_alarm_sound(self):
        try:
            base_dir = _get_base_dir()
            alarm_file = base_dir / "../media_files/Alarm-sound-samples/humordome-security-alert-sound-453297.mp3"
            if alarm_file.exists():
                pygame.mixer.init()
                self._initialized_pygame = True
                pygame.mixer.music.load(str(alarm_file))
                self._alarm_loaded = True
                self._logger.info("Alarm loaded")
            else:
                self._logger.warning("Alarm file not found")
        except Exception as e:
            self._logger.warning(f"Alarm load failed: {e}")

    def play_alarm(self) -> bool:
        if not self._alarm_loaded:
            return False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
                return True
        except Exception as e:
            self._logger.error(f"Play failed: {e}")
        return False

    def stop_alarm(self):
        try:
            if self._alarm_loaded and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        except:
            pass

# ------------------------------
# Factory function (load known faces)
# ------------------------------
def create_security_guard(config: Optional[SecurityGuardConfig] = None,
                          face_directory: Optional[str] = None,
                          **kwargs) -> EnhancedSecurityGuard:
    cfg = config or SecurityGuardConfig()
    logger = _setup_logging("Factory")
    detector = InsightFaceDetector(cfg.insightface_model, cfg.insightface_det_size)
    embeddings_dict = {}
    face_dir = face_directory or str(_get_base_dir() / "../known_faces")
    if os.path.exists(face_dir):
        logger.info(f"Loading known faces from {face_dir}")
        for person_name in os.listdir(face_dir):
            person_path = os.path.join(face_dir, person_name)
            if not os.path.isdir(person_path):
                continue
            person_embs = []
            for img_file in os.listdir(person_path):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(person_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    boxes, embs, _, _ = detector.detect(img)
                    if embs:
                        person_embs.append(embs[0])
                        logger.debug(f"Loaded {person_name}/{img_file}")
                except Exception as e:
                    logger.warning(f"Failed {person_name}/{img_file}: {e}")
            if person_embs:
                embeddings_dict[person_name] = person_embs
                logger.info(f"Loaded {len(person_embs)} embeddings for {person_name}")
    else:
        logger.warning(f"Face directory not found: {face_dir}")

    return EnhancedSecurityGuard(config=cfg, known_embeddings_dict=embeddings_dict, **kwargs)

# ------------------------------
# Main entry
# ------------------------------
if __name__ == "__main__":
    config = SecurityGuardConfig(
        frame_interval=10,
        enable_new_person_detection=True,
        face_tolerance=0.5,
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
    )
    print("\n" + "="*60)
    print("Enhanced Security Guard - Improved Re-ID & Best Face")
    print("="*60 + "\n")
    video_path = "../media_files/WIN_20260227_22_00_29_Pro.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = "output_improved.avi"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    guard = create_security_guard(
        config=config,
        show=True,
        model="yolo26m-pose.pt",
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
    print("\n" + "="*60)
    print("SUMMARY")
    for key, val in guard.stats.items():
        print(f"  {key}: {val}")
    print("="*60)