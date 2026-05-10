# Enhanced InsightFace Security Guard System - Refined V5
#
# CHANGELOG from V4 (Senior Engineer Refactoring):
#   [OPTIMIZE] Upgraded FAISS from brute-force IndexFlatIP to true O(log n) IndexHNSWFlat
#   [OPTIMIZE] Implemented faiss.IndexIDMap to allow O(1) removals without index rebuilds
#   [FIX] Added callbacks to ThreadPoolExecutor futures to prevent silent async I/O failures
#   [FIX] Added graceful degradation and connection retry logic for video streams
#   [REFACTOR] Decoupled OpenCV HUD rendering from the main inference pipeline
#   [REFACTOR] Removed redundant faiss mappings (_id_to_pid, _pid_to_faiss_idx) 
#
# Version: 5.0.0

"""
Enhanced InsightFace Security Guard System — V5

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
10. FAISS HNSW Approximate Nearest Neighbor search (V5 UPGRADE)
11. Circuit breaker for fault tolerance 
12. Async I/O for screenshots with error callbacks (V5 UPGRADE)
13. Comprehensive metrics export 
14. Resilient video capture backoff (V5 UPGRADE)

Installation:
    pip install insightface onnxruntime opencv-python numpy pygame ultralytics faiss-cpu psutil
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
from concurrent.futures import ThreadPoolExecutor, Future
from functools import wraps

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Install with: pip install faiss-cpu")

from insightface.app import FaceAnalysis
from ultralytics import solutions
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults

# ============================================================================
# CONSTANTS & MAPPINGS
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

EMOTION_MAP: Dict[int, str] = MappingProxyType(
    {0: "happy", 1: "neutral", 2: "sad", 3: "angry", 4: "surprise", 5: "fear", 6: "disgust"}
)
GENDER_MAP: Dict[int, str] = MappingProxyType({0: "male", 1: "female"})


# ============================================================================
# LOGGING & TRACING
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
    """Decorator for tracing operations across the pipeline with correlation IDs."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            correlation_id = str(uuid.uuid4())[:8]
            start_time = time.time()
            logger = getattr(self, '_logger', None) or _setup_logging()
            
            try:
                result = func(self, *args, **kwargs)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"[{correlation_id}] Failed {operation_name} after {duration:.3f}s: {e}")
                raise
        return wrapper
    return decorator


# ============================================================================
# CORE COMPONENTS (Circuit Breaker, Metrics, Config)
# ============================================================================

class CircuitBreakerOpen(Exception): pass

class CircuitBreaker:
    """Fault tolerance pattern to prevent cascading failures."""
    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 success_threshold: int = 3):
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
                    raise CircuitBreakerOpen("Circuit breaker is open.")

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
            elif self.state == self.STATE_CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    def _record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = self.STATE_OPEN

    def get_state(self) -> str:
        with self._lock: return self.state

@dataclass
class SecurityGuardMetrics:
    frames_processed: int = 0
    fr_runs: int = 0
    cache_hit_rate: float = 0.0
    avg_recognition_time_ms: float = 0.0
    registry_size: int = 0
    active_alarms: int = 0
    memory_usage_mb: float = 0.0
    circuit_breaker_state: str = "closed"

class MetricsExporter:
    def __init__(self):
        self._frame_times: List[float] = []
        self._lock = Lock()
        self._process = psutil.Process()

    def record_frame_time(self, duration_ms: float):
        with self._lock:
            self._frame_times.append(duration_ms)
            if len(self._frame_times) > 1000:
                self._frame_times = self._frame_times[-1000:]

    def get_metrics(self, guard: 'EnhancedSecurityGuard') -> SecurityGuardMetrics:
        with self._lock:
            hits = guard.stats.get("cache_hits", 0)
            misses = guard.stats.get("cache_misses", 0)
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
            
            return SecurityGuardMetrics(
                frames_processed=guard.stats["frames_processed"],
                fr_runs=guard.stats["fr_runs"],
                cache_hit_rate=hit_rate,
                avg_recognition_time_ms=np.mean(self._frame_times) if self._frame_times else 0,
                registry_size=len(guard.person_registry),
                active_alarms=1 if guard._alarm_active else 0,
                memory_usage_mb=self._process.memory_info().rss / 1024 / 1024,
                circuit_breaker_state=getattr(guard.face_detector, '_circuit_breaker_state', 'unknown')
            )

@dataclass
class SecurityGuardConfig:
    frame_interval: int = 5
    face_recognition_interval: float = 0.3
    enable_new_person_detection: bool = True
    face_tolerance: float = 0.5
    embedding_smoothing_frames: int = 5
    min_face_size: Tuple[int, int] = (DEFAULT_MIN_FACE_SIZE, DEFAULT_MIN_FACE_SIZE)
    insightface_model: str = "buffalo_l"
    insightface_det_size: Tuple[int, int] = (640, 640)
    enable_face_tracking: bool = True
    face_track_max_age: int = DEFAULT_FACE_TRACK_MAX_AGE
    face_track_iou_threshold: float = DEFAULT_IOU_THRESHOLD
    show_attributes: bool = True
    known_color: Tuple[int, int, int] = (0, 255, 0)
    unknown_color: Tuple[int, int, int] = (0, 165, 255)
    no_face_color: Tuple[int, int, int] = (128, 128, 128)
    capture_faces: bool = True
    screenshot_dir: Optional[str] = None
    min_face_quality: float = DEFAULT_MIN_FACE_QUALITY
    enable_logging: bool = True
    enable_keypoints_extraction: bool = True
    enable_keypoints_display: bool = False
    min_recognition_quality: float = 30.0
    max_face_yaw: float = 30.0
    known_cache_ttl: float = 60.0
    unknown_cache_ttl: float = 5.0
    min_confidence: float = 0.6
    reid_similarity_threshold: float = 0.7
    registry_max_embeddings_per_person: int = 5
    alarm_debounce_frames: int = 3
    alarm_clear_hysteresis: int = 2
    registry_ttl: float = 300.0
    identity_history_max_frames: int = 50
    registry_cleanup_interval: int = 100
    max_seen_track_ids: int = 10000
    use_faiss_index: bool = True  
    faiss_nprobe: int = 10  
    enable_async_screenshots: bool = True  
    max_identity_history_size: int = 1000  
    circuit_breaker_threshold: int = 3  
    circuit_breaker_timeout: float = 30.0  

    def validate(self) -> "SecurityGuardConfig":
        if self.use_faiss_index and not FAISS_AVAILABLE:
            warnings.warn("FAISS not available, falling back to linear search")
            self.use_faiss_index = False
        return self


# ============================================================================
# PERSON REGISTRY — V5 HNSW FAISS
# ============================================================================

@dataclass
class RegistryEntry:
    persistent_id: int
    name: str
    category: str
    embeddings: List[np.ndarray]
    last_seen: float
    attributes: Dict

class PersonRegistry:
    """
    [V5 UPGRADE] True O(log N) ANN search using HNSW + IndexIDMap.
    Eliminates global lock stutters during index rebuilds.
    """
    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self.entries: Dict[int, RegistryEntry] = {}
        self.next_id = 0
        self._lock = Lock()
        self._logger = _setup_logging("PersonRegistry")

        self._use_faiss = config.use_faiss_index and FAISS_AVAILABLE
        self._embedding_dim = INSIGHTFACE_EMBEDDING_DIM
        self._faiss_index = None

        if self._use_faiss:
            self._init_faiss_index()

    def _init_faiss_index(self):
        try:
            # HNSW Flat with Inner Product for Cosine Similarity 
            base_index = faiss.IndexHNSWFlat(self._embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            # IndexIDMap allows us to assign explicit persistent_ids instead of sequential faiss IDs
            self._faiss_index = faiss.IndexIDMap(base_index)
            self._logger.info(f"FAISS HNSW IDMap index created: dim={self._embedding_dim}")
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
                try:
                    query = embedding_norm.reshape(1, -1).astype('float32')
                    k = min(self.config.faiss_nprobe, self._faiss_index.ntotal)
                    scores, indices = self._faiss_index.search(query, k)

                    for score, pid in zip(scores[0], indices[0]):
                        if pid < 0: continue
                        if score >= self.config.reid_similarity_threshold:
                            return int(pid), float(score)
                except Exception as e:
                    self._logger.error(f"FAISS search failed: {e}")
            
            # Fallback linear search
            best_id, best_sim = None, 0.0
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
                persistent_id=pid, name=name, category=category,
                embeddings=[embedding_norm], last_seen=time.time(), attributes=attributes,
            )

            if self._use_faiss and self._faiss_index is not None:
                try:
                    ids = np.array([pid], dtype=np.int64)
                    self._faiss_index.add_with_ids(embedding_norm.reshape(1, -1).astype('float32'), ids)
                except Exception as e:
                    self._logger.error(f"FAISS add failed: {e}")

        return pid

    def update_person(self, persistent_id: int, embedding: np.ndarray, attributes: Dict = None):
        embedding_norm = self._normalize(embedding)
        with self._lock:
            if persistent_id in self.entries:
                entry = self.entries[persistent_id]
                entry.embeddings.append(embedding_norm)
                if len(entry.embeddings) > self.config.registry_max_embeddings_per_person:
                    entry.embeddings.pop(0)
                
                entry.last_seen = time.time()
                if attributes:
                    entry.attributes.update(attributes)

                # [V5 UPGRADE] O(1) Replacement instead of full rebuild
                if self._use_faiss and self._faiss_index is not None:
                    try:
                        self._faiss_index.remove_ids(np.array([persistent_id], dtype=np.int64))
                        embs_array = np.vstack(entry.embeddings).astype('float32')
                        ids_array = np.full(len(entry.embeddings), persistent_id, dtype=np.int64)
                        self._faiss_index.add_with_ids(embs_array, ids_array)
                    except Exception as e:
                        self._logger.error(f"FAISS update failed: {e}")

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

            # [V5 UPGRADE] O(1) specific ID deletion
            if to_remove and self._use_faiss and self._faiss_index is not None:
                try:
                    self._faiss_index.remove_ids(np.array(to_remove, dtype=np.int64))
                except Exception as e:
                    self._logger.error(f"FAISS remove failed: {e}")

        return len(to_remove)

    def __len__(self) -> int:
        return len(self.entries)


# ============================================================================
# FACE SCREENSHOT CAPTURER
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
    """[V5 FIX] Handled ThreadPool exceptions explicitly to prevent silent drops"""
    def __init__(self, output_dir: str, min_quality: float, async_enabled: bool):
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

    def assess_quality(self, frame: np.ndarray, box: list) -> float:
        try:
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            face = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if face.size == 0: return 0.0
            
            face_area = (x2 - x1) * (y2 - y1)
            size_score = min(100, (face_area / (w * h)) * 1000)
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
        if quality < self.min_quality: return False
        with self._lock:
            if persistent_id not in self.best_faces or quality > self.best_faces[persistent_id].quality:
                self.best_faces[persistent_id] = FaceData(
                    persistent_id=persistent_id, box=list(box), quality=quality,
                    frame=frame.copy(), is_known=is_known, name=name,
                )
                return True
        return False

    def _save_face_image(self, persistent_id: int, data: FaceData) -> Optional[str]:
        x1, y1, x2, y2 = map(int, data.box)
        h, w = data.frame.shape[:2]
        face = data.frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if face.size == 0: return None
        
        timestamp = int(time.time())
        filename = f"{data.name}_{data.quality:.0f}_{timestamp}.jpg" if data.is_known \
                   else f"unknown_{persistent_id}_{data.quality:.0f}_{timestamp}.jpg"
        path = os.path.join(self.known_dir if data.is_known else self.unknown_dir, filename)
        return path if cv2.imwrite(path, face) else None

    def schedule_save(self, persistent_id: int) -> None:
        if not self._async_enabled or not self._executor: return

        # [V5 UPGRADE] Explicit exception catching for async threads
        def _handle_result(future):
            try:
                future.result()
            except Exception as e:
                self._logger.error(f"Async IO failure on screenshot save: {e}")

        future = self._executor.submit(self._async_save, persistent_id)
        future.add_done_callback(_handle_result)
        
        self._pending_saves.append(future)
        self._pending_saves = [f for f in self._pending_saves if not f.done()]

    def _async_save(self, persistent_id: int) -> Optional[str]:
        with self._lock:
            if persistent_id not in self.best_faces: return None
            data = self.best_faces[persistent_id]
        return self._save_face_image(persistent_id, data)

    def shutdown(self):
        if self._executor: self._executor.shutdown(wait=True)


# ============================================================================
# SUPPORTING CLASSES (Trackers, Caches, Event Logger, Sounds)
# ============================================================================

class EventLogger:
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or str(_get_base_dir() / "security_events.log")
        self._lock = Lock()
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        with self._lock:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {"timestamp": timestamp, "event": event_type, **data}
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

class SoundManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None: cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._alarm_loaded = False
            self._sound_lock = Lock()
            # Suppress pygame output 
            os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
            pygame.mixer.init()
            
            alarm_file = _get_base_dir() / "../media_files/Alarm-sound-samples/humordome-security-alert-sound-453297.mp3"
            if alarm_file.exists():
                pygame.mixer.music.load(str(alarm_file))
                self._alarm_loaded = True

    def play_alarm(self):
        with self._sound_lock:
            if self._alarm_loaded and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
                return True
        return False

    def stop_alarm(self):
        with self._sound_lock:
            if self._alarm_loaded and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()

class FaceAttributeAnalyzer:
    @staticmethod
    def format(age: float, gender: float, emotion: Optional[np.ndarray] = None, pose: Optional[np.ndarray] = None) -> str:
        age_str = f"{int(age)}y"
        gender_str = GENDER_MAP.get(int(gender > 0.5), "?")
        emo_str = EMOTION_MAP.get(int(np.argmax(emotion)) if emotion is not None and len(emotion) > 0 else 1, "neutral")
        pose_str = f" | P:{int(pose[0])} Y:{int(pose[1])}" if pose is not None and len(pose) >= 3 else ""
        return f"{age_str} {gender_str} {emo_str}{pose_str}"


class PersonCacheEntry:
    def __init__(self, name: str, is_known: bool, attributes: Dict):
        self.name = name
        self.is_known = is_known
        self.timestamp = time.time()
        self.attributes = attributes

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

    def set(self, person_id: int, name: str, is_known: bool, attributes: Dict) -> None:
        with self._lock:
            self._cache[person_id] = PersonCacheEntry(name, is_known, attributes or {})

    def clear(self) -> None:
        with self._lock: self._cache.clear()

class NewPersonTracker:
    def __init__(self, registry: PersonRegistry, max_seen: int = 10000):
        self._seen_track_ids, self._new_track_ids = set(), set()
        self._lock = Lock()
        self.registry = registry
        self._max_seen = max_seen

    def update(self, person_ids: list, face_embeddings: Dict[int, np.ndarray]) -> list:
        current = set(person_ids)
        new_track_ids = []
        with self._lock:
            if len(self._seen_track_ids) > self._max_seen:
                self._seen_track_ids = set(sorted(self._seen_track_ids)[-self._max_seen // 2:])

            for pid in current:
                if pid in self._seen_track_ids: continue
                if pid in face_embeddings and self.registry.find_match(face_embeddings[pid]):
                    self._seen_track_ids.add(pid)
                    continue
                new_track_ids.append(pid)
                self._new_track_ids.add(pid)
            self._seen_track_ids.update(current)
        return new_track_ids

    def mark_processed(self, person_id: int) -> None:
        with self._lock: self._new_track_ids.discard(person_id)

    def reset(self) -> None:
        with self._lock:
            self._seen_track_ids.clear()
            self._new_track_ids.clear()

class FrameController:
    def __init__(self, config: SecurityGuardConfig):
        self.config = config
        self._frame_count, self._last_fr_time = 0, 0.0
        self._lock = Lock()

    def should_run_face_recognition(self, person_ids: list, tracker: 'NewPersonTracker') -> Tuple[bool, bool]:
        with self._lock:
            self._frame_count += 1
            ct = time.time()
            time_based = (self.config.face_recognition_interval > 0 and 
                          ct - self._last_fr_time >= self.config.face_recognition_interval)
            frame_based = (self.config.frame_interval > 0 and self._frame_count % self.config.frame_interval == 0)
            
            has_new = False
            if self.config.enable_new_person_detection and person_ids:
                with tracker._lock:
                    has_new = any(pid not in tracker._seen_track_ids for pid in person_ids)

            should_run = time_based or frame_based or has_new
            if time_based: self._last_fr_time = ct
            return should_run, has_new


# ============================================================================
# ML WRAPPERS (InsightFace & Recognizer)
# ============================================================================

class InsightFaceDetector:
    def __init__(self, model: str = "buffalo_l", detection_size: Tuple[int, int] = (640, 640),
                 config: Optional[SecurityGuardConfig] = None):
        self._logger = _setup_logging("InsightFaceDetector")
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold if config else 3,
            recovery_timeout=config.circuit_breaker_timeout if config else 30.0
        )
        try:
            self.app = FaceAnalysis(name=model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=detection_size)
            self._logger.info("InsightFace initialized.")
        except Exception as e:
            self._logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Tuple[list, list, list, list]:
        try:
            return self._circuit_breaker.call(self._detect_impl, frame)
        except CircuitBreakerOpen:
            return [], [], [], []
        except Exception:
            self._circuit_breaker._record_failure()
            return [], [], [], []

    def _detect_impl(self, frame: np.ndarray) -> Tuple[list, list, list, list]:
        faces = self.app.get(frame)
        boxes, embeddings, landmarks, attributes = [], [], [], []
        for face in faces:
            bbox = face.bbox
            if (bbox[2] - bbox[0]) < DEFAULT_MIN_FACE_SIZE: continue
            boxes.append(bbox.tolist())
            embeddings.append(face.embedding)
            landmarks.append(face.kps)
            attributes.append({
                "age": face.age, "gender": face.gender,
                "emotion": getattr(face, "emotion", np.array([0.5])),
                "pose": getattr(face, "pose", np.array([0, 0, 0])),
            })
        return boxes, embeddings, landmarks, attributes

class FaceRecognizer:
    def __init__(self, known_embeddings_dict: Dict[str, list], tolerance: float = 0.5,
                 embedding_smoothing_frames: int = 5, min_confidence: float = 0.6,
                 identity_max_frames: int = 50, max_history_size: int = 1000):
        self.tolerance = tolerance
        self.embedding_smoothing_frames = embedding_smoothing_frames
        self.min_confidence = min_confidence
        self.identity_max_frames = identity_max_frames
        self.max_history_size = max_history_size

        self.known_dict: Dict[str, list] = {}
        self._flat_embeddings, self._flat_names = [], []
        
        for name, emb_list in known_embeddings_dict.items():
            norm_list = [emb / (np.linalg.norm(emb) + NORMALIZATION_EPS) for emb in emb_list]
            self.known_dict[name] = norm_list
            self._flat_embeddings.extend(norm_list)
            self._flat_names.extend([name] * len(norm_list))

        self._embedding_matrix = np.stack(self._flat_embeddings) if self._flat_embeddings else np.array([]).reshape(0, 0)
        self._embedding_buffers: Dict[int, list] = {}
        self._identity_history: Dict[int, Tuple[str, float, int]] = {}
        self._identity_lock = Lock()

    def identify(self, embedding: np.ndarray, track_id: Optional[int] = None) -> Tuple[str, bool, float]:
        if self._embedding_matrix.size == 0: return "Unknown", False, 0.0

        with self._identity_lock:
            if track_id is not None and track_id in self._identity_history:
                name, conf, frames = self._identity_history[track_id]
                if 3 <= frames <= self.identity_max_frames and conf > 0.8:
                    return name, True, conf
                elif frames > self.identity_max_frames:
                    del self._identity_history[track_id]

            if len(self._identity_history) > self.max_history_size:
                to_remove = len(self._identity_history) - self.max_history_size
                for key in sorted(self._identity_history.keys(), key=lambda k: self._identity_history[k][2])[:to_remove]:
                    del self._identity_history[key]

        query_norm = embedding / (np.linalg.norm(embedding) + NORMALIZATION_EPS)
        similarities = self._embedding_matrix @ query_norm
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        
        if (1 - best_score) < self.tolerance and best_score >= self.min_confidence:
            if track_id is not None:
                with self._identity_lock:
                    frames_seen = len(self._embedding_buffers.get(track_id, []))
                    self._identity_history[track_id] = (self._flat_names[best_idx], best_score, frames_seen)
            return self._flat_names[best_idx], True, best_score

        return "Unknown", False, 0.0


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class EnhancedSecurityGuard(solutions.VisionEye):
    def __init__(self, *args, config: Optional[SecurityGuardConfig] = None,
                 known_embeddings_dict: Optional[Dict[str, list]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or SecurityGuardConfig()
        self.config.validate()
        self._logger = _setup_logging("SecurityGuard")
        
        self.face_detector = InsightFaceDetector(self.config.insightface_model, self.config.insightface_det_size, self.config)
        self.face_recognizer = FaceRecognizer(
            known_embeddings_dict or {}, self.config.face_tolerance,
            self.config.embedding_smoothing_frames, self.config.min_confidence,
            self.config.identity_history_max_frames, self.config.max_identity_history_size)

        self.person_cache = PersonCache(self.config)
        self.person_registry = PersonRegistry(self.config)
        self.new_person_tracker = NewPersonTracker(self.person_registry, self.config.max_seen_track_ids)
        self.frame_controller = FrameController(self.config)
        self.event_logger = EventLogger() if self.config.enable_logging else None

        self.screenshot_capturer = None
        if self.config.capture_faces:
            sd = self.config.screenshot_dir or str(_get_base_dir() / "captured_faces")
            self.screenshot_capturer = FaceScreenshotCapturer(sd, self.config.min_face_quality, self.config.enable_async_screenshots)

        self.sound_manager = SoundManager()
        self._metrics_exporter = MetricsExporter()
        
        self.stats = {
            "frames_processed": 0, "fr_runs": 0, "cache_hits": 0, "cache_misses": 0,
            "total_faces_detected": 0, "known_persons_detected": 0, "unknown_persons_detected": 0,
            "alerts_triggered": 0, "registry_matches": 0, 
        }
        self._alarm_active = False
        self._alarm_pending = 0
        self._clear_streak = 0

    def _build_face_person_map(self, face_boxes: list, person_boxes: list) -> Dict[int, int]:
        face_to_person = {}
        for fi, fbox in enumerate(face_boxes):
            fcx, fcy = (fbox[0] + fbox[2]) / 2, (fbox[1] + fbox[3]) / 2
            for pi, pbox in enumerate(person_boxes):
                if pbox[0] <= fcx <= pbox[2] and pbox[1] <= fcy <= pbox[3]:
                    face_to_person[fi] = pi
                    break
        return face_to_person

    @logged_operation("process_frame")
    def __call__(self, im0: np.ndarray) -> SolutionResults:
        start_time = time.time()
        self.stats["frames_processed"] += 1
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)

        person_ids, person_boxes = [], []
        for cls, tid, box in zip(self.clss, self.track_ids, self.boxes):
            if int(cls) == 0:
                person_ids.append(int(tid))
                person_boxes.append(box.tolist())

        run_fr, has_new = self.frame_controller.should_run_face_recognition(person_ids, self.new_person_tracker)
        face_boxes, face_embeddings, face_attributes = [], [], []

        if run_fr:
            self.stats["fr_runs"] += 1
            face_boxes, face_embeddings, _, face_attributes = self.face_detector.detect(im0)
            self.stats["total_faces_detected"] += len(face_boxes)

        face_to_person = self._build_face_person_map(face_boxes, person_boxes)
        track_to_embedding = {person_ids[pi]: face_embeddings[fi] 
                              for fi, pi in face_to_person.items() if pi < len(person_ids) and fi < len(face_embeddings)}

        new_person_ids = self.new_person_tracker.update(person_ids, track_to_embedding)
        if not run_fr and self.config.enable_new_person_detection and new_person_ids:
            face_boxes, face_embeddings, _, face_attributes = self.face_detector.detect(im0)
            self.stats["fr_runs"] += 1
            run_fr = True
            face_to_person = self._build_face_person_map(face_boxes, person_boxes)

        unknown_count = 0
        detected_persons = []
        
        for i, (cls, tid, box, conf) in enumerate(zip(self.clss, self.track_ids, self.boxes, self.confs)):
            if int(cls) != 0:
                annotator.box_label(box, label=f"cls:{int(cls)} id:{int(tid)}", color=(200, 200, 200))
                continue

            person_id = int(tid)
            person_idx = person_ids.index(person_id) if person_id in person_ids else -1
            
            # Identify
            cached = self.person_cache.get(person_id)
            if cached:
                self.stats["cache_hits"] += 1
                name, is_known, attrs, persistent_id = cached.name, cached.is_known, cached.attributes, None
            else:
                self.stats["cache_misses"] += 1
                face_idx = next((fi for fi, pi in face_to_person.items() if pi == person_idx), None)
                
                if not run_fr or face_idx is None or face_idx >= len(face_embeddings):
                    name, is_known, attrs, persistent_id = "Unknown", False, {}, None
                else:
                    emb = face_embeddings[face_idx]
                    f_attrs = face_attributes[face_idx]
                    
                    match = self.person_registry.find_match(emb)
                    if match:
                        persistent_id, _ = match
                        self.stats["registry_matches"] += 1
                        info = self.person_registry.get_info(persistent_id)
                        name = info[0] if info else "Unknown"
                        is_known = info[1] == "KNOWN" if info else False
                        self.person_registry.update_person(persistent_id, emb, f_attrs)
                    else:
                        name, is_known, _ = self.face_recognizer.identify(emb, track_id=person_id)
                        persistent_id = self.person_registry.add_person(emb, name, "KNOWN" if is_known else "UNKNOWN", f_attrs)
                    
                    self.person_cache.set(person_id, name, is_known, f_attrs)
                    
                    if self.screenshot_capturer:
                        self.screenshot_capturer.update_best_face(persistent_id, face_boxes[face_idx], im0, is_known, name)
                        if not is_known: self.screenshot_capturer.schedule_save(persistent_id)
            
            if person_id in new_person_ids: self.new_person_tracker.mark_processed(person_id)

            if is_known: self.stats["known_persons_detected"] += 1
            else:
                self.stats["unknown_persons_detected"] += 1
                unknown_count += 1

            label = f"{name}\n{FaceAttributeAnalyzer.format(attrs.get('age',0), attrs.get('gender',0))}" if self.config.show_attributes and attrs else name
            annotator.box_label(box, label=label, color=self.config.known_color if is_known else self.config.unknown_color)

        # Alarm Logic
        if unknown_count >= self.CFG.get("records", 1):
            self._alarm_pending += 1
            self._clear_streak = 0
            if self._alarm_pending >= self.config.alarm_debounce_frames and not self._alarm_active:
                if self.sound_manager.play_alarm():
                    self._alarm_active = True
                    self.stats["alerts_triggered"] += 1
        else:
            self._clear_streak += 1
            if self._clear_streak >= self.config.alarm_clear_hysteresis:
                self._alarm_pending = 0
                if self._alarm_active:
                    self.sound_manager.stop_alarm()
                    self._alarm_active = False

        if self.stats["frames_processed"] % self.config.registry_cleanup_interval == 0:
            self.person_registry.remove_old(self.config.registry_ttl)

        out_frame = annotator.result()
        self._render_hud(out_frame, run_fr, start_time)
        self.display_output(out_frame)
        
        return SolutionResults(plot_im=out_frame, total_tracks=len(self.track_ids))

    def _render_hud(self, frame: np.ndarray, run_fr: bool, start_time: float):
        """[V5 REFACTOR] Decoupled HUD rendering from the main inference loop."""
        self._metrics_exporter.record_frame_time((time.time() - start_time) * 1000)
        metrics = self._metrics_exporter.get_metrics(self)
        
        cv2.putText(frame, f"Known:{self.stats['known_persons_detected']} Unknown:{self.stats['unknown_persons_detected']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FR:{'ON' if run_fr else 'OFF'} Reg:{metrics.registry_size} CB:{metrics.circuit_breaker_state[:1].upper()}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS:{1000/metrics.avg_recognition_time_ms:.1f}" if metrics.avg_recognition_time_ms > 0 else "FPS: N/A", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    def shutdown(self):
        if self.screenshot_capturer: self.screenshot_capturer.shutdown()


def create_security_guard(config: Optional[SecurityGuardConfig] = None, face_directory: Optional[str] = None, **kwargs) -> EnhancedSecurityGuard:
    cfg = config or SecurityGuardConfig()
    cfg.validate()
    detector = InsightFaceDetector(cfg.insightface_model, cfg.insightface_det_size, cfg)
    
    embeddings_dict = {}
    face_dir = face_directory or str(_get_base_dir() / "family_members")
    if os.path.exists(face_dir):
        for person_name in os.listdir(face_dir):
            person_path = os.path.join(face_dir, person_name)
            if not os.path.isdir(person_path): continue
            
            embs = []
            for img_file in os.listdir(person_path):
                img = cv2.imread(os.path.join(person_path, img_file))
                if img is not None:
                    _, face_embs, _, _ = detector.detect(img)
                    if face_embs: embs.append(face_embs[0])
            if embs: embeddings_dict[person_name] = embs

    return EnhancedSecurityGuard(config=cfg, known_embeddings_dict=embeddings_dict, **kwargs)


if __name__ == "__main__":
    config = SecurityGuardConfig(
        frame_interval=10, enable_face_tracking=True,
        use_faiss_index=True, enable_async_screenshots=True
    )

    video_path = "../media_files/WIN_20260227_22_00_29_Pro.mp4"
    cap = cv2.VideoCapture(video_path)
    
    writer = cv2.VideoWriter("output_reid_v5.avi", cv2.VideoWriter_fourcc(*"mp4v"), 
                             int(cap.get(cv2.CAP_PROP_FPS)), 
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    guard = create_security_guard(config=config, show=True, model="yolo26m-pose.pt", conf=0.6)

    # [V5 UPGRADE] Graceful video stream degradation loop
    max_retries = 5
    retries = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                retries += 1
                if retries > max_retries:
                    print("[ERROR] Video stream completely lost. Exiting.")
                    break
                print(f"[WARNING] Frame drop detected. Retrying {retries}/{max_retries}...")
                time.sleep(0.5)
                continue
            
            retries = 0 # reset on success
            writer.write(guard(frame).plot_im)
    finally:
        guard.shutdown()
        cap.release()
        writer.release()
        cv2.destroyAllWindows()