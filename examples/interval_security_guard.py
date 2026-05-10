"""
Interval-Based Facial Recognition Security Guard System

This module provides an interval-based facial recognition system that:
- Uses configurable frame/time intervals to control facial recognition frequency
- Detects and tracks new persons entering the frame
- Caches recognized person identities to reduce redundant processing
- Integrates YOLO for person detection and face_recognition for identification

Configuration Options:
- frame_interval: Process every N frames (default: 10)
- face_recognition_interval: Time-based interval in seconds
- enable_new_person_detection: Always run FR for new persons
- person_cache_ttl: Duration to cache person identity (seconds)

Example Usage:
    from interval_security_guard import create_security_guard, IntervalFaceRecognitionConfig

    config = IntervalFaceRecognitionConfig(
        frame_interval=10,
        enable_new_person_detection=True,
        person_cache_ttl=60.0
    )
    guard = create_security_guard(config=config)
"""

import cv2
import numpy as np
import face_recognition
import pygame
import os
import time
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from ultralytics import solutions
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors
from ultralytics.utils import LOGGER
import asyncio
import threading
from threading import Lock


# ========== 🔊 SOUND SETUP ==========
BASE_DIR = Path(__file__).parent.parent
ALARM_FILE = BASE_DIR / "security_alart.mp3"
_alarm_sound = None
_alarm_loaded = False

# Try to load the alarm sound
if ALARM_FILE.exists():
    try:
        pygame.mixer.music.load(str(ALARM_FILE))
        _alarm_loaded = True
        LOGGER.info(f"Alarm sound loaded from {ALARM_FILE}")
    except Exception as e:
        LOGGER.warning(f"Failed to load alarm file '{ALARM_FILE}': {e}")
else:
    LOGGER.warning(f"Alarm file '{ALARM_FILE}' not found.")

# Also try pygame Sound for short alerts
try:
    # Try to load as Sound if music fails
    if not _alarm_loaded and ALARM_FILE.exists():
        _alarm_sound = pygame.mixer.Sound(str(ALARM_FILE))
        _alarm_loaded = True
except Exception as e:
    LOGGER.warning(f"Could not load as Sound: {e}")


# ========== 📁 KNOWN FACE ENCODING LOADER ==========
KNOWN_FACE_DIR = BASE_DIR / "family_members"
known_face_encodings = []
known_face_names = []

if os.path.exists(KNOWN_FACE_DIR):
    for name in os.listdir(KNOWN_FACE_DIR):
        person_dir = os.path.join(KNOWN_FACE_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            path = os.path.join(person_dir, filename)
            try:
                img = face_recognition.load_image_file(path)
                enc = face_recognition.face_encodings(img)
                if enc:
                    known_face_encodings.append(enc[0])
                    known_face_names.append(name)
                    print(f"[INFO] Loaded face for {name} from {filename}")
            except Exception as e:
                print(f"[ERROR] Failed loading {path}: {e}")
else:
    print("[WARNING] No known_faces directory found.")


# ========== CONFIGURATION CLASS ==========
@dataclass
class IntervalFaceRecognitionConfig:
    """
    Configuration for interval-based facial recognition.

    Attributes:
        frame_interval: How often to perform face recognition (every N frames)
                        Lower = more responsive but higher CPU usage
                        Higher = better performance but may miss quick detections
        face_recognition_interval: Time-based interval in seconds (alternative to frame_interval)
        enable_new_person_detection: Always run FR when a NEW person enters frame
        person_cache_ttl: Time in seconds to cache recognized person identity
        face_tolerance: Face recognition tolerance (lower = stricter, 0.4-0.6 recommended)
        min_face_size: Minimum face size to process (tuple: width, height)
        enable_caching: Enable face encoding caching for performance
        max_cache_size: Maximum number of face encodings to cache
        process_unknown_faces: Whether to always process unknown faces for learning
    """

    # Frame interval: Process every N frames (0 = every frame)
    frame_interval: int = 10

    # Time-based interval in seconds (takes precedence if > 0)
    face_recognition_interval: float = 0.0

    # Always recognize new persons entering the frame
    enable_new_person_detection: bool = True

    # Cache recognized persons to avoid redundant FR
    person_cache_ttl: float = 30.0  # seconds

    # Face recognition tolerance (lower = stricter matching)
    face_tolerance: float = 0.55

    # Minimum face size to process
    min_face_size: Tuple[int, int] = (40, 40)

    # Enable face encoding caching
    enable_caching: bool = True

    # Maximum cached face encodings
    max_cache_size: int = 100

    # Process unknown faces even when in interval skip
    process_unknown_faces: bool = True

    # Downscale factor for face recognition (0.25 = 1/4 resolution)
    fr_downscale_factor: float = 0.25

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.frame_interval < 0:
            raise ValueError("frame_interval must be non-negative")
        if not 0.3 <= self.face_tolerance <= 0.7:
            LOGGER.warning("face_tolerance should be between 0.3 and 0.7")
        if self.person_cache_ttl <= 0:
            raise ValueError("person_cache_ttl must be positive")
        return True


# ========== 👤 PERSON IDENTITY CACHE ==========
class PersonIdentityCache:
    """
    Cache for storing recognized person identities.
    Uses OrderedDict for LRU-like behavior.
    """

    def __init__(self, config: IntervalFaceRecognitionConfig):
        self.config = config
        self._cache: OrderedDict[int, Tuple[str, bool, float]] = OrderedDict()
        self._lock = Lock()

    def get(self, track_id: int) -> Optional[Tuple[str, bool]]:
        """Get cached identity for a track ID. Returns (name, is_known)."""
        with self._lock:
            if track_id in self._cache:
                name, is_known, timestamp = self._cache[track_id]
                # Check if cache entry is still valid
                if time.time() - timestamp < self.config.person_cache_ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(track_id)
                    return (name, is_known)
                else:
                    # Expired - remove from cache
                    del self._cache[track_id]
            return None

    def set(self, track_id: int, name: str, is_known: bool) -> None:
        """Set identity for a track ID."""
        with self._lock:
            # Enforce max cache size (LRU eviction)
            while len(self._cache) >= self.config.max_cache_size:
                self._cache.popitem(last=False)

            self._cache[track_id] = (name, is_known, time.time())

    def clear(self) -> None:
        """Clear all cached identities."""
        with self._lock:
            self._cache.clear()

    def get_known_count(self) -> int:
        """Get count of known persons currently cached."""
        with self._lock:
            return sum(1 for _, is_known, _ in self._cache.values() if is_known)


# ========== 🆕 NEW PERSON TRACKER ==========
class NewPersonTracker:
    """
    Tracks which person track IDs have been seen for new detection.
    """

    def __init__(self):
        self._seen_ids: set = set()
        self._new_ids: set = set()
        self._lock = Lock()

    def update(self, current_ids: List[int]) -> List[int]:
        """
        Update tracking and return list of NEW person IDs.

        Args:
            current_ids: List of current person track IDs in frame

        Returns:
            List of NEW track IDs that weren't seen before
        """
        current_set = set(current_ids)

        with self._lock:
            # Find new IDs not seen before
            new_ids = current_set - self._seen_ids

            # Update seen IDs
            self._seen_ids.update(current_ids)
            self._new_ids.update(new_ids)

            # Return new IDs for this frame
            return list(new_ids)

    def is_new(self, track_id: int) -> bool:
        """Check if a track ID is a new detection."""
        with self._lock:
            return track_id in self._new_ids

    def reset_new(self, track_id: int) -> None:
        """Mark a track ID as no longer new (after processing)."""
        with self._lock:
            self._new_ids.discard(track_id)


# ========== ⏱️ INTERVAL CONTROLLER ==========
class FrameIntervalController:
    """
    Controls when face recognition should be performed based on:
    1. Frame interval (every N frames)
    2. Time interval (every N seconds)
    3. New person detection
    """

    def __init__(self, config: IntervalFaceRecognitionConfig):
        self.config = config
        self._frame_count: int = 0
        self._last_fr_time: float = 0.0
        self._lock = Lock()

    def should_run_face_recognition(self, new_person_ids: List[int]) -> bool:
        """
        Determine if face recognition should run this frame.

        Args:
            new_person_ids: List of new person track IDs in frame

        Returns:
            True if face recognition should be performed
        """
        with self._lock:
            self._frame_count += 1
            current_time = time.time()

            # Always run for new persons if enabled
            if self.config.enable_new_person_detection and new_person_ids:
                return True

            # Check time-based interval
            if self.config.face_recognition_interval > 0:
                if current_time - self._last_fr_time >= self.config.face_recognition_interval:
                    self._last_fr_time = current_time
                    return True
                return False

            # Check frame-based interval
            if self.config.frame_interval > 0:
                return self._frame_count % self.config.frame_interval == 0

            # Default: run every frame
            return True

    def reset(self) -> None:
        """Reset frame counter."""
        with self._lock:
            self._frame_count = 0
            self._last_fr_time = time.time()


# ========== 🤖 MAIN SECURITY GUARD CLASS ==========
class IntervalBasedAiSecurityGuard(solutions.VisionEye):
    """
    Interval-based AI Security Guard with configurable facial recognition.

    Features:
    - Configurable frame/time interval for face recognition
    - Automatic new person detection triggering
    - Person identity caching to reduce redundant FR
    - Multiple person tracking support
    - Performance optimizations

    Example configuration:
        config = IntervalFaceRecognitionConfig(
            frame_interval=10,  # Process every 10th frame
            enable_new_person_detection=True,
            person_cache_ttl=30.0
        )
    """

    def __init__(
        self,
        *args,
        config: Optional[IntervalFaceRecognitionConfig] = None,
        known_face_encodings: Optional[List] = None,
        known_face_names: Optional[List] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Initialize configuration
        self.config = config or IntervalFaceRecognitionConfig()
        self.config.validate()

        # Known faces
        self.known_face_encodings = known_face_encodings or []
        self.known_face_names = known_face_names or []

        # Sound control
        self.sound_played = False

        # Initialize subsystems
        self.person_cache = PersonIdentityCache(self.config)
        self.new_person_tracker = NewPersonTracker()
        self.interval_controller = FrameIntervalController(self.config)

        # Performance tracking
        self._total_frames = 0
        self._fr_runs = 0
        self._cache_hits = 0
        self._new_person_triggers = 0

        LOGGER.info(f"Initialized IntervalBasedAiSecurityGuard with config: {self.config}")

    def play_sound(self):
        """Plays the alarm sound if not already playing."""
        if not self.sound_played:
            # Check if pygame mixer is initialized
            if not pygame.mixer.get_init():
                LOGGER.warning("pygame mixer not initialized, attempting to reinitialize...")
                try:
                    pygame.mixer.init()
                except Exception as e:
                    LOGGER.error(f"Failed to initialize pygame mixer: {e}")
                    return

            try:
                # Check if music is loaded
                if _alarm_loaded:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
                        self.sound_played = True
                        LOGGER.info("🚨 Alarm Triggered: Unknown person detected")
                else:
                    # No alarm loaded - just log
                    LOGGER.info("🚨 Unknown person detected (no alarm sound loaded)")
                    self.sound_played = True
            except Exception as e:
                LOGGER.error(f"Error playing sound: {e}")
                # Don't repeatedly try to play if it fails
                self.sound_played = True

    def reset_sound(self):
        """Stops the alarm sound and resets state."""
        if self.sound_played:
            try:
                if pygame.mixer.get_init() and _alarm_loaded:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                self.sound_played = False
            except Exception as e:
                LOGGER.error(f"Error resetting sound: {e}")
                self.sound_played = False

    def _detect_faces_in_frame(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        Detect faces in frame with performance optimizations.

        Args:
            frame: Input frame in BGR format

        Returns:
            Tuple of (face_locations, face_encodings) in ORIGINAL frame coordinates
        """
        h, w, _ = frame.shape

        # Downscale for faster processing
        downscale = self.config.fr_downscale_factor
        small_frame = cv2.resize(frame, (0, 0), fx=downscale, fy=downscale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        try:
            face_locations = face_recognition.face_locations(rgb_small_frame)
        except Exception as e:
            LOGGER.warning(f"Face location detection error: {e}")
            face_locations = []

        # Filter small faces and scale to original frame coordinates
        filtered_locations = []
        for top, right, bottom, left in face_locations:
            # Scale to original frame coordinates
            orig_top = int(top / downscale)
            orig_right = int(right / downscale)
            orig_bottom = int(bottom / downscale)
            orig_left = int(left / downscale)

            face_w = orig_right - orig_left
            face_h = orig_bottom - orig_top

            if face_w >= self.config.min_face_size[0] and face_h >= self.config.min_face_size[1]:
                filtered_locations.append((orig_top, orig_right, orig_bottom, orig_left))

        # Get encodings for valid faces (using small frame coordinates)
        face_encodings = []
        if filtered_locations:
            try:
                # Convert back to small frame coords for encoding
                small_locations = []
                for top, right, bottom, left in filtered_locations:
                    small_locations.append(
                        (int(top * downscale), int(right * downscale), int(bottom * downscale), int(left * downscale))
                    )
                face_encodings = face_recognition.face_encodings(rgb_small_frame, small_locations)
            except Exception as e:
                LOGGER.warning(f"Face encoding error: {e}")

        return filtered_locations, face_encodings

    def _identify_face(self, face_encoding) -> Tuple[str, bool]:
        """
        Identify a face against known faces database.

        Args:
            face_encoding: Face encoding to identify

        Returns:
            Tuple of (name, is_known)
        """
        if not self.known_face_encodings:
            return ("Unknown", False)

        try:
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < self.config.face_tolerance:
                name = self.known_face_names[best_match_index]
                return (name, True)
        except Exception as e:
            LOGGER.warning(f"Face matching error: {e}")

        return ("Unknown", False)

    def _associate_face_with_person(
        self, person_box: List[int], face_locations: List, scale_factor: float
    ) -> Optional[Tuple]:
        """
        Associate a face with a person's bounding box.

        Args:
            person_box: [x1, y1, x2, y2] person's bounding box
            face_locations: List of face locations
            scale_factor: Scale factor for face locations

        Returns:
            Matched face encoding or None
        """
        person_box_left, person_box_top, person_box_right, person_box_bottom = person_box

        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Scale face location to original frame
            scaled_top = int(top * scale_factor)
            scaled_right = int(right * scale_factor)
            scaled_bottom = int(bottom * scale_factor)
            scaled_left = int(left * scale_factor)

            # Calculate face center
            face_center_x = (scaled_left + scaled_right) // 2
            face_center_y = (scaled_top + scaled_bottom) // 2

            # Check if face center is inside person box
            if (
                person_box_left <= face_center_x <= person_box_right
                and person_box_top <= face_center_y <= person_box_bottom
            ):
                return (scaled_top, scaled_right, scaled_bottom, scaled_left)

        return None

    def __call__(self, im0: np.ndarray) -> SolutionResults:
        """
        Process a single frame with interval-based facial recognition.

        Args:
            im0: Input frame

        Returns:
            SolutionResults with annotated frame
        """
        self._total_frames += 1

        # Extract tracks from YOLO
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)

        # Get current person track IDs
        current_person_ids = []
        person_boxes = {}

        for cls, t_id, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            if int(cls) == 0:  # Person class
                track_id = int(t_id)
                current_person_ids.append(track_id)
                person_boxes[track_id] = box.tolist()

        # Check for new persons
        new_person_ids = self.new_person_tracker.update(current_person_ids)

        # Determine if we should run face recognition
        should_run_fr = self.interval_controller.should_run_face_recognition(new_person_ids)

        # Initialize face data
        face_locations = []
        face_encodings = []

        if should_run_fr:
            self._fr_runs += 1
            if new_person_ids:
                self._new_person_triggers += len(new_person_ids)

            # Detect faces in the entire frame (optimization)
            face_locations, face_encodings = self._detect_faces_in_frame(im0)

        # Process each detected person
        unknown_person_count = 0

        for cls, t_id, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            if int(cls) != 0:  # Skip non-person classes
                annotator.box_label(box, label=self.adjust_box_label(cls, conf, t_id), color=colors(int(t_id), True))
                annotator.visioneye(box, self.vision_point)
                continue

            track_id = int(t_id)
            person_box = box.tolist()

            # Try to get cached identity first
            cached_identity = self.person_cache.get(track_id)

            if cached_identity is not None:
                # Use cached identity
                name, is_known = cached_identity
                self._cache_hits += 1
            elif should_run_fr and face_encodings:
                # Run face recognition for this person
                # Face locations are now in original frame coordinates from _detect_faces_in_frame
                matched_face = self._associate_face_with_person(person_box, face_locations, scale_factor=1.0)

                if matched_face is not None:
                    # Find the corresponding encoding
                    try:
                        face_idx = face_locations.index(matched_face)
                        if face_idx < len(face_encodings):
                            name, is_known = self._identify_face(face_encodings[face_idx])
                        else:
                            name, is_known = "Unknown", False
                    except ValueError:
                        # Face location not found in list
                        name, is_known = "Unknown", False
                else:
                    name, is_known = "Unknown", False

                # Cache the result (always cache)
                self.person_cache.set(track_id, name, is_known)
            else:
                # No FR this frame - try cache one more time (might have expired)
                cached_identity = self.person_cache.get(track_id)
                if cached_identity is not None:
                    name, is_known = cached_identity
                    self._cache_hits += 1
                else:
                    # Still no cache - mark as unknown but also cache it to avoid repeated FR
                    name, is_known = "Unknown", False
                    # Don't cache unknown here to allow retry on next interval

            # Mark new person as processed
            if track_id in new_person_ids:
                self.new_person_tracker.reset_new(track_id)

            # Update counters
            if not is_known:
                unknown_person_count += 1
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known

            # Create label
            label = f"{name}" if is_known else "Unknown"

            # Get base label and create final label
            base_label = self.adjust_box_label(int(cls), float(conf) if conf is not None else 0.0, t_id)
            prefix = str(self.CFG.get("person_label_prefix", label))
            custom_label = f"{prefix}:"
            final_label = f"{custom_label} {base_label}" if base_label else custom_label

            # Draw annotations
            annotator.box_label(box, label=final_label, color=colors(int(t_id), True))
            annotator.visioneye(box, self.vision_point)

        # Trigger alarm based on unknown person count
        records = self.CFG.get("records", 1)
        if unknown_person_count >= records:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(asyncio.to_thread(self.play_sound))
            except RuntimeError:
                threading.Thread(target=self.play_sound, daemon=True).start()
            except Exception as e:
                LOGGER.exception("Failed to schedule alarm: %s", e)
                self.play_sound()
        else:
            self.reset_sound()

        # Generate output
        plot_im = annotator.result()
        self.display_output(plot_im)

        # Add performance stats overlay (optional)
        if self._total_frames % 30 == 0:  # Update every 30 frames
            self._log_performance_stats()

        # Display track count on frame
        total_tracks = len(getattr(self, "track_ids", []))
        cv2.putText(plot_im, f"Tracks: {total_tracks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display interval info on frame
        interval_info = f"FR: {'ON' if should_run_fr else 'OFF'} | Frame: {self._total_frames}"
        cv2.putText(plot_im, interval_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))

    def _log_performance_stats(self) -> None:
        """Log performance statistics."""
        fr_rate = (self._fr_runs / self._total_frames * 100) if self._total_frames > 0 else 0
        cache_hit_rate = (self._cache_hits / self._total_frames * 100) if self._total_frames > 0 else 0

        LOGGER.info(
            f"Performance | Frames: {self._total_frames} | "
            f"FR Runs: {self._fr_runs} ({fr_rate:.1f}%) | "
            f"Cache Hits: {self._cache_hits} ({cache_hit_rate:.1f}%) | "
            f"New Person Triggers: {self._new_person_triggers}"
        )


# ========== 🏭 FACTORY FUNCTION ==========
def create_security_guard(
    config: Optional[IntervalFaceRecognitionConfig] = None, known_face_dir: str = None, **kwargs
) -> IntervalBasedAiSecurityGuard:
    """
    Factory function to create an IntervalBasedAiSecurityGuard instance.

    Args:
        config: IntervalFaceRecognitionConfig instance (or None for defaults)
        known_face_dir: Directory containing known face images (defaults to project family_members/)
        **kwargs: Additional arguments for VisionEye

    Returns:
        Configured IntervalBasedAiSecurityGuard instance

    Example:
        # Default configuration
        guard = create_security_guard()

        # Custom interval configuration
        config = IntervalFaceRecognitionConfig(
            frame_interval=15,  # Every 15 frames
            enable_new_person_detection=True,
            person_cache_ttl=60.0
        )
        guard = create_security_guard(config=config)
    """
    # Use default path if not provided
    if known_face_dir is None:
        known_face_dir = str(Path(__file__).parent.parent / "family_members")

    # Load known faces
    _known_face_encodings = []
    _known_face_names = []

    if os.path.exists(known_face_dir):
        for name in os.listdir(known_face_dir):
            person_dir = os.path.join(known_face_dir, name)
            if not os.path.isdir(person_dir):
                continue
            for filename in os.listdir(person_dir):
                path = os.path.join(person_dir, filename)
                try:
                    img = face_recognition.load_image_file(path)
                    enc = face_recognition.face_encodings(img)
                    if enc:
                        _known_face_encodings.append(enc[0])
                        _known_face_names.append(name)
                        LOGGER.info(f"[INFO] Loaded face for {name} from {filename}")
                except Exception as e:
                    LOGGER.warning(f"[ERROR] Failed loading {path}: {e}")
    else:
        LOGGER.warning(f"[WARNING] Known face directory '{known_face_dir}' not found.")

    return IntervalBasedAiSecurityGuard(
        config=config, known_face_encodings=_known_face_encodings, known_face_names=_known_face_names, **kwargs
    )


# ========== 🎬 MAIN EXECUTION ==========
if __name__ == "__main__":
    # Example configurations:

    # Configuration 1: High responsiveness (every 5 frames + new person detection)
    config_responsive = IntervalFaceRecognitionConfig(
        frame_interval=5, enable_new_person_detection=True, person_cache_ttl=30.0, face_tolerance=0.55
    )

    # Configuration 2: Balanced (every 10 frames) - RECOMMENDED
    config_balanced = IntervalFaceRecognitionConfig(
        frame_interval=10, enable_new_person_detection=True, person_cache_ttl=60.0, face_tolerance=0.55
    )

    # Configuration 3: Maximum performance (every 30 frames)
    config_performance = IntervalFaceRecognitionConfig(
        frame_interval=30,
        enable_new_person_detection=True,
        person_cache_ttl=120.0,
        face_tolerance=0.6,  # Slightly more tolerant
    )

    # Configuration 4: Time-based (every 0.5 seconds)
    config_time_based = IntervalFaceRecognitionConfig(
        face_recognition_interval=0.5,  # Takes precedence over frame_interval
        enable_new_person_detection=True,
        person_cache_ttl=30.0,
    )

    # Choose your configuration here!
    SELECTED_CONFIG = config_balanced  # Change to use different config

    # Video source
    # cap = cv2.VideoCapture("../media_files/ruhama.mp4")
    # cap = cv2.VideoCapture("../media_files/istockphoto-2002563994-640_adpp_is.mp4")
    # cap = cv2.VideoCapture("../media_files/WIN_20260227_22_00_29_Pro.mp4")
    cap = cv2.VideoCapture("../media_files/istockphoto-2240284006-640_adpp_is.mp4")
    # cap = cv2.VideoCapture("../media_files/istockphoto-2240272228-640_adpp_is.mp4")
    assert cap.isOpened(), "Error reading video file"

    # Video writer
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Define points
    center_point = (w // 2, h // 2)
    opposite_point = (w // 2 - 250, h - 10)

    # Initialize the security guard with selected config
    AiSecurityGuard = IntervalBasedAiSecurityGuard(
        show=True,  # display the output
        model="yolo26m.pt",  # use any model that Ultralytics supports
        classes=[0, 2],  # generate visioneye view for specific classes
        vision_point=opposite_point,  # the point where VisionEye will view objects and draw tracks
        conf=0.3,
        iou=0.5,
        # verbose=True,
        known_face_encodings=known_face_encodings,
        known_face_names=known_face_names,
        records=1,
        config=SELECTED_CONFIG,  # Use the interval-based configuration
    )

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()

        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        results = AiSecurityGuard(im0)

        print(results)  # access the output

        video_writer.write(results.plot_im)  # write the video file

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()  # destroy all opened windows

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total Frames Processed: {AiSecurityGuard._total_frames}")
    print(f"Face Recognition Runs: {AiSecurityGuard._fr_runs}")
    print(f"Cache Hits: {AiSecurityGuard._cache_hits}")
    print(f"New Person Triggers: {AiSecurityGuard._new_person_triggers}")
    if AiSecurityGuard._total_frames > 0:
        fr_rate = AiSecurityGuard._fr_runs / AiSecurityGuard._total_frames * 100
        print(f"FR Execution Rate: {fr_rate:.1f}%")
    print("=" * 60)
