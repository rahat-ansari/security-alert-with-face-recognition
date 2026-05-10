# ============================================================================
# VERSION.PY - Complete Verified Implementation
# ============================================================================
#
# Face Detection and Recognition Security System - Enhanced Version
# Addresses three main issues:
# 1. False positive detections reduced through geometry validation and NMS
# 2. Duplicate face cropping eliminated through embedding-based deduplication
# 3. Unknown subject consistency achieved through persistent unknown face registry
#
# ============================================================================

"""
Enhanced Face Detection and Recognition Security System
======================================================

This module provides a comprehensive solution for face detection pipelines with:
1. FALSE POSITIVE REDUCTION - Geometry validation and NMS deduplication
2. DUPLICATE ELIMINATION - Embedding-based semantic deduplication
3. UNKNOWN CONSISTENCY - Persistent unknown face registry for cross-session matching

Version: 3.0.0 (Verified Implementation)
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
import pickle
import hashlib
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
# ENHANCED CONFIGURATION
# ============================================================================


@dataclass
class EnhancedFaceDetectionConfig:
    """
    Configuration for enhanced face detection pipeline.

    Addresses all three issues through configurable parameters:

    ISSUE 1 - FALSE POSITIVE REDUCTION:
    - min_face_aspect_ratio: Validates face width/height ratio (typically 0.6-1.0)
    - min_detection_confidence: Filters low-confidence detections
    - enable_landmark_validation: Uses facial landmarks to validate structure
    - nms_iou_threshold: Removes overlapping detections

    ISSUE 2 - DUPLICATE ELIMINATION:
    - enable_embedding_dedup: Uses cosine similarity to detect duplicate faces
    - embedding_similarity_threshold: Threshold for considering faces as duplicates

    ISSUE 3 - UNKNOWN CONSISTENCY:
    - enable_unknown_registry: Enables persistent storage of unknown faces
    - unknown_similarity_threshold: Threshold for matching unknown faces across sessions
    - unknown_face_ttl_days: How long to keep unknown faces in registry
    """

    # Basic settings (existing)
    min_face_size: Tuple[int, int] = (40, 40)
    face_tolerance: float = 0.5
    min_confidence: float = 0.6

    # ISSUE 1: False positive reduction settings
    # =========================================================================
    min_face_aspect_ratio: float = 0.6  # Minimum width/height ratio for valid face
    max_face_aspect_ratio: float = 1.0  # Maximum width/height ratio for valid face
    min_detection_confidence: float = 0.5  # Minimum detection confidence score
    enable_landmark_validation: bool = True  # Validate face landmark geometry
    landmark_distance_threshold: float = 0.3  # Normalized landmark distance threshold
    enable_nms: bool = True  # Apply Non-Maximum Suppression
    nms_iou_threshold: float = 0.3  # IoU threshold for NMS

    # ISSUE 2: Duplicate detection settings
    # =========================================================================
    enable_embedding_dedup: bool = True  # Enable embedding-based deduplication
    embedding_similarity_threshold: float = 0.9  # >90% similarity = duplicate

    # ISSUE 3: Unknown face registry settings
    # =========================================================================
    enable_unknown_registry: bool = True  # Enable persistent unknown face storage
    unknown_registry_path: str = "unknown_faces_registry.pkl"
    max_unknown_faces_stored: int = 1000  # Maximum unknown faces to store
    unknown_similarity_threshold: float = 0.75  # Threshold for matching unknown faces
    unknown_face_ttl_days: int = 30  # Unknown faces expire after 30 days

    # Processing settings
    insightface_model: str = "buffalo_l"
    insightface_det_size: Tuple[int, int] = (640, 640)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Used in NMS deduplication to remove overlapping detections.

    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
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
    """
    Compute cosine similarity between two embedding vectors.

    Used for:
    - Face recognition (known faces)
    - Duplicate detection within frame
    - Unknown face matching across sessions

    Args:
        emb1, emb2: Embedding vectors (typically 512-dimensional)

    Returns:
        Similarity score between -1 and 1 (0 to 1 for normalized face embeddings)
    """
    # Ensure normalized vectors
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
    return float(np.dot(emb1, emb2))


def compute_embedding_hash(embedding: np.ndarray) -> str:
    """
    Create a hash of an embedding for efficient storage/lookup.

    Uses binarization of the embedding for hashing - bits above 0
    are set to 1, below 0 are set to 0.

    Args:
        embedding: Face embedding vector

    Returns:
        16-character hex string hash
    """
    binary = (embedding > 0).astype(np.uint8).tobytes()
    return hashlib.sha256(binary).hexdigest()[:16]


def _setup_logging(name: str = "EnhancedFaceSystem") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _get_base_dir() -> Path:
    """Get base directory for file operations."""
    try:
        return Path(__file__).parent.parent
    except NameError:
        return Path.cwd()


# ============================================================================
# ISSUE 1: FALSE POSITIVE REDUCTION - FACE GEOMETRY VALIDATOR
# ============================================================================


class FaceGeometryValidator:
    """
    Validates face detections using geometric constraints.

    Addresses FALSE POSITIVE DETECTIONS by:
    1. Checking aspect ratio of face bounding boxes
    2. Validating facial landmark positions
    3. Ensuring landmarks form a valid facial structure

    Rationale: Non-face regions often have irregular shapes or
    landmark positions that don't correspond to valid facial geometry.
    Objects misidentified as faces (hands, shadows, patterns) typically
    fail one or more of these validation checks.
    """

    def __init__(self, config: EnhancedFaceDetectionConfig):
        self.config = config
        self.logger = _setup_logging("FaceGeometryValidator")

    def validate_aspect_ratio(self, box: List[float]) -> bool:
        """
        Validate face bounding box aspect ratio.

        Human faces typically have width/height ratio between 0.6 and 1.0.
        Objects misidentified as faces often fall outside this range:
        - Very wide: might be a body part or object
        - Very tall: might be a false detection on patterns

        Args:
            box: Face bounding box [x1, y1, x2, y2]

        Returns:
            True if aspect ratio is within valid range
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        if height <= 0:
            return False

        aspect_ratio = width / height

        is_valid = self.config.min_face_aspect_ratio <= aspect_ratio <= self.config.max_face_aspect_ratio

        if not is_valid:
            self.logger.debug(f"Rejected: aspect ratio {aspect_ratio:.2f} outside valid range")

        return is_valid

    def validate_landmarks(self, kps: np.ndarray) -> bool:
        """
        Validate facial landmark positions.

        InsightFace provides 5 keypoints:
        - kps[0]: Left eye
        - kps[1]: Right eye
        - kps[2]: Nose
        - kps[3]: Left mouth corner
        - kps[4]: Right mouth corner

        These should form a consistent facial structure:
        - Eyes should be roughly at same y-level (within threshold)
        - Nose should be between eyes and below eye level
        - Mouth should be below nose
        - Mouth width should be proportional to inter-ocular distance

        Args:
            kps: 5-point facial keypoints array

        Returns:
            True if landmarks form valid facial structure
        """
        if kps is None or len(kps) < 5:
            return False

        try:
            # Extract landmark coordinates
            left_eye = kps[0]
            right_eye = kps[1]
            nose = kps[2]
            left_mouth = kps[3]
            right_mouth = kps[4]

            # Calculate inter-ocular distance (eye distance = ~0.3 * face width)
            eye_distance = np.linalg.norm(right_eye - left_eye)

            # Eye level check - eyes should be at similar y-coordinates
            # Large difference suggests invalid detection
            eye_level_diff = abs(left_eye[1] - right_eye[1]) / eye_distance
            if eye_level_diff > self.config.landmark_distance_threshold * 2:
                self.logger.debug(f"Rejected: eyes not at same level (diff={eye_level_diff:.2f})")
                return False

            # Nose should be roughly centered between eyes
            eye_center = (left_eye + right_eye) / 2
            nose_offset = np.linalg.norm(nose[:2] - eye_center[:2]) / eye_distance

            # Nose should be below eyes and roughly centered
            if nose[1] < min(left_eye[1], right_eye[1]):
                self.logger.debug(f"Rejected: nose above eyes")
                return False

            if nose_offset > 1.5:  # Nose too far from eye center
                self.logger.debug(f"Rejected: nose offset too large ({nose_offset:.2f})")
                return False

            # Mouth should be below nose
            mouth_center = (left_mouth + right_mouth) / 2
            if mouth_center[1] < nose[1]:
                self.logger.debug(f"Rejected: mouth above nose")
                return False

            # Mouth width should be proportional to eye distance
            # Typical ratio: 1.0 - 2.0
            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            if mouth_width / eye_distance < 0.8 or mouth_width / eye_distance > 2.5:
                self.logger.debug(f"Rejected: mouth width abnormal ({mouth_width / eye_distance:.2f})")
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Landmark validation error: {e}")
            return False

    def validate_detection(
        self, box: List[float], kps: Optional[np.ndarray] = None, confidence: Optional[float] = None
    ) -> bool:
        """
        Combined validation of a face detection.

        Validates through multiple stages:
        1. Minimum size check (existing logic)
        2. Aspect ratio validation
        3. Landmark geometry validation (if enabled)
        4. Confidence threshold (if provided)

        Args:
            box: Face bounding box [x1, y1, x2, y2]
            kps: Optional facial keypoints
            confidence: Optional detection confidence score

        Returns:
            True if detection passes all validation checks
        """
        # Stage 1: Minimum size check (existing logic)
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        if width < self.config.min_face_size[0] or height < self.config.min_face_size[1]:
            return False

        # Stage 2: Aspect ratio validation
        if not self.validate_aspect_ratio(box):
            return False

        # Stage 3: Confidence threshold (if provided)
        if confidence is not None and confidence < self.config.min_detection_confidence:
            self.logger.debug(f"Rejected: confidence {confidence:.2f} below threshold")
            return False

        # Stage 4: Landmark geometry validation (if enabled and available)
        if self.config.enable_landmark_validation and kps is not None:
            if not self.validate_landmarks(kps):
                return False

        return True


# ============================================================================
# ISSUE 1 & 2: NMS DEDUPLICATION
# ============================================================================


class FaceNMSDeduplicator:
    """
    Applies Non-Maximum Suppression to remove overlapping face detections.

    Addresses DUPLICATE CROPPING by:
    - Removing detections that overlap significantly (IoU > threshold)
    - Keeping the detection with highest confidence/size

    Rationale: When multiple overlapping detections of the same face occur,
    we want to keep only the best one to avoid processing duplicates.
    This is particularly important for faces partially occluded or
    detected at multiple scales.
    """

    def __init__(self, config: EnhancedFaceDetectionConfig):
        self.config = config
        self.logger = _setup_logging("FaceNMS")

    def apply_nms(self, boxes: List[List[float]], confidences: Optional[List[float]] = None) -> List[int]:
        """
        Apply NMS to remove overlapping detections.

        Algorithm:
        1. Sort detections by confidence (descending)
        2. Keep highest confidence detection
        3. Remove all detections with IoU > threshold with kept detection
        4. Repeat until no detections remain

        Args:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            confidences: Optional confidence scores (uses box area if not provided)

        Returns:
            Indices of boxes to keep
        """
        if not boxes:
            return []

        n = len(boxes)

        # Use box areas as fallback confidences (larger faces = higher priority)
        if confidences is None:
            confidences = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]

        # Sort by confidence (descending)
        indices = sorted(range(n), key=lambda i: confidences[i], reverse=True)

        keep = []

        while indices:
            # Keep the highest confidence detection
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # Remove overlapping detections
            current_box = boxes[current]
            new_indices = []

            for idx in indices[1:]:
                iou = compute_iou(current_box, boxes[idx])
                if iou <= self.config.nms_iou_threshold:
                    new_indices.append(idx)

            indices = new_indices

        self.logger.debug(f"NMS: {n} detections -> {len(keep)} unique faces")
        return keep


# ============================================================================
# ISSUE 2: EMBEDDING-BASED FACE DEDUPLICATION
# ============================================================================


class EmbeddingFaceDeduplicator:
    """
    Deduplicates faces using embedding similarity.

    Addresses DUPLICATE CROPPING by:
    - Computing embedding similarity between detected faces
    - Grouping faces with similarity > threshold as the same person
    - Keeping the best quality face from each group

    Rationale: Even with NMS, different detections of the same face
    (slight overlaps, different poses) can still pass through.
    Embedding similarity provides semantic deduplication - if two faces
    have the same identity (high embedding similarity), we only process one.
    """

    def __init__(self, config: EnhancedFaceDetectionConfig):
        self.config = config
        self.logger = _setup_logging("EmbeddingDeduplicator")

    def find_duplicates(
        self, embeddings: List[np.ndarray], quality_scores: Optional[List[float]] = None
    ) -> List[List[int]]:
        """
        Find groups of duplicate faces based on embedding similarity.

        Uses pairwise cosine similarity to identify faces belonging to
        the same person. All faces with similarity >= threshold are
        grouped together.

        Args:
            embeddings: List of face embedding vectors
            quality_scores: Optional quality scores for each face

        Returns:
            List of duplicate groups, each group is a list of indices
        """
        if not embeddings or len(embeddings) <= 1:
            return []

        n = len(embeddings)
        quality_scores = quality_scores or [1.0] * n

        # Track which faces have been assigned to a group
        assigned = [False] * n
        duplicate_groups = []

        for i in range(n):
            if assigned[i]:
                continue

            # Start a new group with face i
            current_group = [i]
            assigned[i] = True

            # Find all similar faces (same identity)
            for j in range(i + 1, n):
                if assigned[j]:
                    continue

                similarity = compute_cosine_similarity(embeddings[i], embeddings[j])

                if similarity >= self.config.embedding_similarity_threshold:
                    current_group.append(j)
                    assigned[j] = True

            # Only keep groups with multiple faces (actual duplicates)
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                self.logger.debug(
                    f"Found duplicate group: {len(current_group)} faces, "
                    f"avg similarity: {self._get_group_similarity(embeddings, current_group):.2f}"
                )

        return duplicate_groups

    def _get_group_similarity(self, embeddings: List[np.ndarray], group: List[int]) -> float:
        """Calculate average pairwise similarity within a group."""
        if len(group) < 2:
            return 1.0

        similarities = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                sim = compute_cosine_similarity(embeddings[group[i]], embeddings[group[j]])
                similarities.append(sim)

        return np.mean(similarities) if similarities else 1.0

    def get_best_face_index(self, group: List[int], quality_scores: List[float]) -> int:
        """
        Get the best face from a duplicate group based on quality.

        Args:
            group: List of indices belonging to the same person
            quality_scores: Quality scores for each face

        Returns:
            Index of the best face in the original list
        """
        if not group:
            return -1

        # Return the face with highest quality score
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
    """
    Represents an unknown face stored in the persistent registry.

    Stores embedding and metadata for cross-session matching of
    unknown faces. Supports temporal smoothing of embeddings for
    improved matching accuracy over time.
    """

    embedding: np.ndarray
    embedding_hash: str
    first_seen: float  # Unix timestamp
    last_seen: float  # Unix timestamp
    session_count: int = 1
    images_captured: int = 0
    assigned_id: str = ""

    def __post_init__(self):
        if not self.assigned_id:
            # Generate unique ID based on hash and timestamp
            self.assigned_id = f"unknown_{self.embedding_hash[:8]}_{int(self.first_seen)}"


class UnknownFaceRegistry:
    """
    Persistent registry for unknown face embeddings.

    Addresses INCONSISTENT UNKNOWN SUBJECT HANDLING by:
    - Storing embeddings of unknown faces persistently
    - Matching new unknown faces against stored ones
    - Assigning consistent IDs to the same unknown person across sessions
    - Providing temporal consistency within a session

    Rationale: When the same unknown person re-enters the system
    (different session, different track ID), we want to recognize
    them as the same person instead of creating a new identity each time.

    Features:
    - Persistent storage (survives application restarts)
    - TTL-based expiration (configurable retention period)
    - Weighted embedding updates for improved matching
    - Thread-safe operations
    """

    def __init__(self, config: EnhancedFaceDetectionConfig):
        self.config = config
        self.logger = _setup_logging("UnknownFaceRegistry")

        # In-memory storage: unknown_id -> UnknownFaceEntry
        self._entries: Dict[str, UnknownFaceEntry] = {}

        # Session-specific tracking: track_id -> unknown_id
        # Cleared at start of each processing session
        self._session_unknown_ids: Dict[int, str] = {}

        # Lock for thread safety
        self._lock = Lock()

        # Load persisted registry on initialization
        if config.enable_unknown_registry:
            self._load_registry()

    def _get_registry_path(self) -> Path:
        """Get the path for the registry persistence file."""
        base_dir = _get_base_dir()
        return base_dir / self.config.unknown_registry_path

    def _load_registry(self) -> None:
        """Load registry from persistent storage."""
        registry_path = self._get_registry_path()

        if registry_path.exists():
            try:
                with open(registry_path, "rb") as f:
                    data = pickle.load(f)

                # Validate and load entries
                entries_data = data.get("entries", {})
                current_time = time.time()
                ttl_seconds = self.config.unknown_face_ttl_days * 24 * 3600

                for entry_id, entry_dict in entries_data.items():
                    # Check if entry has expired
                    if current_time - entry_dict["last_seen"] > ttl_seconds:
                        continue

                    self._entries[entry_id] = UnknownFaceEntry(
                        embedding=np.array(entry_dict["embedding"]),
                        embedding_hash=entry_dict["embedding_hash"],
                        first_seen=entry_dict["first_seen"],
                        last_seen=entry_dict["last_seen"],
                        session_count=entry_dict.get("session_count", 1),
                        images_captured=entry_dict.get("images_captured", 0),
                        assigned_id=entry_dict.get("assigned_id", entry_id),
                    )

                self.logger.info(f"Loaded {len(self._entries)} unknown faces from registry")

            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to persistent storage."""
        if not self.config.enable_unknown_registry:
            return

        registry_path = self._get_registry_path()

        try:
            # Ensure directory exists
            registry_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "entries": {
                    entry_id: {
                        "embedding": entry.embedding.tolist(),
                        "embedding_hash": entry.embedding_hash,
                        "first_seen": entry.first_seen,
                        "last_seen": entry.last_seen,
                        "session_count": entry.session_count,
                        "images_captured": entry.images_captured,
                        "assigned_id": entry.assigned_id,
                    }
                    for entry_id, entry in self._entries.items()
                },
                "metadata": {"created": time.time(), "version": "1.0"},
            }

            with open(registry_path, "wb") as f:
                pickle.dump(data, f)

            self.logger.debug(f"Saved {len(self._entries)} unknown faces to registry")

        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")

    def find_matching_unknown(
        self, embedding: np.ndarray, min_similarity: Optional[float] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Find if this embedding matches any known unknown face.

        Uses cosine similarity to compare against all stored unknown
        embeddings. Returns the best match above threshold.

        Args:
            embedding: Face embedding to match
            min_similarity: Optional custom threshold (uses config if not provided)

        Returns:
            Tuple of (unknown_id, similarity) if match found, None otherwise
        """
        threshold = min_similarity or self.config.unknown_similarity_threshold

        # Normalize embedding for comparison
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        best_match = None
        best_similarity = threshold

        for entry_id, entry in self._entries.items():
            similarity = compute_cosine_similarity(embedding, entry.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry_id

        if best_match:
            self.logger.info(f"Matched unknown face: {best_match} (similarity: {best_similarity:.3f})")

        return (best_match, best_similarity) if best_match else None

    def register_unknown(self, embedding: np.ndarray, track_id: Optional[int] = None, quality: float = 1.0) -> str:
        """
        Register a new unknown face or update existing.

        If embedding matches an existing unknown (above threshold),
        updates that entry. Otherwise creates a new entry.

        Uses temporal smoothing: new observations blend with existing
        embeddings (30% new, 70% old) for improved stability.

        Args:
            embedding: Face embedding
            track_id: Optional track ID for session tracking
            quality: Face quality score

        Returns:
            Assigned unknown ID
        """
        with self._lock:
            current_time = time.time()
            embedding_hash = compute_embedding_hash(embedding)

            # Check if this matches an existing unknown face
            match = self.find_matching_unknown(embedding)

            if match:
                # Update existing entry - it's the same person
                unknown_id, similarity = match
                entry = self._entries[unknown_id]
                entry.last_seen = current_time
                entry.session_count += 1
                entry.images_captured += 1

                # Temporal smoothing: blend new observation with existing
                alpha = 0.3  # Weight for new observation (30%)
                entry.embedding = (1 - alpha) * entry.embedding + alpha * embedding
                entry.embedding = entry.embedding / (np.linalg.norm(entry.embedding) + 1e-8)

                unknown_id = entry.assigned_id

            else:
                # Create new unknown face entry
                # First, enforce maximum storage limit (FIFO eviction)
                if len(self._entries) >= self.config.max_unknown_faces_stored:
                    self._evict_oldest()

                normalized_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                entry = UnknownFaceEntry(
                    embedding=normalized_embedding,
                    embedding_hash=embedding_hash,
                    first_seen=current_time,
                    last_seen=current_time,
                    images_captured=1,
                )

                self._entries[entry.assigned_id] = entry
                unknown_id = entry.assigned_id

                self.logger.info(f"Registered new unknown face: {unknown_id}")

            # Track for session consistency (same person in same session)
            if track_id is not None:
                self._session_unknown_ids[track_id] = unknown_id

            # Persist changes to disk
            self._save_registry()

            return unknown_id

    def get_session_unknown_id(self, track_id: int) -> Optional[str]:
        """Get the unknown ID assigned to a track within this session."""
        return self._session_unknown_ids.get(track_id)

    def clear_session_tracking(self) -> None:
        """
        Clear session-specific tracking.

        Call at the start of each new processing session to ensure
        clean state while preserving persistent registry.
        """
        with self._lock:
            self._session_unknown_ids.clear()

    def _evict_oldest(self) -> None:
        """Remove the oldest entry from the registry (FIFO)."""
        if not self._entries:
            return

        oldest_id = min(self._entries.keys(), key=lambda k: self._entries[k].last_seen)

        del self._entries[oldest_id]
        self.logger.info(f"Evicted oldest unknown face: {oldest_id}")

    def get_statistics(self) -> Dict:
        """Get registry statistics for monitoring."""
        with self._lock:
            return {
                "total_unknowns": len(self._entries),
                "session_tracked": len(self._session_unknown_ids),
                "oldest_entry": min((e.last_seen for e in self._entries.values()), default=0),
                "newest_entry": max((e.last_seen for e in self._entries.values()), default=0),
            }


# ============================================================================
# QUALITY ASSESSMENT (Enhanced)
# ============================================================================


class EnhancedQualityAssessor:
    """
    Enhanced face quality assessment with multiple factors.

    Calculates quality score based on:
    - Face size relative to frame
    - Brightness (not too dark, not too bright)
    - Sharpness (focus quality)
    - Contrast
    """

    SIZE_WEIGHT = 0.30
    BRIGHTNESS_WEIGHT = 0.15
    SHARPNESS_WEIGHT = 0.25
    CONTRAST_WEIGHT = 0.20

    def assess(self, frame: np.ndarray, box: List[float]) -> float:
        """
        Calculate overall quality score for a face region.

        Args:
            frame: Full frame image
            box: Face bounding box [x1, y1, x2, y2]

        Returns:
            Quality score (0-100)
        """
        try:
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]

            # Clamp to frame boundaries
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                return 0.0

            # Size score - larger faces are better
            face_area = (x2 - x1) * (y2 - y1)
            frame_area = w * h
            size_score = min(100, (face_area / frame_area) * 1000)

            # Brightness score - prefer well-lit faces
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            bright_score = max(0, 100 - abs(brightness - 128) * 0.8)

            # Sharpness score - focused faces are better
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            sharp_score = min(100, lap.var() / 10)

            # Contrast score
            contrast_score = min(100, np.std(gray) / 30 * 100)

            return (
                size_score * self.SIZE_WEIGHT
                + bright_score * self.BRIGHTNESS_WEIGHT
                + sharp_score * self.SHARPNESS_WEIGHT
                + contrast_score * self.CONTRAST_WEIGHT
            )

        except Exception:
            return 0.0


# ============================================================================
# ENHANCED FACE DETECTOR (Complete Pipeline)
# ============================================================================


class EnhancedFaceDetector:
    """
    Complete enhanced face detection pipeline integrating all improvements.

    This combines:
    1. FaceGeometryValidator - reduces false positives
    2. FaceNMSDeduplicator - removes overlapping detections
    3. EmbeddingFaceDeduplicator - semantic deduplication
    4. UnknownFaceRegistry - consistent unknown handling

    The detection pipeline works as follows:

    Step 1: Initial Detection
        - Run InsightFace to get raw face detections
        - Extract boxes, embeddings, landmarks, attributes

    Step 2: Geometry Validation (False Positive Reduction)
        - Filter by aspect ratio
        - Filter by detection confidence
        - Validate landmark geometry

    Step 3: NMS Deduplication
        - Remove spatially overlapping detections
        - Keep the largest/highest confidence

    Step 4: Embedding Deduplication
        - Compare embeddings using cosine similarity
        - Group faces with >90% similarity as duplicates
        - Keep only the best quality face from each group

    Step 5: Unknown Face Matching
        - Compare against persistent unknown registry
        - Match if similarity > 75%
        - Register new unknowns if no match

    Returns:
        Validated boxes, embeddings, landmarks, attributes, unknown_ids
    """

    def __init__(
        self,
        insightface_model: str = "buffalo_l",
        detection_size: Tuple[int, int] = (640, 640),
        config: Optional[EnhancedFaceDetectionConfig] = None,
    ):
        self.config = config or EnhancedFaceDetectionConfig()
        self.logger = _setup_logging("EnhancedFaceDetector")

        # Initialize InsightFace
        try:
            self.app = FaceAnalysis(name=insightface_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=detection_size)
            self.logger.info(f"InsightFace initialized: {insightface_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            raise

        # Initialize validators and processors
        self.geometry_validator = FaceGeometryValidator(self.config)
        self.nms_deduplicator = FaceNMSDeduplicator(self.config)
        self.embedding_deduplicator = EmbeddingFaceDeduplicator(self.config)
        self.unknown_registry = UnknownFaceRegistry(self.config)
        self.quality_assessor = EnhancedQualityAssessor()

    def detect(
        self, frame: np.ndarray, quality_scores: Optional[List[float]] = None
    ) -> Tuple[
        List[List[float]],  # Validated face boxes
        List[np.ndarray],  # Face embeddings
        List,  # Landmarks
        List[Dict],  # Attributes
        List[str],  # Unknown IDs (new - for Issue 3)
    ]:
        """
        Enhanced face detection with full validation pipeline.

        Full pipeline processing:
        1. Initial InsightFace detection
        2. Geometry validation (aspect ratio, landmarks)
        3. NMS deduplication
        4. Embedding-based semantic deduplication
        5. Unknown face matching/registration

        Args:
            frame: Input image frame (BGR format)
            quality_scores: Optional pre-computed quality scores

        Returns:
            Tuple of:
            - boxes: Valid face bounding boxes
            - embeddings: Face embeddings for recognition
            - landmarks: Facial keypoints
            - attributes: Face attributes (age, gender, etc.)
            - unknown_ids: Unknown face registry IDs
        """
        # Step 1: Initial detection with InsightFace
        faces = self.app.get(frame)

        if not faces:
            return [], [], [], [], []

        # Extract raw detections
        raw_boxes = []
        raw_embeddings = []
        raw_landmarks = []
        raw_attributes = []
        raw_kps = []

        for face in faces:
            bbox = face.bbox

            # Basic size filtering (existing logic)
            if (bbox[2] - bbox[0]) < self.config.min_face_size[0] or (bbox[3] - bbox[1]) < self.config.min_face_size[1]:
                continue

            raw_boxes.append(bbox.tolist())
            raw_embeddings.append(face.embedding)
            raw_landmarks.append(face.kps)
            raw_kps.append(face.kps)

            # Get detection confidence if available
            det_score = getattr(face, "det_score", None)

            raw_attributes.append(
                {
                    "age": face.age,
                    "gender": face.gender,
                    "emotion": getattr(face, "emotion", None),
                    "pose": getattr(face, "pose", None),
                    "confidence": det_score,  # Detection confidence
                }
            )

        if not raw_boxes:
            return [], [], [], [], []

        # Step 2: Geometry validation (False Positive Reduction)
        valid_indices = []

        for i, (box, kps) in enumerate(zip(raw_boxes, raw_kps)):
            confidence = raw_attributes[i].get("confidence")

            if self.geometry_validator.validate_detection(box, kps, confidence):
                valid_indices.append(i)

        # Filter to valid detections only
        boxes = [raw_boxes[i] for i in valid_indices]
        embeddings = [raw_embeddings[i] for i in valid_indices]
        landmarks = [raw_landmarks[i] for i in valid_indices]
        attributes = [raw_attributes[i] for i in valid_indices]

        self.logger.debug(f"After geometry validation: {len(boxes)}/{len(raw_boxes)} faces valid")

        if not boxes:
            return [], [], [], [], []

        # Step 3: NMS deduplication (Spatial deduplication)
        if self.config.enable_nms:
            confidences = [attr.get("confidence", 0.5) for attr in attributes]
            keep_indices = self.nms_deduplicator.apply_nms(boxes, confidences)

            boxes = [boxes[i] for i in keep_indices]
            embeddings = [embeddings[i] for i in keep_indices]
            landmarks = [landmarks[i] for i in keep_indices]
            attributes = [attributes[i] for i in keep_indices]

            self.logger.debug(f"After NMS: {len(boxes)} faces remain")

        # Step 4: Embedding-based deduplication (Semantic deduplication)
        if self.config.enable_embedding_dedup and len(boxes) > 1:
            # Compute quality scores if not provided
            if quality_scores is None:
                quality_scores = [self.quality_assessor.assess(frame, box) for box in boxes]

            duplicate_groups = self.embedding_deduplicator.find_duplicates(embeddings, quality_scores)

            if duplicate_groups:
                # Get indices to remove (all but best in each group)
                indices_to_remove = set()

                for group in duplicate_groups:
                    best_idx = self.embedding_deduplicator.get_best_face_index(group, quality_scores)
                    for idx in group:
                        if idx != best_idx:
                            indices_to_remove.add(idx)

                # Keep only unique faces
                keep_mask = [i not in indices_to_remove for i in range(len(boxes))]

                boxes = [boxes[i] for i, keep in enumerate(keep_mask) if keep]
                embeddings = [embeddings[i] for i, keep in enumerate(keep_mask) if keep]
                landmarks = [landmarks[i] for i, keep in enumerate(keep_mask) if keep]
                attributes = [attributes[i] for i, keep in enumerate(keep_mask) if keep]

                self.logger.debug(f"After embedding dedup: {len(boxes)} unique faces")

        # Step 5: Unknown face identification (Cross-session consistency)
        unknown_ids = []

        for embedding in embeddings:
            match = self.unknown_registry.find_matching_unknown(embedding)

            if match:
                unknown_id, similarity = match
                unknown_ids.append(unknown_id)
            else:
                # Register as new unknown
                unknown_id = self.unknown_registry.register_unknown(embedding)
                unknown_ids.append(unknown_id)

        return boxes, embeddings, landmarks, attributes, unknown_ids

    def clear_session(self) -> None:
        """
        Clear session-specific data.

        Call at the start of each new processing session to ensure
        clean session tracking while preserving persistent registry.
        """
        self.unknown_registry.clear_session_tracking()

    def get_unknown_statistics(self) -> Dict:
        """Get statistics about the unknown face registry."""
        return self.unknown_registry.get_statistics()


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================


def example_usage():
    """
    Example demonstrating how to use the enhanced face detection pipeline.
    """
    # Configuration with all three improvements enabled
    config = EnhancedFaceDetectionConfig(
        # ISSUE 1: False positive reduction
        min_face_aspect_ratio=0.6,
        max_face_aspect_ratio=1.0,
        min_detection_confidence=0.5,
        enable_landmark_validation=True,
        enable_nms=True,
        nms_iou_threshold=0.3,
        # ISSUE 2: Duplicate elimination
        enable_embedding_dedup=True,
        embedding_similarity_threshold=0.9,
        # ISSUE 3: Unknown consistency
        enable_unknown_registry=True,
        unknown_registry_path="data/unknown_faces_registry.pkl",
        unknown_similarity_threshold=0.75,
    )

    # Initialize enhanced detector
    detector = EnhancedFaceDetector(insightface_model="buffalo_l", detection_size=(640, 640), config=config)

    # Quality assessor for duplicate detection
    quality_assessor = EnhancedQualityAssessor()

    # Example: Process a frame
    # frame = cv2.imread("test_image.jpg")
    #
    # # Get quality scores for each potential face
    # boxes, _, _, _, _ = detector.detect(frame)
    # quality_scores = [quality_assessor.assess(frame, box) for box in boxes]
    #
    # # Run enhanced detection with quality scores
    # boxes, embeddings, landmarks, attributes, unknown_ids = detector.detect(
    #     frame,
    #     quality_scores=quality_scores
    # )
    #
    # print(f"Detected {len(boxes)} unique faces")
    # print(f"Unknown IDs: {unknown_ids}")

    # Print statistics
    stats = detector.get_unknown_statistics()
    print(f"Unknown registry stats: {stats}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    example_usage()
