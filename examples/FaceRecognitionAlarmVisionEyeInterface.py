import cv2

from ultralytics import solutions

# import cv2
# from numpy import source

from ultralytics import solutions
# from ultralytics.utils.plotting import Annotator

import os

# import cv2
import numpy as np
import face_recognition
import pygame
import mediapipe as mp

# from ultralytics import solutions
# from ultralytics import YOLO
# from ultralytics.solutions.config import SolutionConfig
from ultralytics.utils import LOGGER

# Fix 1: Remove unused BaseSolution import
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors
from scipy.spatial import distance as dist
import asyncio
import threading

# ========== 🔧 FACE QUALITY EXTRACTION CONFIGURATION ==========
THRESHOLD_MIN = 80  # Minimum acceptable brightness
THRESHOLD_MAX = 200  # Maximum acceptable brightness
CENTROID_DISTANCE_THRESHOLD = 115  # Pixels for matching same person across frames
SAVE_DIR = "cropped_faces"  # Directory to save best face crops


def get_quality_score(image):
    """
    Calculates a quality score based on sharpness (Laplacian) and exposure.

    Args:
        image: BGR image (numpy array)

    Returns:
        tuple: (quality_score, brightness)
            - quality_score: Combined score based on sharpness and exposure
            - brightness: Mean brightness value from YUV channel
    """
    # Calculate sharpness using Laplacian variance
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()

    # Calculate brightness using YUV color space
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    brightness = np.mean(yuv[:, :, 0])

    # Calculate exposure score (higher when brightness is closer to ideal 128)
    exposure_score = 255 - abs(brightness - 128)

    # Combined quality score
    quality_score = sharpness * exposure_score

    return quality_score, brightness


class CentroidFaceTracker:
    """
    Centroid-based face tracking to associate face detections across frames.
    Maintains persistent IDs for each unique face using centroid distance thresholding.
    """

    def __init__(self, distance_threshold: int = CENTROID_DISTANCE_THRESHOLD):
        """
        Initialize the centroid tracker.

        Args:
            distance_threshold: Maximum distance (pixels) to consider same person
        """
        self.distance_threshold = distance_threshold
        self.trackers: dict[int, tuple[int, int]] = {}  # {track_id: centroid (cx, cy)}
        self.next_id: int = 0
        self.best_shots: dict[int, dict] = {}  # {track_id: {'score': float, 'crop': image, 'brightness': float}}

    def update(self, face_boxes: list[tuple], frame: np.ndarray) -> dict[int, tuple]:
        """
        Update tracker with new face detections and return matched track IDs.

        Args:
            face_boxes: List of face bounding boxes as (x, y, w, h)
            frame: Current frame for quality evaluation

        Returns:
            Dictionary mapping face index to (track_id, centroid)
        """
        current_frame_faces = []

        # Extract centroids from face boxes
        for box in face_boxes:
            x, y, w, h = box
            cx, cy = x + w // 2, y + h // 2
            current_frame_faces.append({"box": box, "centroid": (cx, cy)})

        matched_faces = {}

        # Match detected faces to persistent IDs using Centroid Tracking
        if current_frame_faces:
            if not self.trackers:
                # First frame: assign new IDs to all faces
                for f in face_boxes:
                    self.trackers[self.next_id] = f["centroid"]
                    matched_faces[len(matched_faces)] = (self.next_id, f["centroid"])
                    self.next_id += 1
            else:
                ids = list(self.trackers.keys())
                coords = list(self.trackers.values())

                for f in current_frame_faces:
                    # Find closest existing person
                    distances = dist.cdist([f["centroid"]], coords)
                    idx = np.argmin(distances)

                    if distances[0][idx] < self.distance_threshold:
                        # Same person - update tracker
                        tid = ids[idx]
                        self.trackers[tid] = f["centroid"]

                        # Evaluate quality for current track
                        x, y, w, h = f["box"]
                        face_roi = frame[y : y + h, x : x + w]

                        if face_roi.size > 0:
                            score, brightness = get_quality_score(face_roi)

                            # Only update best shot if brightness is within acceptable range
                            if THRESHOLD_MIN < brightness < THRESHOLD_MAX:
                                if tid not in self.best_shots or score > self.best_shots[tid]["score"]:
                                    # Create crop with padding
                                    ih, iw = frame.shape[:2]
                                    pw, ph = int(w * 0.2), int(h * 0.2)
                                    crop = frame[
                                        max(0, y - ph) : min(ih, y + h + ph), max(0, x - pw) : min(iw, x + w + pw)
                                    ]
                                    if crop.size > 0:
                                        self.best_shots[tid] = {
                                            "score": score,
                                            "crop": crop.copy(),
                                            "brightness": brightness,
                                        }

                        matched_faces[len(matched_faces)] = (tid, f["centroid"])
                    else:
                        # New person detected
                        self.trackers[self.next_id] = f["centroid"]
                        matched_faces[len(matched_faces)] = (self.next_id, f["centroid"])
                        self.next_id += 1

        return matched_faces

    def save_best_crops(self, save_dir: str = SAVE_DIR) -> list[str]:
        """
        Save all best face crops to the specified directory.

        Args:
            save_dir: Directory to save cropped faces

        Returns:
            List of saved file paths
        """
        import os

        os.makedirs(save_dir, exist_ok=True)

        saved_files = []
        for tid, data in self.best_shots.items():
            filename = os.path.join(save_dir, f"person_{tid}_best.jpg")
            cv2.imwrite(filename, data["crop"])
            saved_files.append(filename)
            LOGGER.info(f"✨ Saved: {filename} (Brightness: {data['brightness']:.1f}, Score: {data['score']:.1f})")

        return saved_files

    def get_best_shots_summary(self) -> dict:
        """Get summary of best shots for each tracked person."""
        return {
            tid: {"brightness": data["brightness"], "score": data["score"]} for tid, data in self.best_shots.items()
        }

    def reset(self):
        """Reset all trackers and best shots."""
        self.trackers.clear()
        self.best_shots.clear()
        self.next_id = 0


# ========== 🔊 SOUND SETUP ==========
pygame.mixer.init()
ALARM_FILE = "../media_files/Alarm-sound-samples/humordome-security-alert-sound-453297.mp3"
if os.path.exists(ALARM_FILE):
    pygame.mixer.music.load(ALARM_FILE)
else:
    print(f"[WARNING] Alarm file '{ALARM_FILE}' not found.")


# ========== 🧠 KNOWN FACE ENCODING LOADER ==========
KNOWN_FACE_DIR = "../family_members/"
known_face_encodings, known_face_names = [], []

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


class AiSecurityGuard(solutions.VisionEye):
    """
    Enhanced AI Security Guard with face recognition, alarm system, and quality-based face extraction.

    Extends the base VisionEye class with:
    - Face recognition using face_recognition library
    - Alarm system triggered by unknown person detection
    - Centroid-based face tracking across frames
    - Quality-based face extraction (Laplacian sharpness + brightness scoring)
    - Best face crop saving for each unique person
    """

    def __init__(
        self,
        *args,
        known_face_encodings=None,
        known_face_names=None,
        enable_face_quality_extraction: bool = True,
        face_quality_distance_threshold: int = CENTROID_DISTANCE_THRESHOLD,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.known_face_encodings = known_face_encodings or []
        self.known_face_names = known_face_names or []
        self.sound_played = False
        # Best practice: Set face recognition tolerance during initialization
        self.face_tolerance = 0.55
        self.vision_point = self.CFG["vision_point"]
        self.records = self.CFG.get("records", 1)

        # Face quality extraction settings
        self.enable_face_quality_extraction = enable_face_quality_extraction
        if self.enable_face_quality_extraction:
            self.face_tracker = CentroidFaceTracker(distance_threshold=face_quality_distance_threshold)
        else:
            self.face_tracker = None

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
        # Face boxes for quality tracking (updated during processing)
        self.current_face_boxes: list[tuple] = []

    def play_sound(self):
        """Plays the alarm sound if it's not already playing."""
        if not self.sound_played:
            if pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
                self.sound_played = True
                LOGGER.info("🚨 Alarm Triggered: Unknown person count reached threshold.")

    def reset_sound(self):
        """Stops the alarm sound and resets the state."""
        if self.sound_played:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
            self.sound_played = False
            LOGGER.info("🟢 Alarm Reset: Area clear.")

    def save_best_face_crops(self, save_dir: str = SAVE_DIR) -> list[str]:
        """
        Save the best quality face crops for all tracked persons.

        Args:
            save_dir: Directory to save cropped faces

        Returns:
            List of saved file paths
        """
        if self.face_tracker:
            return self.face_tracker.save_best_crops(save_dir)
        return []

    def get_face_tracking_summary(self) -> dict:
        """
        Get summary of face tracking statistics.

        Returns:
            Dictionary with tracking stats and best shots info
        """
        if self.face_tracker:
            return {
                "total_unique_faces": len(self.face_tracker.trackers),
                "best_shots": self.face_tracker.get_best_shots_summary(),
            }
        return {"total_unique_faces": 0, "best_shots": {}}

    def reset_face_tracker(self):
        """Reset the face tracker and clear all best face crops."""
        if self.face_tracker:
            self.face_tracker.reset()
            LOGGER.info("🧹 Face tracker reset.")

    def __call__(self, im0):
        """
        Processes a single frame for person detection and face recognition.
        This implementation follows best practices for accuracy and performance.
        """
        # 1. Get person detections from the base class
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)

        unknown_person_count = 0

        # 2. Optimize by finding all faces in the frame at once (on a smaller version)
        # Using MediaPipe for Face Detection
        ih, iw, _ = im0.shape
        rgb_frame = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)

        face_locations = []
        self.current_face_boxes = []

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * iw))
                y = max(0, int(bbox.ymin * ih))
                w = max(1, int(bbox.width * iw))
                h = max(1, int(bbox.height * ih))

                self.current_face_boxes.append((x, y, w, h))
                # face_recognition expects (top, right, bottom, left)
                face_locations.append((y, x + w, y + h, x))

        # Get encodings using the full frame and locations found by MediaPipe
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        else:
            face_encodings = []

        # Update centroid face tracker with detected faces for quality extraction
        if self.enable_face_quality_extraction and self.face_tracker and self.current_face_boxes:
            self.face_tracker.update(self.current_face_boxes, im0)

        # 3. Iterate through detected objects from YOLO
        for cls, t_id, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            # Fix 3: Fixed logic - Skip non-person classes (person is class 0 in COCO)
            if int(cls) != 0:
                # For non-person classes, use default labeling
                annotator.box_label(box, label=self.adjust_box_label(cls, conf, t_id), color=colors(int(t_id), True))
                annotator.visioneye(box, self.vision_point)
                continue

            # Person class processing - perform face recognition
            name = "Unknown"
            is_known = False

            # 4. Associate faces with person boxes
            # Check if any detected face is inside this person's bounding box
            person_box_left, person_box_top, person_box_right, person_box_bottom = map(int, box)

            for (face_top, face_right, face_bottom, face_left), face_encoding in zip(face_locations, face_encodings):
                # Check if the center of the face is inside the person's box
                face_center_x = (face_left + face_right) // 2
                face_center_y = (face_top + face_bottom) // 2

                if (
                    person_box_left <= face_center_x <= person_box_right
                    and person_box_top <= face_center_y <= person_box_bottom
                ):
                    # 5. Use robust face matching for the associated face
                    if self.known_face_encodings:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        # Fix 4: Use self.face_tolerance instead of hardcoded value
                        if face_distances[best_match_index] < self.face_tolerance:
                            name = self.known_face_names[best_match_index]
                            is_known = True

                    # Once a face is matched to this person, stop checking other faces
                    break

            # 6. Update counter and draw labels
            if not is_known:
                unknown_person_count += 1
                color = (0, 0, 255)  # Red for Unknown
                label = f"Unknown"
            else:
                color = (0, 255, 0)  # Green for Known
                label = f"{name}"

            # Build base label from the existing adjust_box_label()
            base_label = self.adjust_box_label(int(cls), float(conf) if conf is not None else 0.0, t_id)

            # Fix 5: Removed redundant check - we already know cls == 0 here
            prefix = str(self.CFG.get("person_label_prefix", label))
            custom_label = f"{prefix}:"
            # if base_label exists, concat both for full display
            final_label = f"{custom_label} {base_label}" if base_label else custom_label

            # draw final label and vision eye mapping
            annotator.box_label(box, label=final_label, color=colors(int(t_id), True))
            annotator.visioneye(box, self.vision_point)

        # 7. Trigger alarm based on the COUNT of unknown people and the 'records' threshold
        if unknown_person_count >= self.records:
            # Schedule play_sound asynchronously (uses asyncio.to_thread when an event loop is running,
            # otherwise falls back to a daemon thread). This avoids blocking the main detection loop.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(asyncio.to_thread(self.play_sound))
            except RuntimeError:
                # No running asyncio loop (common in regular scripts), use a background thread
                threading.Thread(target=self.play_sound, daemon=True).start()
            except Exception as e:
                LOGGER.exception("Failed to schedule play_sound asynchronously: %s", e)
                # As a last resort, call synchronously (play_sound is idempotent)
                self.play_sound()
        else:
            self.reset_sound()

        plot_im = annotator.result()
        self.display_output(plot_im)

        # Display track count on the frame
        total_tracks = len(getattr(self, "track_ids", []))
        cv2.putText(plot_im, f"Tracks: {total_tracks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display face tracking info if enabled
        if self.enable_face_quality_extraction and self.face_tracker:
            face_summary = self.get_face_tracking_summary()
            unique_faces = face_summary.get("total_unique_faces", 0)
            cv2.putText(plot_im, f"Faces: {unique_faces}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))


# cap = cv2.VideoCapture("../media_files/ruhama.mp4")
# cap = cv2.VideoCapture("../media_files/istockphoto-2002563994-640_adpp_is.mp4")
# cap = cv2.VideoCapture("../media_files/WIN_20260227_22_00_29_Pro.mp4")
cap = cv2.VideoCapture("../media_files/istockphoto-2002566174-640_adpp_is.mp4")
# cap = cv2.VideoCapture("../media_files/istockphoto-2240272228-640_adpp_is.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Define frontal view point (top-center)
# frontal_point = (w // 2, 10)
center_point = (w // 2, h // 2)
opposite_point = (w // 2 - 250, h - 300)

# Initialize vision eye object
# visioneye = solutions.VisionEye(
#     show=True,  # display the output
#     model="yolo12x.pt",  # use any model that Ultralytics supports, e.g., YOLOv10
#     # classes=[0, 2],  # generate visioneye view for specific classes
#     vision_point=(50, 50),  # the point where VisionEye will view objects and draw tracks
# )
AiSecurityGuard = AiSecurityGuard(
    show=True,  # display the output
    model="yolo26m.pt",  # use any model that Ultralytics supports, e.g., YOLOv10
    classes=[0, 2],  # generate visioneye view for specific classes
    # vision_point=(20, 20),  # the point where VisionEye will view objects and draw tracks
    vision_point=opposite_point,  # the point where VisionEye will view objects and draw tracks
    conf=0.1,
    iou=0.9,
    # verbose=True,
    # persist=True,
    tracker="bytetrack.yaml",
    known_face_encodings=known_face_encodings,
    known_face_names=known_face_names,
    records=1,
    enable_face_quality_extraction=True,  # Enable quality-based face extraction
    face_quality_distance_threshold=115,  # Centroid distance threshold for face tracking
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

# Save best face crops after video processing completes
print("\n" + "=" * 50)
print("Saving best face crops...")
saved_files = AiSecurityGuard.save_best_face_crops()
print(f"✅ Saved {len(saved_files)} best face crops to '{SAVE_DIR}' directory")

# Print face tracking summary
face_summary = AiSecurityGuard.get_face_tracking_summary()
print(f"\n📊 Face Tracking Summary:")
print(f"   - Total unique faces tracked: {face_summary['total_unique_faces']}")
for tid, info in face_summary.get("best_shots", {}).items():
    print(f"   - Person {tid}: Brightness={info['brightness']:.1f}, Quality Score={info['score']:.1f}")
