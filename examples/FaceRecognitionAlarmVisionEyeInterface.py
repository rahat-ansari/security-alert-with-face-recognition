import cv2
import os
from pathlib import Path
import numpy as np
import face_recognition
import pygame

from ultralytics import solutions
from ultralytics.utils import LOGGER
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


# ========== ⚙️ CONFIGURATION ==========
class SecurityConfig:
    """Configuration for AI Security Guard system."""

    BASE_DIR = Path(__file__).parent.parent
    KNOWN_FACE_DIR = BASE_DIR / "family_members"
    ALARM_FILE = BASE_DIR / "security_alart.mp3"
    VIDEO_INPUT = BASE_DIR / "media_files" / "গোয়াল থেকে গরু চুরির দু_র্ধ_র্ষ দৃশ্য ধরা পড়লো সিসিটিভিতে 720p.mp4"
    VIDEO_OUTPUT = Path(__file__).parent / "visioneye_output.avi"

    # Face recognition parameters
    FACE_TOLERANCE = 0.55  # Lower = stricter matching
    RESIZE_SCALE = 0.25  # Scale for face detection (0.25 = 4x smaller)

    # Model parameters
    MODEL = "yolo11m.pt"
    CONFIDENCE = 0.3
    IOU = 0.5
    UNKNOWN_THRESHOLD = 15  # Trigger alarm when N unknown persons detected


def load_known_faces(face_dir: Path) -> tuple:
    """Load face encodings from directory.

    Args:
        face_dir: Path to directory with person subdirectories containing images

    Returns:
        Tuple of (encodings_list, names_list)
    """
    encodings, names = [], []
    face_dir = Path(face_dir)

    if not face_dir.exists():
        LOGGER.warning(f"Face directory not found: {face_dir}")
        return encodings, names

    for person_dir in face_dir.iterdir():
        if not person_dir.is_dir():
            continue
        for img_path in person_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                img = face_recognition.load_image_file(str(img_path))
                encs = face_recognition.face_encodings(img)
                if encs:
                    encodings.append(encs[0])
                    names.append(person_dir.name)
                    LOGGER.info(f"Loaded face for {person_dir.name} from {img_path.name}")
            except Exception as e:
                LOGGER.error(f"Failed loading {img_path}: {e}")

    if encodings:
        LOGGER.info(f"Successfully loaded {len(encodings)} face encoding(s) from {len(set(names))} person(s)")
    return encodings, names


def init_alarm_system(alarm_file: Path) -> bool:
    """Initialize pygame mixer and load alarm file.

    Args:
        alarm_file: Path to alarm audio file

    Returns:
        True if alarm is ready, False otherwise
    """
    try:
        pygame.mixer.init()
        if alarm_file.exists():
            pygame.mixer.music.load(str(alarm_file))
            LOGGER.info(f"Alarm loaded: {alarm_file}")
            return True
        else:
            LOGGER.warning(f"Alarm file not found: {alarm_file}")
            return False
    except Exception as e:
        LOGGER.error(f"Failed to initialize alarm: {e}")
        return False


# ========== 🔊 INITIALIZE SYSTEMS ==========
config = SecurityConfig()
alarm_ready = init_alarm_system(config.ALARM_FILE)
known_face_encodings, known_face_names = load_known_faces(config.KNOWN_FACE_DIR)


class AiSecurityGuard(solutions.VisionEye):
    """AI-powered security system with face recognition and alarm."""

    def __init__(self, *args, known_face_encodings=None, known_face_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.known_face_encodings = known_face_encodings or []
        self.known_face_names = known_face_names or []
        self.sound_played = False
        self.face_tolerance = config.FACE_TOLERANCE
        self.resize_scale = config.RESIZE_SCALE
        self.vision_point = self.CFG.get("vision_point", (50, 50))
        self.unknown_threshold = self.CFG.get("records", config.UNKNOWN_THRESHOLD)

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

    def match_face(self, face_encoding) -> tuple:
        """Match a face encoding to known faces.

        Args:
            face_encoding: Face encoding from face_recognition library

        Returns:
            Tuple of (person_name, is_known)
        """
        if not self.known_face_encodings:
            return "Unknown", False

        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(distances)

        if distances[best_match_index] < self.face_tolerance:
            return self.known_face_names[best_match_index], True
        return "Unknown", False

    def generate_label(self, name: str, is_known: bool, base_label: str) -> str:
        """Generate display label for person.

        Args:
            name: Person's name
            is_known: Whether person is recognized
            base_label: Base label from detection

        Returns:
            Formatted label string
        """
        status = "✓" if is_known else "✗"
        prefix = f"{status} {name}"
        return f"{prefix} | {base_label}" if base_label else prefix

    def __call__(self, im0):
        """Process frame for person detection and face recognition.

        Args:
            im0: Input frame

        Returns:
            SolutionResults with annotated frame
        """
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)
        unknown_person_count = 0

        # Detect faces in downscaled frame for efficiency
        small_frame = cv2.resize(im0, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Scale factor to convert back to original image size
        scale_factor = 1 / self.resize_scale

        # Process each detected person
        for cls, t_id, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            cls_int = int(cls)
            if cls_int != 0:  # Only process persons (COCO class 0)
                annotator.box_label(box, label=self.adjust_box_label(cls, conf, t_id), color=colors(int(t_id), True))
                annotator.visioneye(box, self.vision_point)
                continue

            name = "Unknown"
            is_known = False
            person_box = tuple(map(int, box))
            person_box_left, person_box_top, person_box_right, person_box_bottom = person_box

            # Find face belonging to this person
            for (face_top, face_right, face_bottom, face_left), face_encoding in zip(face_locations, face_encodings):
                # Scale face coordinates back to original image size
                face_top = int(face_top * scale_factor)
                face_right = int(face_right * scale_factor)
                face_bottom = int(face_bottom * scale_factor)
                face_left = int(face_left * scale_factor)

                # Check if face center is inside person bounding box
                face_center_x = (face_left + face_right) // 2
                face_center_y = (face_top + face_bottom) // 2

                if (
                    person_box_left <= face_center_x <= person_box_right
                    and person_box_top <= face_center_y <= person_box_bottom
                ):
                    name, is_known = self.match_face(face_encoding)
                    break

            # Update counter and visualization
            if not is_known:
                unknown_person_count += 1
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known

            base_label = self.adjust_box_label(cls_int, float(conf) if conf is not None else 0.0, t_id)
            final_label = self.generate_label(name, is_known, base_label)
            annotator.box_label(box, label=final_label, color=color)
            annotator.visioneye(box, self.vision_point)

        # Trigger alarm if unknown person count exceeds threshold
        if unknown_person_count >= self.unknown_threshold:
            self.play_sound()
        else:
            self.reset_sound()

        plot_im = annotator.result()
        self.display_output(plot_im)

        # Display statistics on frame
        total_tracks = len(getattr(self, "track_ids", []))
        cv2.putText(plot_im, f"Tracks: {total_tracks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return SolutionResults(plot_im=plot_im, total_tracks=total_tracks)


# ========== 🎬 MAIN VIDEO PROCESSING ==========
def main():
    """Main video processing loop."""
    # Validate input file
    if not config.VIDEO_INPUT.exists():
        LOGGER.error(f"Video file not found: {config.VIDEO_INPUT}")
        return

    if not known_face_encodings:
        LOGGER.warning("No known faces loaded. Running in detection-only mode.")

    # Open video
    cap = cv2.VideoCapture(str(config.VIDEO_INPUT))
    if not cap.isOpened():
        LOGGER.error(f"Failed to open video: {config.VIDEO_INPUT}")
        return

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0 or w == 0 or h == 0:
        LOGGER.error("Invalid video properties")
        cap.release()
        return

    # Initialize video writer
    video_writer = cv2.VideoWriter(str(config.VIDEO_OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not video_writer.isOpened():
        LOGGER.error(f"Failed to initialize video writer: {config.VIDEO_OUTPUT}")
        cap.release()
        return

    # Initialize security guard
    ai_security_guard = AiSecurityGuard(
        show=True,
        model=config.MODEL,
        conf=config.CONFIDENCE,
        iou=config.IOU,
        known_face_encodings=known_face_encodings,
        known_face_names=known_face_names,
        records=config.UNKNOWN_THRESHOLD,
    )

    # Process video
    frame_count = 0
    try:
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                LOGGER.info("Video processing completed.")
                break

            results = ai_security_guard(im0)
            video_writer.write(results.plot_im)
            frame_count += 1

            if frame_count % 30 == 0:
                LOGGER.debug(f"Processed {frame_count} frames")

    except KeyboardInterrupt:
        LOGGER.info("Video processing interrupted by user.")
    except Exception as e:
        LOGGER.error(f"Error during video processing: {e}")
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        LOGGER.info(f"Output saved to: {config.VIDEO_OUTPUT}")
        LOGGER.info(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()
