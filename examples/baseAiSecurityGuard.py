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

# from ultralytics import solutions
# from ultralytics import YOLO
# from ultralytics.solutions.config import SolutionConfig
from ultralytics.utils import LOGGER

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors
import asyncio
import threading

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
    def __init__(self, *args, known_face_encodings=None, known_face_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.known_face_encodings = known_face_encodings or []
        self.known_face_names = known_face_names or []
        self.sound_played = False
        # Best practice: Set face recognition tolerance during initialization
        self.face_tolerance = 0.55
        self.vision_point = self.CFG["vision_point"]
        self.records = self.CFG.get("records", 1)
        # self.show = self.CFG.get("show", True)

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
        # This is much faster than processing crops for each person.
        h, w, _ = im0.shape
        small_frame = cv2.resize(im0, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # 3. Iterate through detected PERSONS from YOLO
        for cls, t_id, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            if int(cls) == 0:  # Skip if not a person
                name = "Unknown"
                is_known = False

                # 4. Associate faces with person boxes
                # Check if any detected face is inside this person's bounding box
                person_box_left, person_box_top, person_box_right, person_box_bottom = map(int, box)

                for (face_top, face_right, face_bottom, face_left), face_encoding in zip(
                    face_locations, face_encodings
                ):
                    # Scale face locations back to original image size
                    face_top *= 4
                    face_right *= 4
                    face_bottom *= 4
                    face_left *= 4

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

                            if face_distances[best_match_index] < self.face_tolerance:
                                name = self.known_face_names[best_match_index]
                                is_known = True

                        # Once a face is matched to this person, stop checking other faces
                        break

                # 6. Update counter and draw labels
                if not is_known:
                    unknown_person_count += 1
                    color = (0, 0, 255)  # Red for Unknown
                    # label = f"Unknown ({conf:.2f})"
                    label = f"Unknown"
                else:
                    color = (0, 255, 0)  # Green for Known
                    label = f"{name}"
                    # label = f"{name} ({conf:.2f})"

                # annotator.box_label(box, label, color=color)

                # annotator.visioneye(box, self.vision_point)
                # build base label from the existing adjust_box_label()
                base_label = self.adjust_box_label(int(cls), float(conf) if conf is not None else 0.0, t_id)

                # custom label for 'person' class (COCO id 0). Use CFG override if provided.
                if int(cls) == 0:
                    prefix = str(self.CFG.get("person_label_prefix", label))
                    custom_label = f"{prefix}:"
                    # if base_label exists, concat both for full display
                    final_label = f"{custom_label} {base_label}" if base_label else custom_label
                else:
                    final_label = base_label

                # draw final label and vision eye mapping
                annotator.box_label(box, label=final_label, color=colors(int(t_id), True))
            else:
                # For non-person classes, use default labeling
                annotator.box_label(box, label=self.adjust_box_label(cls, conf, t_id), color=colors(int(t_id), True))

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

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("../media_files/ruhama.mp4")
# cap = cv2.VideoCapture("../media_files/istockphoto-2002563994-640_adpp_is.mp4")
# cap = cv2.VideoCapture("../media_files/WIN_20260227_22_00_29_Pro.mp4")
# cap = cv2.VideoCapture("../media_files/istockphoto-2240272228-640_adpp_is.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Define frontal view point (top-center)
# frontal_point = (w // 2, 10)
center_point = (w // 2, h // 2)
opposite_point = (w // 2 - 250, h - 10)

# Initialize vision eye object
# visioneye = solutions.VisionEye(
#     show=True,  # display the output
#     model="yolo12x.pt",  # use any model that Ultralytics supports, e.g., YOLOv10
#     # classes=[0, 2],  # generate visioneye view for specific classes
#     vision_point=(50, 50),  # the point where VisionEye will view objects and draw tracks
# )
AiSecurityGuard = AiSecurityGuard(
    show=True,  # display the output
    model="yolo26m-pose.pt",  # use any model that Ultralytics supports, e.g., YOLOv10
    classes=[0, 2],  # generate visioneye view for specific classes    
    vision_point=opposite_point,  # the point where VisionEye will view objects and draw tracks
    conf=0.2,
    iou=0.7,
    # persist=True,  # persist tracks even when objects are not detected in the current frame   
    tracker="bytetrack.yaml",  # use ByteTrack for more robust tracking
    known_face_encodings=known_face_encodings,
    known_face_names=known_face_names,
    records=1,
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
