
# from ultralytics import solutions
# from ultralytics.utils.plotting import Annotator
# from ultralytics.utils.plotting import colors

# import os
# import cv2

# import numpy as np
# import face_recognition
# import pygame

# # from ultralytics import solutions
# from ultralytics import YOLO
# from ultralytics.solutions.config import SolutionConfig
# from ultralytics.utils import LOGGER

# from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
# from ultralytics.utils.plotting import colors

# # ========== ⚙️ CONFIGURATION ==========
# SAVE_DIR = "person_cropped_face"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ========== 🔊 SOUND SETUP ==========
# pygame.mixer.init()
# ALARM_FILE = os.path.join(os.path.dirname(__file__), "../media_files/Alarm-sound-samples/humordome-security-alert-sound-453297.mp3")
# if os.path.exists(ALARM_FILE):
#     pygame.mixer.music.load(ALARM_FILE)

# # ========== 🧠 KNOWN FACE LOADER (Unchanged) ==========
# # ... (Keep your existing KNOWN_FACE_DIR loading logic here) ...
# KNOWN_FACE_DIR = "known_faces/"
# known_face_encodings, known_face_names = [], []
# if os.path.exists(KNOWN_FACE_DIR):
#     for name in os.listdir(KNOWN_FACE_DIR):
#         if os.path.isdir(os.path.join(KNOWN_FACE_DIR, name)):
#             for filename in os.listdir(os.path.join(KNOWN_FACE_DIR, name)):
#                 if filename.lower().endswith((".png", ".jpg", ".jpeg")):
#                     image = face_recognition.load_image_file(os.path.join(KNOWN_FACE_DIR, name, filename))
#                     encodings = face_recognition.face_encodings(image)
#                     if encodings:
#                         known_face_encodings.append(encodings[0])
#                         known_face_names.append(name)
# # ... [Your existing loading code] ...


# class FaceRecognitionAlarmVisionEye(solutions.VisionEye):
#     def __init__(
#         self,
#         *args,
#         known_face_encodings=None,
#         known_face_names=None,
#         face_tolerance=0.55,
#         sharpness_threshold=40,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.known_face_encodings = known_face_encodings or []
#         self.known_face_names = known_face_names or []
#         self.sound_played = False
#         self.vision_point = self.CFG["vision_point"]
#         self.records = self.CFG.get("records", 1)

#         # --- IMPROVEMENTS START ---
#         self.identity_map = {}  # Stores {track_id: "Name"} to relax recognition
#         self.best_scores = {}  # Stores {track_id: highest_quality_score}
#         self.face_tolerance = face_tolerance
#         self.sharpness_threshold = sharpness_threshold

#         # New: Add pose estimation model for anomaly detection
#         self.pose_model = YOLO("yolov8n-pose.pt")
#         # --- IMPROVEMENTS END ---

#         # New: Tracking best scores to prevent saving thousands of low-quality images
#         # Structure: { track_id: highest_quality_score }
#         # self.best_scores = {}

#     def get_quality_score(self, face_img):
#         """Calculates a score based on exposure and sharpness."""
#         if face_img.size == 0:
#             return 0

#         # 1. Exposure Score (Target brightness 128)
#         yuv = cv2.cvtColor(face_img, cv2.COLOR_BGR2YUV)
#         avg_brightness = np.mean(yuv[:, :, 0])
#         exposure_score = 255 - abs(avg_brightness - 128)

#         # 2. Sharpness Score (Laplacian variance)
#         sharpness_score = cv2.Laplacian(face_img, cv2.CV_64F).var()

#         # If too blurry, discard
#         if sharpness_score < self.sharpness_threshold:
#             return 0

#         return exposure_score + (sharpness_score * 0.5)

#     def save_best_crop(self, im0, box, person_id, name, quality):
#         """Saves the face crop if it's the best seen so far for this ID."""
#         # Only save if quality is significantly better than previous best for this track
#         if quality > self.best_scores.get(person_id, 0):
#             self.best_scores[person_id] = quality

#             # Define folder: person_cropped_face/John_Doe/ or person_cropped_face/ID_5/
#             folder_name = name.replace(" ", "_") if name != "Unknown" else f"Unknown_ID_{person_id}"
#             path = os.path.join(SAVE_DIR, folder_name)
#             os.makedirs(path, exist_ok=True)

#             # Crop logic with 20% padding
#             x1, y1, x2, y2 = map(int, box)
#             h, w, _ = im0.shape
#             pw, ph = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
#             crop = im0[max(0, y1 - ph) : min(h, y2 + ph), max(0, x1 - pw) : min(w, x2 + pw)]

#             if crop.size > 0:
#                 filename = f"best_face_score_{int(quality)}.jpg"
#                 full_path = os.path.join(path, filename)
#                 cv2.imwrite(full_path, crop)
#                 LOGGER.info(f"📸 Saved best face for {folder_name} (Score: {int(quality)})")

#     def process_quality_crop(self, im0, box, t_id, name):
#         """Finds a face within the person box, gets quality, and saves the best."""
#         x1, y1, x2, y2 = map(int, box)
#         person_roi = im0[y1:y2, x1:x2]

#         if person_roi.size == 0:
#             return

#         # Use a smaller version for faster face detection
#         scale = 0.5
#         small_roi = cv2.resize(person_roi, (0, 0), fx=scale, fy=scale)
#         rgb_small_roi = cv2.cvtColor(small_roi, cv2.COLOR_BGR2RGB)

#         face_locations = face_recognition.face_locations(rgb_small_roi, model="cnn")

#         if face_locations:
#             # Assuming one face per person for simplicity
#             f_top, f_right, f_bottom, f_left = face_locations[0]

#             # Scale back to original ROI coordinates
#             f_left, f_top, f_right, f_bottom = int(f_left / scale), int(f_top / scale), int(f_right / scale), int(
#                 f_bottom / scale
#             )

#             # Get the actual face crop from the ROI
#             face_crop = person_roi[f_top:f_bottom, f_left:f_right]

#             # Calculate quality
#             quality = self.get_quality_score(face_crop)

#             if quality > 0:
#                 # The box for saving needs to be in the main image's coordinate system
#                 abs_f_left, abs_f_top = x1 + f_left, y1 + f_top
#                 abs_f_right, abs_f_bottom = x1 + f_right, y1 + f_bottom
#                 self.save_best_crop(im0, [abs_f_left, abs_f_top, abs_f_right, abs_f_bottom], t_id, name, quality)

#     def play_sound(self):
#         """Plays the alarm sound if it's not already playing."""
#         if not self.sound_played:
#             if pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
#                 pygame.mixer.music.play()
#                 self.sound_played = True
#                 LOGGER.info("🚨 Alarm Triggered: Unknown person count reached threshold.")

#     def reset_sound(self):
#         """Stops the alarm sound and resets the state."""
#         if self.sound_played:
#             if pygame.mixer.get_init():
#                 pygame.mixer.music.stop()
#             self.sound_played = False
#             LOGGER.info("🟢 Alarm Reset: Area clear.")

#     def detect_fall(self, box):
#         """Heuristic to detect a fall: person's bounding box is wider than tall."""
#         x1, y1, x2, y2 = box
#         width, height = x2 - x1, y2 - y1
#         return width > height

#     def __call__(self, im0):
#         self.extract_tracks(im0)
#         annotator = SolutionAnnotator(im0, line_width=self.line_width)

#         # Get pose estimation results
#         pose_results = self.pose_model(im0, verbose=False)
#         for r in pose_results:
#             for kpts in r.keypoints:
#                 if kpts.has_visible:
#                     annotator.kpts(kpts.data[0], kpts.shape, kpts_color=(0, 255, 0))

#         # Cleanup: Remove IDs from identity_map that are no longer being tracked
#         current_ids = set(self.track_ids)
#         self.identity_map = {tid: name for tid, name in self.identity_map.items() if tid in current_ids}
#         unknown_person_count = 0

#         for box, t_id in zip(self.boxes, self.track_ids):
#             # STEP 1: Check if we already know this person
#             if t_id in self.identity_map:
#                 name = self.identity_map[t_id]
#                 is_known = name != "Unknown"
#             else:
#                 # STEP 2: Only run face recognition if they are NEW or UNKNOWN
#                 name, is_known = self.perform_face_recognition(im0, box)
#                 if is_known:
#                     self.identity_map[t_id] = name  # Latch the identity

#             # STEP 3: Handle Cropping for best exposure/sharpness
#             if self.best_scores.get(t_id, 0) < 220:  # Threshold for 'perfect'
#                 self.process_quality_crop(im0, box, t_id, name)

#             # Annotation logic...
#             color = (0, 255, 0) if is_known else (0, 0, 255)
#             if not is_known:
#                 unknown_person_count += 1
#             annotator.box_label(box, f"{name} ID:{t_id}", color=color)

#             # STEP 4: Anomaly Detection (Fall Detection)
#             if self.detect_fall(box):
#                 annotator.box_label(box, "FALLEN", color=(255, 0, 255)) # Purple for fallen

#         # Alarm Logic
#         if unknown_person_count >= self.records:
#             self.play_sound()
#         else:
#             self.reset_sound()

#         return SolutionResults(plot_im=annotator.result(), total_tracks=len(self.track_ids))

#     def perform_face_recognition(self, im0, person_box):
#             """Dedicated helper to isolate recognition calls."""
#             x1, y1, x2, y2 = map(int, person_box)
#             person_roi = im0[y1:y2, x1:x2]
            
#             if person_roi.size == 0:
#                 return "Unknown", False

#             # Detect face inside the person's bounding box only (Targeted Recognition)
#             rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
#             f_locations = face_recognition.face_locations(rgb_roi)
#             f_encodings = face_recognition.face_encodings(rgb_roi, f_locations)

#             for f_enc in f_encodings:
#                 distances = face_recognition.face_distance(self.known_face_encodings, f_enc)
#                 if len(distances) > 0 and np.min(distances) < self.face_tolerance:
#                     return self.known_face_names[np.argmin(distances)], True
            
#             return "Unknown", False

# # Main loop remains basically the same...


# if __name__ == "__main__":
#     # cap = cv2.VideoCapture(0)
#     cap = cv2.VideoCapture(os.path.join(os.path.dirname(__file__), "../media_files/WIN_20251103_14_11_20_Pro.mp4"))
#     # cap = cv2.VideoCapture("media_files/person/ruhama/VID_20251122_142652.mp4")
#     # cap = cv2.VideoCapture("../media_files/istockphoto-2240284006-640_adpp_is.mp4")
#     # cap = cv2.VideoCapture(os.path.join(os.path.dirname(__file__), "../media_files/istockphoto-2002566174-640_adpp_is.mp4"))
#     # cap = cv2.VideoCapture("../media_files/istockphoto-2240284006-640_adpp_is.mp4")
#     # cap = cv2.VideoCapture("../media_files/ruhama.mp4")
#     # assert cap.isOpened(), "Error reading video file"

#     # Video writer
#     w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
#     video_writer = cv2.VideoWriter("visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#     # Initialize vision eye object
#     visioneyeInterface = FaceRecognitionAlarmVisionEye(
#         show=True,  # display the output
#         model="yolo11n.pt",  # use any model that Ultralytics support, i.e, YOLOv10
#         # classes=[0, 19],  # generate visioneye view for specific classes
#         vision_point=(50, 50),  # the point, where vision will view objects and draw tracks
#         known_face_encodings=known_face_encodings,
#         known_face_names=known_face_names,
#         records=1,  # number of unknown persons to trigger alarm
#         conf=0.4,
#         iou=0.7,
#         face_tolerance=0.6,
#         sharpness_threshold=50,
#         # verbose=True,
#         # show_labels=True,
#     )


# # Process video
# while cap.isOpened():
#     success, im0 = cap.read()

#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break

#     results = visioneyeInterface(im0)

#     print(results)  # access the output

#     video_writer.write(results.plot_im)  # write the video file
#     cv2.imshow("VisionEye Output", results.plot_im)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()

import os
import cv2
import numpy as np
import face_recognition
from collections import defaultdict, deque

from ultralytics import YOLO
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults

# ================= CONFIG =================
SAVE_DIR = "person_cropped_face"
os.makedirs(SAVE_DIR, exist_ok=True)

KNOWN_FACE_DIR = "../known_faces/"

FACE_SIM_THRESHOLD = 0.6
FACE_TOLERANCE = 0.6
MAX_HISTORY = 5
MIN_FACE_SIZE = 80
SHARPNESS_THRESHOLD = 50

# COCO class mapping (partial for livestock)
CLASS_MAP = {
    0: "Person",
    15: "Cat",
    16: "Dog",
    17: "Horse",
    18: "Sheep",
    19: "Cow",
}

# ================= LOAD KNOWN FACES =================
def load_known_faces():
    encodings = []
    names = []

    if not os.path.exists(KNOWN_FACE_DIR):
        return encodings, names

    for person_name in os.listdir(KNOWN_FACE_DIR):
        person_path = os.path.join(KNOWN_FACE_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        for file in os.listdir(person_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(person_path, file)
                image = face_recognition.load_image_file(img_path)
                enc = face_recognition.face_encodings(image)
                if enc:
                    encodings.append(enc[0])
                    names.append(person_name)

    return encodings, names


# ================= UTILS =================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ================= FACE QUALITY =================
class FaceQuality:
    def __init__(self, sharpness_thresh=SHARPNESS_THRESHOLD):
        self.sharpness_thresh = sharpness_thresh

    def score(self, face):
        if face.size == 0:
            return 0

        h, w = face.shape[:2]
        if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
            return 0

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Exposure
        brightness = np.mean(gray)
        exposure = 255 - abs(brightness - 128)

        # Sharpness
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < self.sharpness_thresh:
            return 0

        # Contrast
        contrast = gray.std()

        return (0.4 * exposure) + (0.4 * sharpness) + (0.2 * contrast)


# ================= DEDUPLICATION =================
class FaceDeduplicator:
    def __init__(self):
        self.db = defaultdict(list)

    def is_duplicate(self, name, emb):
        for e in self.db[name]:
            if cosine_similarity(e, emb) > FACE_SIM_THRESHOLD:
                return True
        return False

    def add(self, name, emb):
        self.db[name].append(emb)


# # ================= FACE RECOGNITION =================
# class FaceRecognizer:
#     def __init__(self, encodings, names, tolerance=FACE_TOLERANCE):
#         self.known_encodings = encodings
#         self.known_names = names
#         self.tolerance = tolerance

#     def recognize(self, face_img):
#         rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

#         locations = face_recognition.face_locations(rgb)
#         if not locations:
#             return None, "Unknown"

#         encodings = face_recognition.face_encodings(rgb, locations)
#         if not encodings:
#             return None, "Unknown"

#         # Select largest face
#         areas = [(b[2]-b[0])*(b[1]-b[3]) for b in locations]
#         idx = np.argmax(areas)

#         emb = encodings[idx]

#         if len(self.known_encodings) == 0:
#             return emb, "Unknown"

#         distances = face_recognition.face_distance(self.known_encodings, emb)
#         min_idx = np.argmin(distances)

#         if distances[min_idx] < self.tolerance:
#             return emb, self.known_names[min_idx]

#         return emb, "Unknown"

import faiss
import numpy as np
import cv2
import face_recognition

class FaceRecognizer:
    def __init__(self, encodings, names, tolerance=FACE_TOLERANCE):
        self.known_names = names
        self.tolerance = tolerance
        
        # 1. Initialize FAISS Index
        if len(encodings) > 0:
            # Face encodings are 128-dimensional vectors
            dimension = 128 
            self.index = faiss.IndexFlatL2(dimension)
            
            # Convert list of encodings to a float32 numpy array (required by FAISS)
            embeddings_matrix = np.array(encodings).astype('float32')
            
            # 2. Add the known faces to the index
            self.index.add(embeddings_matrix)
        else:
            self.index = None

    def recognize(self, face_img):
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        
        if not locations:
            return None, "Unknown"

        encodings = face_recognition.face_encodings(rgb, locations)
        if not encodings:
            return None, "Unknown"

        # Select largest face
        areas = [(b[2]-b[0])*(b[1]-b[3]) for b in locations]
        idx = np.argmax(areas)
        query_emb = encodings[idx].astype('float32').reshape(1, -1)

        if self.index is None:
            return query_emb[0], "Unknown"

        # 3. FAISS Search: Find the 1 nearest neighbor (k=1)
        # distances (D) and indices (I)
        D, I = self.index.search(query_emb, k=1)

        best_dist = D[0][0]
        best_idx = I[0][0]

        # FAISS returns -1 if no match is found or index is empty
        if best_idx != -1 and best_dist < self.tolerance:
            return query_emb[0], self.known_names[best_idx]

        return query_emb[0], "Unknown"

# ================= LIVESTOCK =================
class LivestockMonitor:
    def __init__(self):
        self.class_counts = defaultdict(int)

    def update(self, classes):
        self.class_counts.clear()
        for cls in classes:
            self.class_counts[int(cls)] += 1

    def get_counts(self):
        return dict(self.class_counts)


# ================= MAIN PIPELINE =================
class SmartVision(BaseSolution):
    def __init__(self, *args, known_face_encodings=None, known_face_names=None, **kwargs):
        # Extract custom face recognition parameters BEFORE passing to parent
        # This prevents ValueError from SolutionConfig which only accepts valid Ultralytics parameters
        self.known_face_encodings = known_face_encodings or []
        self.known_face_names = known_face_names or []
        
        # Pass only valid Ultralytics solution parameters to parent class
        super().__init__(*args, **kwargs)
        
        # Initialize face recognition components with extracted parameters
        self.recognizer = FaceRecognizer(self.known_face_encodings, self.known_face_names)
        self.quality = FaceQuality()
        self.dedup = FaceDeduplicator()
        self.livestock = LivestockMonitor()

        self.identity_map = {}
        self.history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
        self.best_scores = {}

    def smooth_identity(self, t_id, name):
        self.history[t_id].append(name)
        return max(set(self.history[t_id]), key=self.history[t_id].count)

    def extract_roi(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        return frame[y1:y2, x1:x2]

    def save_face(self, face, name, emb, score):
        if emb is None:
            return

        if self.dedup.is_duplicate(name, emb):
            return

        folder = os.path.join(SAVE_DIR, name.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)

        filename = f"{int(score)}.jpg"
        cv2.imwrite(os.path.join(folder, filename), face)

        self.dedup.add(name, emb)

    def __call__(self, im0):
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0)

        # Update livestock counts
        self.livestock.update(self.clss)

        for box, t_id, cls in zip(self.boxes, self.track_ids, self.clss):

            # -------- PERSON PROCESSING --------
            if int(cls) == 0:
                roi = self.extract_roi(im0, box)

                emb, name = self.recognizer.recognize(roi)
                name = self.smooth_identity(t_id, name)

                self.identity_map[t_id] = name

                score = self.quality.score(roi)

                if score > self.best_scores.get(t_id, 0):
                    self.best_scores[t_id] = score
                    self.save_face(roi, name, emb, score)

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                annotator.box_label(box, f"{name} ID:{t_id}", color=color)

            # -------- LIVESTOCK --------
            else:
                label = CLASS_MAP.get(int(cls), f"Class {int(cls)}")
                annotator.box_label(box, label, color=(255, 255, 0))

        # -------- DISPLAY COUNTS --------
        y = 30
        for cls, count in self.livestock.get_counts().items():
            label = CLASS_MAP.get(cls, f"Class {cls}")
            cv2.putText(im0, f"{label}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 25

        return SolutionResults(plot_im=annotator.result())


# ================= MAIN =================
if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)  # or video path
    cap = cv2.VideoCapture("../media_files/WIN_20260227_22_00_29_Pro.mp4")

    known_encodings, known_names = load_known_faces()

    vision = SmartVision(
        model="yolo11n.pt",
        known_face_encodings=known_encodings,
        known_face_names=known_names,
        conf=0.4,
        iou=0.7,
        show=True
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = vision(frame)

        cv2.imshow("Smart Vision", results.plot_im)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()