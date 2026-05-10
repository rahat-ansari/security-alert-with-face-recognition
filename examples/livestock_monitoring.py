
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import os

# ========== ⚙️ CONFIGURATION ==========
# Define the classes you want to count. These are from the COCO dataset.
# You can add more classes if your model supports them.
LIVESTOCK_CLASSES = {
    'cow': 19,
    'horse': 20,
    'sheep': 21,
    'dog': 16, # Often found on farms
    'cat': 15  # Often found on farms
}
# Reverse mapping for easy lookup
CLASS_NAMES = {v: k for k, v in LIVESTOCK_CLASSES.items()}


def main(video_path="livestock_video.mp4", output_path="livestock_output.avi"):
    """
    Main function to process a video for livestock monitoring.
    """
    # ========== 🧠 MODEL INITIALIZATION ==========
    # Using a standard YOLOv8 model trained on COCO dataset.
    # For better results, you might want to fine-tune a model on a specific livestock dataset.
    model = YOLO("yolov8n.pt")

    # ========== 🎬 VIDEO SETUP ==========
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        # As a fallback, try to find a sample video in the media_files directory
        fallback_path = os.path.join(os.path.dirname(__file__), "../media_files/conveyer apple and bottle counting/conveyer apple and bottle counting.mp4")
        if os.path.exists(fallback_path):
            print(f"Using fallback video: {fallback_path}")
            video_path = fallback_path
        else:
            print("Could not find a video to process. Exiting.")
            return

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error opening video file {video_path}"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("Processing video for livestock monitoring...")

    # ========== 🔄 PROCESSING LOOP ==========
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video processing completed.")
            break

        # Perform detection
        results = model(im0, verbose=False)

        # Initialize annotator and class counts for the current frame
        annotator = Annotator(im0, line_width=2)
        class_counts = {name: 0 for name in LIVESTOCK_CLASSES.keys()}

        # Process detections
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                if cls_id in CLASS_NAMES:
                    class_name = CLASS_NAMES[cls_id]
                    class_counts[class_name] += 1
                    annotator.box_label(box.xyxy[0], f'{class_name} {box.conf:.2f}', color=(0, 255, 0))

        # ========== 📊 DISPLAY COUNTS ==========
        # Create a display text for the counts
        y_offset = 30
        for class_name, count in class_counts.items():
            text = f"{class_name.capitalize()}: {count}"
            cv2.putText(im0, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 30


        # Show the output frame
        cv2.imshow("Livestock Monitoring", im0)
        video_writer.write(im0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ========== 🧹 CLEANUP ==========
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {output_path}")

if __name__ == "__main__":
    # You can change the video path here to your own video file.
    # The default is a placeholder. A fallback to a sample video is included.
    main()
