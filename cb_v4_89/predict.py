import cv2
import os
import mediapipe as mp
import tkinter as tk
from ultralytics import YOLO
from tkinter import messagebox


# Load YOLO model on Apple Metal GPU
model = YOLO("runs/detect/train3/weights/best.pt").to("mps")


# Load Mediapipe Face Detection (faster than OpenCV)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Define class names
CLASS_NAMES = {0: "beer", 1: "chocolate"}

# Function to show a pop-up warning
def show_warning(food_type):
    if os.fork() == 0:  # Run warning in a new process
        root = tk.Tk()
        root.withdraw()  # Hide main window
        messagebox.showwarning("Warning!", f"Stop! You are trying to eat {food_type}!")
        root.destroy()
        os._exit(0)

cap = cv2.VideoCapture(0)  # Open webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (Mediapipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces using Mediapipe (much faster than OpenCV)
    face_results = face_detector.process(rgb_frame)

    faces = []
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            fx, fy, fw, fh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            faces.append((fx, fy, fw, fh))

    # Run YOLO inference (much faster with "mps" on Mac)
    results = model.predict(frame, conf=0.5)
    detections = results[0].boxes.data  # Get bounding boxes


    guilty_detected = False
    guilty_foods = []  # Store all detected guilty foods

    # Filter out detections that overlap with a face
    filtered_detections = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)  # Convert to integer

        # Ignore if it overlaps with a face
        is_face = any(
            (x1 > fx and x2 < fx + fw and y1 > fy and y2 < fy + fh)  # Only remove if fully inside face
            for fx, fy, fw, fh in faces
        )

        if not is_face:
            filtered_detections.append([x1, y1, x2, y2, conf, cls])
            guilty_detected = True  # Mark that guilty food is detected
            guilty_foods.append(CLASS_NAMES.get(cls, "unknown"))  # Store detected guilty food

    # Draw detections on the frame (excluding faces)
    for x1, y1, x2, y2, conf, cls in filtered_detections:
        class_name = CLASS_NAMES.get(cls, "unknown")
        color = (255, 0, 0) if cls == 0 else (0, 0, 255)  # Blue for beer, Red for chocolate

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{class_name}: {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show warning if guilty food is detected
    if guilty_detected:
        guilty_food_text = " & ".join(set(guilty_foods))  # Merge multiple detections
        cv2.putText(frame, f" WARNING: {guilty_food_text} Detected! YOU FAT ASS", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        show_warning(guilty_food_text)  # Show warning for detected food

    cv2.imshow("YOLO Detection (Face Filtering Applied)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
