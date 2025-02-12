from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

# Open the default webcam (0 is usually the built-in camera on MacBooks)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Perform inference on the frame (YOLO detection)
    results = model.predict(frame)

    # Get the annotated frame with bounding boxes
    annotated_frame = results[0].plot()

    # Display the resulting frame with YOLO detections
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# model.predict(source="", show=True, verbose=True)

