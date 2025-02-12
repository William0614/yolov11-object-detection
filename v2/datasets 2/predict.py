from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source="3.jpg", show=True, verbose=True)


