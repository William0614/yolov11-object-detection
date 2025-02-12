from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(data = "data.yaml", imgsz=380, batch=8, epochs=100, workers=1, device="cpu")
