from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(data = "data.yaml", imgsz=416, batch=8, epochs=100, workers=1, device="cpu", augment=True, cache = True, resume=False)
