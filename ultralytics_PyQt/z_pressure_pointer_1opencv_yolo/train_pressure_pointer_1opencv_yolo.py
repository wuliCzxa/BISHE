from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(data='pressure_pointer_1opencv_yolo.yaml', workers=0, epochs=200, batch=8)
