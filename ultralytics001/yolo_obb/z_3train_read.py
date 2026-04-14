from ultralytics import YOLO


model = YOLO('yolo11-obb.yaml').load('yolo11n-obb.pt')
model.train(data='yaml/dataset_4read.yaml', epochs=200, batch=8, workers=0)

