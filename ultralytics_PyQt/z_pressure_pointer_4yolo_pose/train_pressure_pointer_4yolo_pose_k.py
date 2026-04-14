from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

model.train(data='pressure_pointer_4yolo_pose_k.yaml', workers=0, epochs=200, batch=8)
