from ultralytics import YOLO

yolo = YOLO(model='weight/weights/best.pt', task='detect')
result = yolo(source='pressure_pointer_2yolo/images/train/', save=True)
