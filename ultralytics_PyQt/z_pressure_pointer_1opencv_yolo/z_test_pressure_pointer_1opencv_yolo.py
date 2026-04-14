from ultralytics import YOLO

yolo = YOLO(model='weight/weights/best.pt', task='detect')
result = yolo(source='pressure_pointer_1opencv_yolo/images/train/', save=True)
