from ultralytics import YOLO

yolo = YOLO(model='weight/4read/weights/best.pt', task='detect')
result = yolo(source='datasets/pressure_4read/images/train/', save=True)
