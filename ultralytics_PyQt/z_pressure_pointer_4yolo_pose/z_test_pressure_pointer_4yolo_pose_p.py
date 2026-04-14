from ultralytics import YOLO

yolo = YOLO(model='z_weight_p/weights/best.pt', task='detect')
result = yolo(source='mydata_pointer_pose_p/images/train/', save=True)
