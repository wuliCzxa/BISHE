from ultralytics import YOLO

yolo = YOLO(model='z_weight_k/weights/best.pt', task='detect')
result = yolo(source='mydata_pointer_pose_k/images/train/', save=True)