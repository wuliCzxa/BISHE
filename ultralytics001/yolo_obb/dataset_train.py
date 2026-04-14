from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# model.train(data='yaml/dataset_1biaopan_all.yaml', workers=0, epochs=200, batch=8)
# model.train(data='yaml/dataset_2biaopan_nolabel.yaml', workers=0, epochs=200, batch=8)
model.train(data='yaml/dataset_3biaopan_label.yaml', workers=0, epochs=200, batch=8)
