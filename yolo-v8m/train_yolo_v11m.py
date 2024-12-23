from ultralytics import YOLO

model = YOLO('yolov10m.pt')

model.train(data="../yjsk_old/custom.yaml", epochs=100)
