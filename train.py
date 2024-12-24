from ultralytics import YOLO

# Load a model
model = YOLO("/mnt/d/Study/codes/home_codes/road_seg/cfg/yolov8s-seg.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="/mnt/d/Study/codes/home_codes/road_seg/data/yolo/sun_sight/data.yaml", epochs=30, imgsz=1280, batch=8)