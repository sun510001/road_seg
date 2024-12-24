from ultralytics import YOLO
import pickle

model = YOLO('/mnt/d/Study/codes/home_codes/road_seg/runs/segment/train_imgsz1280/weights/best.pt')
results = model.predict(["/mnt/d/Study/codes/home_codes/road_seg/data/yolo/sun_sight/test/images/SunSight1017.jpg", "/mnt/d/Study/codes/home_codes/road_seg/data/yolo/sun_sight/test/images/SunSight1071.jpg"], embed=[-1])

with open("pred.pkl", "wb") as f:
   pickle.dump(results, f)