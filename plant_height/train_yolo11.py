import os.path as osp
from pathlib import Path
from ultralytics import YOLO


current_file_dir = Path(__file__).resolve().parent
pretrain_model = osp.join(current_file_dir, Path("../data/yolo11m-pose.pt"))
data_cfg = osp.join(current_file_dir, Path("../../data/plant_height/plant_height.yaml"))
workspace = osp.join(current_file_dir, Path("../runs"))

# Load a model
model = YOLO(pretrain_model)

# Train the model
results = model.train(
    data=data_cfg, 
    epochs=100, 
    batch=16,
    imgsz=640,
    project=workspace,
    name="plant_height_yolo11",)
