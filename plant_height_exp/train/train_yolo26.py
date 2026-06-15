import os.path as osp
from pathlib import Path
from ultralytics import YOLO


current_file_dir = Path(__file__).resolve().parent
pretrain_model = osp.join(current_file_dir, Path("../../data/yolo26m-pose.pt"))
data_cfg = osp.join(current_file_dir, Path("../../../data/plant_height/plant_height.yaml"))
workspace = osp.join(current_file_dir, Path("../../runs_none_pretrain"))

# Load a model
model = YOLO(pretrain_model)

results = model.train(
    data=data_cfg,
    epochs=40,
    imgsz=640,
    batch=16,
    name="plant_height_yolo26",
    project=workspace,
)
