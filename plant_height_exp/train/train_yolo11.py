import os.path as osp
from pathlib import Path
from ultralytics import YOLO


current_file_dir = Path(__file__).resolve().parent
data_cfg = osp.join(current_file_dir, Path("../../../data/plant_height/plant_height.yaml"))
workspace = osp.join(current_file_dir, Path("../../runs_none_pretrain"))

model = YOLO("yolo11_m.yaml")

results = model.train(
    data=data_cfg,
    epochs=50,
    imgsz=640,
    batch=16,
    name="plant_height_yolo11",
    project=workspace,
)
