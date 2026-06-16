import os.path as osp
from pathlib import Path
from ultralytics import YOLO


current_file_dir = Path(__file__).resolve().parent
data_cfg = osp.join(current_file_dir, Path("../../data/plant_height/plant_height.yaml"))
workspace = osp.join(current_file_dir, Path("../../runs_ablations"))

model = YOLO("full_11.yaml")

results = model.train(
    data=data_cfg,
    epochs=100,
    imgsz=640,
    batch=16,
    name="full_11",
    lr0=0.01,
    lrf=0.01,
    project=workspace,
    height_mae=True,
)
