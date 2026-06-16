import os.path as osp
from pathlib import Path
from ultralytics import YOLO


current_file_dir = Path(__file__).resolve().parent
data_cfg = osp.join(current_file_dir, Path("../../data/plant_height/plant_height.yaml"))
workspace = osp.join(current_file_dir, Path("../../runs_none_pretrain"))

model = YOLO("v1_hpmodule_colorcontrast_11_v2.yaml")

results = model.train(
    data=data_cfg,
    epochs=200,
    imgsz=640,
    batch=16,
    name="v1_hpmodule_colorcontrast_11_v2",
    lr0=0.01,
    lrf=0.01,
    project=workspace,
    height_mae=True,
)
