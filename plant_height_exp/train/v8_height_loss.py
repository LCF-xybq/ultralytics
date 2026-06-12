import os.path as osp
from pathlib import Path
from ultralytics import YOLO


current_file_dir = Path(__file__).resolve().parent
data_cfg = osp.join(current_file_dir, Path("../../../data/plant_height/plant_height.yaml"))
workspace = osp.join(current_file_dir, Path("../../runs_none_pretrain"))

model = YOLO(osp.join(current_file_dir, "yolov8_m_height_loss.yaml"))

results = model.train(
    data=data_cfg,
    epochs=400,
    imgsz=640,
    batch=16,
    name="yolov8_height_loss",
    lr0=0.01,
    lrf=0.01,
    project=workspace,
    height=2.0,
    height_warmup_epochs=50,
    height_ramp_epochs=30,
)
