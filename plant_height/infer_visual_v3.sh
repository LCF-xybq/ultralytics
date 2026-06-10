#!/bin/bash


# v8
python plant_height_v3.py --weight ../runs/plant_height_yolov8/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolov8/test_results_v3/labels_raw \
    --save_img ../runs/plant_height_yolov8/test_results_v3/images_raw \

sleep 1

python plant_height_v3.py --weight ../runs/plant_height_yolov8/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolov8/test_results_v3/labels_point_correct \
    --save_img ../runs/plant_height_yolov8/test_results_v3/images_point_correct \
    --clean ../runs/plant_height_yolov8/test_results_v3/gradient

# 11
python plant_height_v3.py --weight ../runs/plant_height_yolo11/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolo11/test_results_v3/labels_raw \
    --save_img ../runs/plant_height_yolo11/test_results_v3/images_raw

sleep 1

python plant_height_v3.py --weight ../runs/plant_height_yolo11/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolo11/test_results_v3/labels_point_correct \
    --save_img ../runs/plant_height_yolo11/test_results_v3/images_point_correct \
    --clean ../runs/plant_height_yolo11/test_results_v3/gradient

# 26
python plant_height_v3.py --weight ../runs/plant_height_yolo26/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolo26/test_results_v3/labels_raw \
    --save_img ../runs/plant_height_yolo26/test_results_v3/images_raw

sleep 1

python plant_height_v3.py --weight ../runs/plant_height_yolo26/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolo26/test_results_v3/labels_point_correct \
    --save_img ../runs/plant_height_yolo26/test_results_v3/images_point_correct \
    --clean ../runs/plant_height_yolo26/test_results_v3/gradient
