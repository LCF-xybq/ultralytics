#!/bin/bash


# v8

python plant_height.py --weight ../runs/plant_height_yolov8/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolov8/test_results/labels_raw \
    --save_img ../runs/plant_height_yolov8/test_results/images_raw

sleep 1

python plant_height.py --weight ../runs/plant_height_yolov8/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolov8/test_results/labels_point_correct \
    --save_img ../runs/plant_height_yolov8/test_results/images_point_correct --point_correct

# 11

python plant_height.py --weight ../runs/plant_height_yolo11/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolo11/test_results/labels_raw \
    --save_img ../runs/plant_height_yolo11/test_results/images_raw

sleep 1

python plant_height.py --weight ../runs/plant_height_yolo11/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolo11/test_results/labels_point_correct \
    --save_img ../runs/plant_height_yolo11/test_results/images_point_correct --point_correct

# 26

python plant_height.py --weight ../runs/plant_height_yolo26/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolo26/test_results/labels_raw \
    --save_img ../runs/plant_height_yolo26/test_results/images_raw

sleep 1

python plant_height.py --weight ../runs/plant_height_yolo26/weights/best.pt \
    --data ../../data/plant_height/images/test/ \
    --save_json ../runs/plant_height_yolo26/test_results/labels_point_correct \
    --save_img ../runs/plant_height_yolo26/test_results/images_point_correct --point_correct
