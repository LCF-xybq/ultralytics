#!/bin/bash


# v8
python metrics.py \
    --input ../runs/plant_height_yolov8/test_results_v2/labels_raw \
    --label ../../data/plant_height/labels/test_json \
    --save ../runs/plant_height_yolov8/test_results_v2/raw_yolo.csv

python metrics.py \
    --input ../runs/plant_height_yolov8/test_results_v2/labels_point_correct \
    --label ../../data/plant_height/labels/test_json \
    --save ../runs/plant_height_yolov8/test_results_v2/point_correct_yolo.csv

# 11
python metrics.py \
    --input ../runs/plant_height_yolo11/test_results_v2/labels_raw \
    --label ../../data/plant_height/labels/test_json \
    --save ../runs/plant_height_yolo11/test_results_v2/raw_yolo.csv

python metrics.py \
    --input ../runs/plant_height_yolo11/test_results_v2/labels_point_correct \
    --label ../../data/plant_height/labels/test_json \
    --save ../runs/plant_height_yolo11/test_results_v2/point_correct_yolo.csv

# 26
python metrics.py \
    --input ../runs/plant_height_yolo26/test_results_v2/labels_raw \
    --label ../../data/plant_height/labels/test_json \
    --save ../runs/plant_height_yolo26/test_results_v2/raw_yolo.csv

python metrics.py \
    --input ../runs/plant_height_yolo26/test_results_v2/labels_point_correct \
    --label ../../data/plant_height/labels/test_json \
    --save ../runs/plant_height_yolo26/test_results_v2/point_correct_yolo.csv
