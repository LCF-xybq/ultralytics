#!/bin/bash


# plant v1
# python metrics.py \
#     --input ../../runs_none_pretrain/v1/test_results_v3/labels_raw \
#     --label ../../../data/plant_height/labels/test_json \
#     --save ../../runs_none_pretrain/v1/test_results_v3/raw_yolo.csv

# python metrics.py \
#     --input ../../runs_none_pretrain/plant_height_yolov8/test_results_v3/labels_point_correct \
#     --label ../../../data/plant_height/labels/test_json \
#     --save ../../runs_none_pretrain/plant_height_yolov8/test_results_v3/point_correct_yolo.csv

# v8
python metrics.py \
    --input ../../runs_none_pretrain/v1_hpmodule_v8_100e/test_results_v3/labels_raw \
    --label ../../../data/plant_height/labels/test_json \
    --save ../../runs_none_pretrain/v1_hpmodule_v8_100e/test_results_v3/raw_yolo.csv

python metrics.py \
    --input ../../runs_none_pretrain/v1_hpmodule_v8_100e/test_results_v3/labels_point_correct \
    --label ../../../data/plant_height/labels/test_json \
    --save ../../runs_none_pretrain/v1_hpmodule_v8_100e/test_results_v3/point_correct_yolo.csv

# 11
python metrics.py \
    --input ../../runs_none_pretrain/v1_hpmodule_11_100e/test_results_v3/labels_raw \
    --label ../../../data/plant_height/labels/test_json \
    --save ../../runs_none_pretrain/v1_hpmodule_11_100e/test_results_v3/raw_yolo.csv

python metrics.py \
    --input ../../runs_none_pretrain/v1_hpmodule_11_100e/test_results_v3/labels_point_correct \
    --label ../../../data/plant_height/labels/test_json \
    --save ../../runs_none_pretrain/v1_hpmodule_11_100e/test_results_v3/point_correct_yolo.csv

# 26
python metrics.py \
    --input ../../runs_none_pretrain/plant_height_yolo26/test_results_v3/labels_raw \
    --label ../../../data/plant_height/labels/test_json \
    --save ../../runs_none_pretrain/plant_height_yolo26/test_results_v3/raw_yolo.csv

python metrics.py \
    --input ../../runs_none_pretrain/plant_height_yolo26/test_results_v3/labels_point_correct \
    --label ../../../data/plant_height/labels/test_json \
    --save ../../runs_none_pretrain/plant_height_yolo26/test_results_v3/point_correct_yolo.csv
