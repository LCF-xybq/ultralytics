#!/bin/bash


# plant v1
# python plant_height_v3.py --weight ../../runs_none_pretrain/v1/weights/best.pt \
#     --data ../../../data/plant_height/images/test/ \
#     --save_json ../../runs_none_pretrain/v1/test_results_v3/labels_raw \
#     --save_img ../../runs_none_pretrain/v1/test_results_v3/images_raw \

# python plant_height_v3.py --weight ../../runs_none_pretrain/v1/weights/best.pt \
#     --data ../../../data/plant_height/images/test/ \
#     --save_json ../../runs_none_pretrain/v1/test_results_v3/labels_point_correct \
#     --save_img ../../runs_none_pretrain/v1/test_results_v3/images_point_correct \  
#     --clean ../../runs_none_pretrain/v1/test_results_v3/gradient  

# v8
python plant_height_v3.py --weight ../../runs_none_pretrain/yolov8_height_loss/weights/best.pt \
    --data ../../../data/plant_height/images/test/ \
    --save_json ../../runs_none_pretrain/yolov8_height_loss/test_results_v3/labels_raw \
    --save_img ../../runs_none_pretrain/yolov8_height_loss/test_results_v3/images_raw

sleep 1

python plant_height_v3.py --weight ../../runs_none_pretrain/yolov8_height_loss/weights/best.pt \
    --data ../../../data/plant_height/images/test/ \
    --save_json ../../runs_none_pretrain/yolov8_height_loss/test_results_v3/labels_point_correct \
    --save_img ../../runs_none_pretrain/yolov8_height_loss/test_results_v3/images_point_correct \
    --clean ../../runs_none_pretrain/yolov8_height_loss/test_results_v3/gradient

# 11
python plant_height_v3.py --weight ../../runs_none_pretrain/v1_hpmodule_11_100e/weights/best.pt \
    --data ../../../data/plant_height/images/test/ \
    --save_json ../../runs_none_pretrain/v1_hpmodule_11_100e/test_results_v3/labels_raw \
    --save_img ../../runs_none_pretrain/v1_hpmodule_11_100e/test_results_v3/images_raw

sleep 1

python plant_height_v3.py --weight ../../runs_none_pretrain/v1_hpmodule_11_100e/weights/best.pt \
    --data ../../../data/plant_height/images/test/ \
    --save_json ../../runs_none_pretrain/v1_hpmodule_11_100e/test_results_v3/labels_point_correct \
    --save_img ../../runs_none_pretrain/v1_hpmodule_11_100e/test_results_v3/images_point_correct \
    --clean ../../runs_none_pretrain/v1_hpmodule_11_100e/test_results_v3/gradient

# 26
python plant_height_v3.py --weight ../../runs_none_pretrain/plant_height_yolo26/weights/best.pt \
    --data ../../../data/plant_height/images/test/ \
    --save_json ../../runs_none_pretrain/plant_height_yolo26/test_results_v3/labels_raw \
    --save_img ../../runs_none_pretrain/plant_height_yolo26/test_results_v3/images_raw

sleep 1

python plant_height_v3.py --weight ../../runs_none_pretrain/plant_height_yolo26/weights/best.pt \
    --data ../../../data/plant_height/images/test/ \
    --save_json ../../runs_none_pretrain/plant_height_yolo26/test_results_v3/labels_point_correct \
    --save_img ../../runs_none_pretrain/plant_height_yolo26/test_results_v3/images_point_correct \
    --clean ../../runs_none_pretrain/plant_height_yolo26/test_results_v3/gradient
