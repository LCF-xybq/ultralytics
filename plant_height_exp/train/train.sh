#!/bin/bash


python train_v1.py

sleep 10

python train_yolov8.py

sleep 10

python train_yolo11.py

sleep 10

python train_yolo26.py