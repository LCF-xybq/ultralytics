#!/bin/bash

img_name="DJI_20250818090519_0023_V(1).jpeg"
cut_off=0.1

python high_pass.py \
    --input "/home/lcf/projects/yolo/data/plant_height/images/test/${img_name}" \
    --output "./${img_name}_result.jpeg" \
    --cutoff "${cut_off}"
