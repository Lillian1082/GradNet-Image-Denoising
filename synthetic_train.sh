#!/usr/bin/env bash

out_file_name="logs_synthetic/nyu_resnet_sk_S50_vanila.out" # MSKResnet
CUDA_VISIBLE_DEVICES=1 python3 train.py --archi Resnet_sk --batchSize 64 --device_ids 1 --lr 1e-3 --cropSize 80 --outf logs_synthetic/nyu_resnet_sk_S50_vanila --epochs 400 --milestone 20 --noiseLevel 50 --grad_weight 0 --train_dataPath data/nyuv2/train --test_dataPath data/nyuv2/val --randomCount 4 > $out_file_name 2>&1 &

out_file_name="logs_synthetic/nyu_resnet_sk_conGradient_before_S50_MSKResnet_1.out" # GradNet(MSKResnet) + Gradient Loss
CUDA_VISIBLE_DEVICES=0 python3 train.py --archi Resnet_sk_conGradient_before --batchSize 48 --device_ids 1 --lr 1e-3 --cropSize 80 --outf logs_synthetic/nyu_resnet_sk_conGradient_before_S50_MSKResnet_1 --epochs 400 --milestone 20 --noiseLevel 50 --grad_weight 0.1 --train_dataPath data/nyuv2/train --test_dataPath data/nyuv2/val --randomCount 4 --pretrained_path logs_synthetic/nyu_resnet_sk_S50_vanila --pretrained_model checkpoint/net_394.pth --pretrained_archi Resnet_sk > $out_file_name 2>&1 & 
