#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_surface_normal.py --log_folder './log/training_release/' \
                                                      --operation 'train_robust_acos_loss' \
                                                      --learning_rate 0.0001 \
                                                      --batch_size 16 \
                                                      --net_architecture 'd_fpn_resnext101' \
                                                      --train_dataset 'scannet_standard' \
                                                      --test_dataset 'scannet_standard' \
                                                      --test_dataset 'nyud' \
                                                      --val_dataset 'scannet_standard'