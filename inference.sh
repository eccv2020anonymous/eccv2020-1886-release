#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python inference_surface_normal.py --checkpoint_path './checkpoints/DFPN_TAL_SR.ckpt' \
                                                           --log_folder './demo_results' \
                                                           --batch_size 8 \
                                                           --net_architecture 'stn_fpn' \
                                                           --test_dataset './demo_dataset'