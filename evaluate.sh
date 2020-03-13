#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python evaluate_surface_normal.py --checkpoint_path './checkpoints/DFPN_TAL_SR.ckpt' \
                                                         --log_folder './log/testing_release/' \
                                                         --batch_size 64 \
                                                         --net_architecture 'stn_fpn' \
                                                         --test_dataset 'kinect_azure_unseen_viewing_directions'