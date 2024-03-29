#!/usr/bin/env bash

set -x
EXP_DIR=exps/vcoco_gen_vlkt_s_r50_dec_3layers

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
        --master_port 29510\
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained params/detr-r50-pre-2branch-vcoco.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 120 \
        --lr_drop 60 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --with_mimic \
        --mimic_loss_coef 10 
        