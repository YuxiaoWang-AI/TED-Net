#!/usr/bin/env bash

set -x
EXP_DIR=exps/hico_gen_vlkt_s_r50_dec_3layers

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
        --master_port 29510\
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained path/to/best_model.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
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
        --mimic_loss_coef 30 \
        --dir_name TED-Net \
        --eval
        