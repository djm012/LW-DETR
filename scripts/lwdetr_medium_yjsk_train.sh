model_name='lwdetr_medium_yjsk'
yjsk_path='./yjsk'
# sh scripts/lwdetr_medium_yjsk_train.sh 
python -u -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    main.py \
    --lr 1e-4 \
    --lr_encoder 1.5e-4 \
    --batch_size 2 \
    --weight_decay 1e-4 \
    --epochs 100 \
    --lr_drop 60 \
    --lr_vit_layer_decay 0.7 \
    --lr_component_decay 0.7 \
    --encoder vit_small \
    --vit_encoder_num_layers 10 \
    --window_block_indexes 0 1 3 6 7 9 \
    --out_feature_indexes 2 4 5 9 \
    --dec_layers 3 \
    --group_detr 13 \
    --two_stage \
    --projector_scale P4 \
    --hidden_dim 256 \
    --sa_nheads 8 \
    --ca_nheads 16 \
    --dec_n_points 2 \
    --bbox_reparam \
    --lite_refpoint_refine \
    --ia_bce_loss \
    --cls_loss_coef 1 \
    --num_select 300 \
    --dataset_file yjsk \
    --yjsk_path $yjsk_path \
    --square_resize_div_64 \
    --use_ema \
    --pretrained_encoder pretrain_weights/caev2_small_300e_objects365.pth \
    --pretrain_weights pretrain_weights/LWDETR_medium_60e_coco.pth \
    --output_dir output/$model_name

    # --pretrain_weights pretrain_weights/LWDETR_medium_30e_objects365.pth \
