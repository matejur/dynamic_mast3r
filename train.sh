#!/bin/bash
torchrun --nproc_per_node=4 train.py \
    --train_dataset "20_000 @ PointOdyssey(root='<path_to_point_odyssey>', split='train', resolution=[(384, 256), (256, 256), (256, 192), (256, 176), (256, 128)], maximum_stride=8, aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=1024, clip_step=2) \
                    + 20_000 @ Kubric(root='<path_to_kubric>', split='train', resolution=[(384, 256), (256, 256), (256, 192), (256, 176), (256, 128)], maximum_stride=8, aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=1024)" \
    --test_dataset "2500 @ PointOdyssey(root='<path_to_point_odyssey>', split='test', resolution=(256, 256), maximum_stride=8, aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=1024) \
                    + 2500 @ Kubric(root='<path_to_kubric>', split='validation', resolution=(256, 256), maximum_stride=8, aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=1024)"  \
    --model "AsymmetricMASt3R(freeze='encoder', pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(256,256), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --pretrained "<path_to_pretrained_model>" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 2 --epochs 50 --batch_size 2 --accum_iter 4 \
    --save_freq 5 --keep_freq 5 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "checkpoints/Dynamic_MASt3R"