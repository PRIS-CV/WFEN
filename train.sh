# export CUDA_VISIBLE_DEVICES=$1
# =================================================================================
# Train WFEN
# =================================================================================
python train.py --gpus 2 --name wfen --model wfen \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot /path/to/datasets/CelebA_18k --dataset_name celeba --batch_size 32 --total_epochs 150 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500 #--continue_train 

# =================================================================================
# Train WFENHD
# =================================================================================

python train.py --gpus 2 --name wfenhd --model wfenhdhd \
    --Gnorm 'in' --g_lr 0.0001 --d_lr 0.0004 --beta1 0.5 --load_size 512 --total_epochs 10 \
    --Dnorm 'in' --num_D 3 --n_layers_D 4 \
    --dataroot /path/to/datasets/CelebA_18k --dataset_name celeba --batch_size 2 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500 #--continue_train


