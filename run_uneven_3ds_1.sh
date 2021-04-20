for i in 10 30 80
do
    python train_uneven.py \
        --outdir=/mnt/hdd/repo_results/stylegan2-ada-pytorch/uneven_linear_3ds \
        --aug=noaug \
        --data=/mnt/hdd/Datasets/disentangle_datasets/3dshapes.zip \
        --cfg=3dshapes \
        --kimg=8000 \
        --gpus=2 \
        --g_reg_interval=0 \
        --gamma=${i} \
        --z_dim=10 \
        --n_samples_per 10 \
        --w1reg_lambda=1 \
        --uneven_reg_type=linear \
        --uneven_reg_maxval=10
done
