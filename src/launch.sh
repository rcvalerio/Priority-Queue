#!/bin/bash
#SBATCH --job-name=pq
#SBATCH --output=logs/%j-%x.out -e logs/%j-%x.err
#SBATCH -n 2 # Number of cpus
#SBATCH --mem=32768 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit
#SBATCH --gres=gpu:3g.20gb:1

#['Inshop', 'cub', 'cars', 'SOP']
#['googlenet', 'bn_inception', 'resnet18', 'resnet50', 'resnet101']
#['MS', 'Contrastive', 'Triplet']
#['sgd', 'adam', 'rmsprop', 'adamw']
#['reduce', 'step']

eval "$(conda shell.bash hook)"
conda activate pq
python train.py --job-id $SLURM_JOB_ID \
                --gpu-id 0 \
                --epochs 60 \
                --loss Contrastive \
                --model resnet50 \
                --embedding-size 128 \
                --batch-size 64 \
                --lr 1e-4 \
                --dataset SOP \
                --pq-size 8192 \
                --pq-warmup 0 \
                --pq-scheduler reduce \
                --grad-acc 1 \
                --IPC 4 \
                --optimizer adam \
                --workers 2 \
                --warm 0 \
                --bn-freeze 0 \
                --weight-decay 5e-4 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25 \
                --l2-norm 1 \
                --contrastive-margin 0.43 \
                --triplet-margin 0.15 \
                --ms-pos-scale 2.0 \
                --ms-neg-scale 40.0 \
                --ms-margin 0.1 \
                --pq-type update \
                --pq-rm-old 1 \
                --pq-rm-loss 1 \
                --pq-upd-int 1 \
