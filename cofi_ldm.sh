#!/bin/bash
#SBATCH --job-name=llada
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --nodelist=n02
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=2

srun python cofi_ldm_no_ds.py \
    --model_id meta-llama/Llama-3.1-8B \
    --dataset ultrachat \
    --train_size 500 \
    --test_size 50 \
    --block_size 16 \
    --epochs 5 \
    --lr 5e-4 \
    --mlp True

