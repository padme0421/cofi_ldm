#!/bin/bash
#SBATCH --job-name=llada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=master
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

srun python ar_baseline.py \
    --model_id meta-llama/Llama-3.1-8B \
    --dataset truthfulqa \
    --train_size 0 \
    --test_size 0 \
    --gen_length 32 \
