#!/bin/bash
#SBATCH --job-name=llada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n01
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

srun python train_mlp2.py