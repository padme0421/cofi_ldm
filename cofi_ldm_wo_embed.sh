#!/bin/bash
#SBATCH --job-name=llada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n02
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=1

srun python cofi_ldm_wo_embed.py

