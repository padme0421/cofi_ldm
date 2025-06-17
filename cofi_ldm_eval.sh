#!/bin/bash
#SBATCH --job-name=llada_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --nodelist=master
#SBATCH --time=1-23:59:59
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=2

srun python cofi_ldm_eval.py \
    --metric bleu \
    --gen_path ar_result_truthfulqa_train0_test0_gen32.jsonl
