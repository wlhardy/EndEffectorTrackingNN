#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00
#SBATCH --job-name=eepp_train
#SBATCH --output=slurm_output/%x-%j.out

apptainer exec --nv --bind /datasets/eepp:/datasets --bind /etc/pki:/etc/pki ./scripts/train_token_dinov2_EEPP.sif bash \
    -c "python3 train_token_x_y_rot.py"