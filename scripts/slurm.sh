#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00
#SBATCH --job-name=eepp_train
#SBATCH --output=slurm_output/%x-%j.out

temp_file=$(mktemp)
echo "$WANDB_API_KEY" > "$temp_file"

apptainer exec --nv --bind /datasets/eepp:/datasets --bind /etc/pki:/etc/pki --bind "$temp_file:/run/wandb_api_key.txt" ./scripts/train_token_dinov2_EEPP.sif bash \
    -c "python3 train_token_x_y_rot.py"