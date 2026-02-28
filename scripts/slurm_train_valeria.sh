#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu_96h
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=eepp_train
#SBATCH --mail-user=william.larrivee-hardy@norlab.ulaval.ca
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH --account=def-phgig4
#SBATCH --reservation=c7_10228

# Create a temporary file to store the key securely
temp_file=$(mktemp)

# Ensure the temporary file is deleted when the script exits
trap 'rm -f "$temp_file"' EXIT

# Write the key to the temporary file
echo "$WANDB_API_KEY" > "$temp_file"

module load apptainer
module load httpproxy

export WANDB_MODE=online

apptainer exec --nv --bind /home/ulaval.ca/wilah/projects/def-phgig4/eepp:/datasets --bind /etc/pki:/etc/pki --bind "$temp_file:/run/wandb_api_key.txt" ./scripts/train_token_dinov3_EEPP.sif bash \
    -c "python3 train_token_x_y_rot_dinov3_reg.py --sweep norlab-ulaval/EndEffectorPosePred/c44wc8sr"