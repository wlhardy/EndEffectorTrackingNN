#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00
#SBATCH --job-name=eepp_train_single_token
#SBATCH --output=slurm_output/%x-%j.out

# Create a temporary file to store the key securely
temp_file=$(mktemp)

# Ensure the temporary file is deleted when the script exits
trap 'rm -f "$temp_file"' EXIT

# Write the key to the temporary file
echo "$WANDB_API_KEY" > "$temp_file"

apptainer exec --nv --bind /home/wilah/workspace/EndEffectorTrackingNN/datasets:/datasets --bind /etc/pki:/etc/pki --bind "$temp_file:/run/wandb_api_key.txt" ./scripts/train_token_dinov3_EEPP.sif bash \
    -c "python3 train_token_x_y_rot_dinov3_single_token_reg.py --sweep norlab-ulaval/EndEffectorPosePred/arci6j0z"
