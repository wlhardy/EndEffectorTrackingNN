#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00
#SBATCH --job-name=eepp_train
#SBATCH --output=%x-%j.out

apptainer instance run --nv --bind /data.wilah_dataset:./datasets/ ./scripts/train_token_dinov2_EEPP.sif eepp_training