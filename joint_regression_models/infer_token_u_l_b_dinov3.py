import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os
import argparse
from tqdm import tqdm
import csv
import heapq

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import eefdataset
import model_token_dinov3_reg_u_l_b

# Reuse the exact helper functions / conventions used in your regression training script
from train_token_u_l_b_dinov3_reg import (
    save_debug_image,
    half_pixels_resize_and_pad,
    discover_dataset_folders,
    normalize_2d,
    ang_error_deg_period360,
)

DINOV3_REPO_DIR = "dinov3"


@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Load checkpoint ===
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # === Load model (REGRESSION) ===
    backbone_model = torch.hub.load(
        DINOV3_REPO_DIR,
        "dinov3_vitb16",
        source="local",
        weights="dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        force_reload=False,
    )
    ee_model = model_token_dinov3_reg_u_l_b.EndEffectorPosePredToken(backbone_model).to(device)
    ee_model.load_state_dict(ckpt["model_state_dict"])
    ee_model.eval()

    # === Dataset (same preprocessing as validation in training) ===
    transform_val = T.Compose([
        T.Lambda(lambda img: TF.rotate(img, 180)),
        T.Lambda(
            lambda img: TF.crop(
                img,
                top=args.top_crop,
                left=args.left_crop,
                height=img.height - args.bottom_crop,
                width=img.width - args.right_crop,
            )
        ),
        T.Lambda(half_pixels_resize_and_pad),
        T.ToTensor(),
    ])

    image_dirs, joint_csvs, xy_csvs = discover_dataset_folders(args.dataset)
    dataset = eefdataset.EEFDataset(
        image_dirs=image_dirs,
        joint_csv_paths=joint_csvs,
        xy_csv_paths=xy_csvs,
        joint_precision=args.precision,
        xy_bin_nbr=args.xy_bin_nbr,
        transform=transform_val,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # === Output folders ===
    csv_path = Path(args.output_dir) / "results.csv"

    # === CSV header (REGRESSION) ===
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "base_joint_gt_deg",
                "base_joint_pred_deg",
                "lower_joint_gt_deg",
                "lower_joint_pred_deg",
                "upper_joint_gt_deg",
                "upper_joint_pred_deg",
            ]
        )

    print("Running inference (regression) ...")
    for batch_i, (images, joint_values) in enumerate(tqdm(dataloader)):
        images = images.to(device)

        # --- GT ---
        base_theta_deg = joint_values['base_joint'].to(device).to(torch.float32)
        base_theta_rad = torch.deg2rad(base_theta_deg)
        base_joint_sin_gt = torch.sin(base_theta_rad)
        base_joint_cos_gt = torch.cos(base_theta_rad)
        base_joint_gt = torch.stack([base_joint_sin_gt, base_joint_cos_gt], dim=-1)
        
        lower_theta_deg = joint_values['lower_joint'].to(device).to(torch.float32)
        lower_theta_rad = torch.deg2rad(lower_theta_deg)
        lower_joint_sin_gt = torch.sin(lower_theta_rad)
        lower_joint_cos_gt = torch.cos(lower_theta_rad)
        lower_joint_gt = torch.stack([lower_joint_sin_gt, lower_joint_cos_gt], dim=-1)

        upper_theta_deg = joint_values['upper_joint'].to(device).to(torch.float32)
        upper_theta_rad = torch.deg2rad(upper_theta_deg)
        upper_joint_sin_gt = torch.sin(upper_theta_rad)
        upper_joint_cos_gt = torch.cos(upper_theta_rad)
        upper_joint_gt = torch.stack([upper_joint_sin_gt, upper_joint_cos_gt], dim=-1)
        
        base_joint_gt = base_joint_gt.to(torch.float32)
        lower_joint_gt = lower_joint_gt.to(torch.float32)
        upper_joint_gt = upper_joint_gt.to(torch.float32)

        # --- Pred ---
        base_joint_sincos, lower_joint_sincos, upper_joint_sincos = ee_model(images)

        base_joint_sincos = normalize_2d(base_joint_sincos)
        lower_joint_sincos = normalize_2d(lower_joint_sincos)
        upper_joint_sincos = normalize_2d(upper_joint_sincos)

        theta_base_pred_rad = torch.atan2(base_joint_sincos[:, 0], base_joint_sincos[:, 1])
        theta_lower_pred_rad = torch.atan2(lower_joint_sincos[:, 0], lower_joint_sincos[:, 1])
        theta_upper_pred_rad = torch.atan2(upper_joint_sincos[:, 0], upper_joint_sincos[:, 1])

        theta_base_pred_deg = torch.rad2deg(theta_base_pred_rad)
        theta_lower_pred_deg = torch.rad2deg(theta_lower_pred_rad)
        theta_upper_pred_deg = torch.rad2deg(theta_upper_pred_rad)

        # === Stream results to CSV ===
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for j in range(images.size(0)):
                global_idx = batch_i * args.batch_size + j

                writer.writerow(
                    [
                        global_idx,
                        base_theta_deg[j].item(),
                        theta_base_pred_deg[j].item(),
                        lower_theta_deg[j].item(),
                        theta_lower_pred_deg[j].item(),
                        upper_theta_deg[j].item(),
                        theta_upper_pred_deg[j].item(),
                    ]
                )

        # Free batch memory
        del images, base_joint_sincos, lower_joint_sincos, upper_joint_sincos
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference (regression) for EndEffectorPosePredToken (DINOv3)")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (.pt)", default="/home/wilah/workspace/EndEffectorTrackingNN/training/checkpoint_20260215_123512/model_checkpoint.pt")
    parser.add_argument("--dataset", type=str, help="Path to dataset folder (same structure as training)", default="/home/wilah/datasets/heshan_october_grapple_data")
    parser.add_argument("--output_dir", type=str, help="Output folder for results", default="results_inference_with_data_aug_reg")
    parser.add_argument("--precision", type=float, default=1.0, help="Ground truth precision used in dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--top_n", type=int, default=30, help="Number of worst/best predictions to save")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--top_crop", type=int, default=1)
    parser.add_argument("--bottom_crop", type=int, default=2)
    parser.add_argument("--left_crop", type=int, default=398)
    parser.add_argument("--right_crop", type=int, default=856)
    parser.add_argument("--xy_bin_nbr", type=int, default=100, help="Number of bins for x and y position")
    args = parser.parse_args()

    run_inference(args)
