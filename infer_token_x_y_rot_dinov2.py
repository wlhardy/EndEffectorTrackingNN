import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torchvision.transforms.v2 import InterpolationMode
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import argparse
from tqdm import tqdm
import csv
import heapq

import eefdataset
import model_token_dinov2

# Reuse exact helpers from training
from train_token_x_y_rot_dinov2 import (
    save_debug_image,
    ang_error_deg_period180,
    discover_dataset_folders,
    normalize_2d,
    half_pixels_resize_and_pad,
)

@torch.no_grad()
def run_inference(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Load checkpoint ===
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # === Load DINOv2 backbone ===
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        ckpt["dinov2_model"],
        force_reload=False,
    )

    ee_model = model_token_dinov2.EndEffectorPosePredToken(backbone).to(device)
    ee_model.load_state_dict(ckpt["model_state_dict"])
    ee_model.eval()

    # === Dataset (same preprocessing as validation in training) ===
    transform_val = T.Compose([
        T.Lambda(lambda img: TF.rotate(img, 180)),
        T.Lambda(lambda img: TF.crop(
            img,
            top=args.top_crop,
            left=args.left_crop,
            height=img.height - args.bottom_crop,
            width=img.width - args.right_crop,
        )),
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
    os.makedirs(args.output_dir, exist_ok=True)
    worst_dir = Path(args.output_dir) / "worst_predictions"
    best_dir = Path(args.output_dir) / "best_predictions"
    worst_dir.mkdir(exist_ok=True, parents=True)
    best_dir.mkdir(exist_ok=True, parents=True)
    csv_path = Path(args.output_dir) / "results.csv"

    # === CSV header (REGRESSION) ===
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx",
            "angular_error_deg",
            "x_error_bins",
            "y_error_bins",
            "total_error",
            "gt_base_joint_deg",
            "pred_base_joint_deg",
            "gt_x_bin",
            "pred_x_bin_float",
            "gt_y_bin",
            "pred_y_bin_float",
            "pred_sin2theta",
            "pred_cos2theta",
        ])

    worst_heap = []
    best_heap = []
    heapq.heapify(worst_heap)
    heapq.heapify(best_heap)

    print("Running inference (DINOv2 regression)...")

    for batch_i, (images, joint_values) in enumerate(tqdm(dataloader)):

        images = images.to(device)

        gt_theta_deg = joint_values["base_joint"].to(device).float()
        gt_x_bin = joint_values["x"].to(device).float()
        gt_y_bin = joint_values["y"].to(device).float()

        # Normalize GT same as training
        gt_x_norm = gt_x_bin / float(args.xy_bin_nbr)
        gt_y_norm = gt_y_bin / float(args.xy_bin_nbr)

        # === Forward ===
        pred_sincos, pred_x_norm, pred_y_norm = ee_model(images)

        # Ensure explicit batch dimension (fixes 0-d tensor issues)
        pred_sincos = pred_sincos.view(-1, 2)
        pred_x_norm = pred_x_norm.view(-1)
        pred_y_norm = pred_y_norm.view(-1)
        gt_theta_deg = gt_theta_deg.view(-1)
        gt_x_bin = gt_x_bin.view(-1)
        gt_y_bin = gt_y_bin.view(-1)

        # Normalize sin/cos vector
        pred_sincos = normalize_2d(pred_sincos)

        # === Decode angle (sin(2θ), cos(2θ)) ===
        phi = torch.atan2(pred_sincos[:, 0], pred_sincos[:, 1])
        theta_pred_rad = 0.5 * phi
        theta_pred_rad = torch.remainder(theta_pred_rad, torch.pi)
        theta_pred_deg = torch.rad2deg(theta_pred_rad)

        angular_error = ang_error_deg_period180(
            theta_pred_deg,
            gt_theta_deg
        )

        # Convert back to bins for interpretability
        pred_x_bin = pred_x_norm * float(args.xy_bin_nbr)
        pred_y_bin = pred_y_norm * float(args.xy_bin_nbr)

        x_error_bins = torch.abs(pred_x_bin - gt_x_bin)
        y_error_bins = torch.abs(pred_y_bin - gt_y_bin)

        total_error = torch.abs(angular_error) + x_error_bins + y_error_bins

        # === Stream results ===
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            for j in range(images.size(0)):
                global_idx = batch_i * args.batch_size + j
                err_val = total_error[j].item()

                writer.writerow([
                    global_idx,
                    angular_error[j].item(),
                    x_error_bins[j].item(),
                    y_error_bins[j].item(),
                    err_val,
                    gt_theta_deg[j].item(),
                    theta_pred_deg[j].item(),
                    gt_x_bin[j].item(),
                    pred_x_bin[j].item(),
                    gt_y_bin[j].item(),
                    pred_y_bin[j].item(),
                    pred_sincos[j, 0].item(),
                    pred_sincos[j, 1].item(),
                ])

                # Worst N
                if len(worst_heap) < args.top_n:
                    heapq.heappush(worst_heap, (err_val, global_idx))
                elif err_val > worst_heap[0][0]:
                    heapq.heapreplace(worst_heap, (err_val, global_idx))

                # Best N
                if len(best_heap) < args.top_n:
                    heapq.heappush(best_heap, (-err_val, global_idx))
                elif err_val < -best_heap[0][0]:
                    heapq.heapreplace(best_heap, (-err_val, global_idx))

        del images
        torch.cuda.empty_cache()

    # === Save worst ===
    worst_heap = sorted(worst_heap, key=lambda x: x[0], reverse=True)

    print(f"Saving {len(worst_heap)} worst predictions...")
    for rank, (err_val, sample_idx) in enumerate(worst_heap):

        img, gt = dataset[sample_idx]
        img_tensor = img.to(device).unsqueeze(0)

        pred_sincos, pred_x_norm, pred_y_norm = ee_model(img_tensor)
        pred_sincos = normalize_2d(pred_sincos)

        phi = torch.atan2(pred_sincos[:, 0], pred_sincos[:, 1])
        theta_pred_deg = torch.rad2deg(
            torch.remainder(0.5 * phi, torch.pi)
        )[0].item()

        pred_x_pix = pred_x_norm.item() * img.shape[2]
        pred_y_pix = pred_y_norm.item() * img.shape[1]

        save_debug_image(
            img,
            gt,
            worst_dir / f"worst_{rank:03d}_idx{sample_idx}_err{err_val:.2f}.png",
            pred_x=pred_x_pix,
            pred_y=pred_y_pix,
            pred_angle=theta_pred_deg,
            nbr_bins_xy=args.xy_bin_nbr,
        )

    print("Inference complete.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inference (regression) for EndEffectorPosePredToken (DINOv2)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/dinov2_base_reg_x_y_rot_half_res/model_checkpoint.pt",
        help="Path to model checkpoint (.pt)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/wilah/datasets/heshan_october_grapple_data",
        help="Path to dataset folder (same structure as training)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_inference_with_data_aug_reg/dinov2_base_reg_x_y_rot_half_res",
        help="Output folder for results",
    )

    parser.add_argument(
        "--precision",
        type=float,
        default=1.0,
        help="Ground truth precision used in dataset",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="Number of worst/best predictions to save",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument("--top_crop", type=int, default=1)
    parser.add_argument("--bottom_crop", type=int, default=2)
    parser.add_argument("--left_crop", type=int, default=398)
    parser.add_argument("--right_crop", type=int, default=856)

    parser.add_argument(
        "--xy_bin_nbr",
        type=int,
        default=100,
        help="Number of bins for x and y position",
    )

    args = parser.parse_args()

    run_inference(args)