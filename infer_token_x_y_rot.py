import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import argparse
from tqdm import tqdm
import csv
import heapq

import eefdataset
import model_token
from train_token_x_y_rot import save_debug_image, half_pixels_resize_and_pad, discover_dataset_folders


@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Load checkpoint ===
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # === Load model ===
    backbone = torch.hub.load('facebookresearch/dinov2', ckpt['dinov2_model'], force_reload=False)
    ee_model = model_token.EndEffectorPosePredToken(backbone,
                                                    num_classes_joint=ckpt['num_classes'],
                                                    nbr_classes_xy=100).to(device)
    ee_model.load_state_dict(ckpt['model_state_dict'])
    ee_model.eval()

    # === Dataset ===
    transform_val = T.Compose([
        T.Lambda(lambda img: TF.rotate(img, 180)),
        T.Lambda(lambda img: TF.crop(img, top=args.top_crop, left=args.left_crop,
                                     height=img.height - args.bottom_crop, width=img.width - args.right_crop)),
        T.Lambda(half_pixels_resize_and_pad),
        T.ToTensor()
    ])
    image_dirs, joint_csvs, xy_csvs = discover_dataset_folders(args.dataset)
    dataset = eefdataset.EEFDataset(image_dirs=image_dirs, joint_csv_paths=joint_csvs,
                                    xy_csv_paths=xy_csvs, joint_precision=args.precision,
                                    xy_bin_nbr=100, transform=transform_val)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)

    # === Output folders ===
    os.makedirs(args.output_dir, exist_ok=True)
    worst_dir = Path(args.output_dir) / "worst_predictions"
    worst_dir.mkdir(exist_ok=True, parents=True)
    csv_path = Path(args.output_dir) / "results.csv"

    # === CSV header ===
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "angular_error", "x_error", "y_error",
                         "total_error", "gt_base_joint", "gt_base_joint_quant", "pred_base_joint",
                         "gt_x", "pred_x", "gt_y", "pred_y"])

    # === Keep top-N worst errors using a min-heap ===
    worst_heap = []  # (total_error, sample_index)
    heapq.heapify(worst_heap)

    print("Running inference ...")
    for batch_i, (images, joint_values) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        base_joint_quant = joint_values['base_joint_quant'].to(device)
        base_x = joint_values['x'].to(device)
        base_y = joint_values['y'].to(device)

        base_joint_logits, base_x_logits, base_y_logits = ee_model(images)
        base_joint_preds = base_joint_logits.argmax(dim=1)
        base_x_preds = base_x_logits.argmax(dim=1)
        base_y_preds = base_y_logits.argmax(dim=1)

        base_joint_pred_angles = eefdataset.class_to_angle(base_joint_preds, args.precision, symmetric=True)
        base_joint_gt_angles = eefdataset.class_to_angle(base_joint_quant, args.precision, symmetric=True)
        angular_error = torch.abs(base_joint_pred_angles - base_joint_gt_angles)
        angular_error = torch.where(angular_error > 90, 180 - angular_error, angular_error)

        x_error = torch.abs(base_x_preds - base_x)
        y_error = torch.abs(base_y_preds - base_y)
        total_error = angular_error + x_error.float() + y_error.float()

        # === Stream results to CSV ===
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for j in range(images.size(0)):
                global_idx = batch_i * args.batch_size + j
                writer.writerow([
                    global_idx,
                    angular_error[j].item(),
                    x_error[j].item(),
                    y_error[j].item(),
                    total_error[j].item(),
                    joint_values['base_joint'][j].item(),
                    base_joint_gt_angles[j].item(),
                    base_joint_pred_angles[j].item(),
                    joint_values['x'][j].item(),
                    base_x_preds[j].item(),
                    joint_values['y'][j].item(),
                    base_y_preds[j].item()
                ])

                # Track top-N worst errors
                err_val = total_error[j].item()
                if len(worst_heap) < args.top_n:
                    heapq.heappush(worst_heap, (err_val, global_idx))
                else:
                    heapq.heappushpop(worst_heap, (err_val, global_idx))

        # Free batch memory
        del images, base_joint_logits, base_x_logits, base_y_logits
        torch.cuda.empty_cache()

    # === Sort heap descending (worst first) ===
    worst_heap.sort(key=lambda x: x[0], reverse=True)

    print(f"Saving {len(worst_heap)} worst predictions ...")
    for rank, (err_val, sample_idx) in enumerate(worst_heap):
        img, gt = dataset[sample_idx]  # reload from dataset
        img_tensor = img.to(device).unsqueeze(0)
        base_joint_logits, base_x_logits, base_y_logits = ee_model(img_tensor)
        pred_angle = eefdataset.class_to_angle(base_joint_logits.argmax(dim=1),
                                            args.precision, symmetric=True)[0].item()
        pred_x = base_x_logits.argmax(dim=1)[0].item()
        pred_y = base_y_logits.argmax(dim=1)[0].item()
        save_debug_image(
            img,
            gt,
            worst_dir / f"worst_{rank:03d}_idx{sample_idx}_err{err_val:.2f}.png",
            pred_x=pred_x / args.xy_bin_nbr * img.shape[2],
            pred_y=pred_y / args.xy_bin_nbr * img.shape[1],
            pred_angle=pred_angle
        )

    # === Plot angular error histogram ===
    print("Generating plots ...")
    results = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    angular_errors = results[:, 1]
    gt_angles = results[:, 5]

    # Compute stats
    mean_err = np.mean(angular_errors)
    std_err = np.std(angular_errors)
    stats_text = f"Mean: {mean_err:.2f}°\nStd: {std_err:.2f}°"

    # --- Histogram ---
    plt.figure()
    plt.hist(angular_errors, bins=50, color='skyblue', edgecolor='black')
    plt.title("Angular Error Distribution (°)")
    plt.xlabel("Error (°)")
    plt.ylabel("Frequency")

    # Add stats text (top-right corner)
    plt.text(
        0.97, 0.97, stats_text,
        ha='right', va='top',
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.savefig(Path(args.output_dir) / "angular_error_histogram.png", dpi=150)
    plt.close()

    # --- Scatter plot ---
    plt.figure()
    plt.scatter(gt_angles, angular_errors, s=10, alpha=0.6)
    plt.xlabel("Ground Truth Base Joint (°)")
    plt.ylabel("Angular Error (°)")
    plt.title("Prediction Error vs Ground Truth Angle")

    # Add stats text again
    plt.text(
        0.97, 0.97, stats_text,
        ha='right', va='top',
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.savefig(Path(args.output_dir) / "error_vs_gt_angle.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for EndEffectorPosePredToken")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset folder (same structure as training)")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Output folder for results")
    parser.add_argument("--precision", type=float, default=1.0, help="Ground truth precision used in dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--top_n", type=int, default=30, help="Number of worst predictions to save")
    parser.add_argument("--top_crop", type=int, default=1)
    parser.add_argument("--bottom_crop", type=int, default=2)
    parser.add_argument("--left_crop", type=int, default=398)
    parser.add_argument("--right_crop", type=int, default=856)
    parser.add_argument("--xy_bin_nbr", type=int, default=100, help="Number of bins for x and y position")
    args = parser.parse_args()

    run_inference(args)
