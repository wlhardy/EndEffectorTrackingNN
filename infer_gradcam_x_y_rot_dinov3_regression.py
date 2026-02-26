import argparse
import csv
import heapq
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_grad_cam
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from torchvision.transforms.v2 import InterpolationMode
from tqdm import tqdm

import eefdataset
import model_token_dinov3_reg

# Reuse the exact helper functions / conventions used in your regression training script
from train_token_x_y_rot_dinov3_reg import (
    ang_error_deg_period180,
    discover_dataset_folders,
    normalize_2d,
)


def half_pixels_resize_and_pad(img, s=np.sqrt(0.5)):
    if hasattr(img, "size"):  # PIL Image
        new_w = round(img.width * s)
        new_h = round(img.height * s)
        img = TF.resize(
            img,
            (new_h, new_w),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        cur_w, cur_h = img.size
    else:  # Tensor (C,H,W)
        _, h, w = img.shape
        new_h = round(h * s)
        new_w = round(w * s)
        img = TF.resize(
            img,
            (new_h, new_w),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        _, cur_h, cur_w = img.shape

    # Compute padding
    pad_w = (14 - cur_w % 14) % 14
    pad_h = (14 - cur_h % 14) % 14

    if pad_w or pad_h:
        # Pad format in torchvision = (left, top, right, bottom)
        img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)

    return img


class ModelOutputWrapper(nn.Module):
    """Wraps a multi-output model to expose a single head output for GradCAM."""

    def __init__(self, model, output_fn):
        super().__init__()
        self.model = model
        self.output_fn = output_fn

    def forward(self, x):
        return self.output_fn(self.model(x))


DINOV3_REPO_DIR = "dinov3"
DINO_CHECKPOINT_DICT = {
    "dinov3_vitb16": "dinov3_pth/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_pth/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vits16": "dinov3_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
}

CAM_TYPES = {
    # "GradCAM": pytorch_grad_cam.GradCAM,
    # "GradCAMPlusPlus": pytorch_grad_cam.GradCAMPlusPlus,
    "GradCAMElementWise": pytorch_grad_cam.GradCAMElementWise,
    "HiResCAM": pytorch_grad_cam.HiResCAM,
    "LayerCAM": pytorch_grad_cam.LayerCAM,
    "EigenCAM": pytorch_grad_cam.EigenCAM,
    "EigenGradCAM": pytorch_grad_cam.EigenGradCAM,
    "KPCA-CAM": pytorch_grad_cam.KPCA_CAM,
    # "XGradCAM": pytorch_grad_cam.XGradCAM,
    # "FinerCAM": pytorch_grad_cam.FinerCAM,
}


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
        args.dinov3_size,
        source="local",
        weights=DINO_CHECKPOINT_DICT[args.dinov3_size],
        force_reload=False,
    )
    ee_model = model_token_dinov3_reg.EndEffectorPosePredToken(backbone_model).to(
        device
    )
    ee_model.load_state_dict(ckpt["model_state_dict"])
    ee_model.eval()

    # === Dataset (same preprocessing as validation in training) ===
    transform_val = T.Compose(
        [
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
        ]
    )

    image_dirs, joint_csvs, xy_csvs = discover_dataset_folders(args.dataset)
    dataset = eefdataset.EEFDataset(
        image_dirs=image_dirs,
        joint_csv_paths=joint_csvs,
        xy_csv_paths=xy_csvs,
        joint_precision=args.precision,
        xy_bin_nbr=args.xy_bin_nbr,
        transform=transform_val,
    )
    # === Select indices based on mode ===
    rng = np.random.default_rng(args.seed)
    if args.best_worst:
        # Run over the full dataset to find the actual best/worst
        selected_indices = list(range(len(dataset)))
    else:
        # Random subset of gradcam_n samples
        selected_indices = rng.permutation(len(dataset))[
            : min(args.gradcam_n, len(dataset))
        ].tolist()

    subset = torch.utils.data.Subset(dataset, selected_indices)
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # === Output folders ===
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = Path(args.output_dir) / "results.csv"

    # === CSV header (REGRESSION) ===
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
            ]
        )

    # === Keep top-N worst/best errors using heaps (best_worst mode only) ===
    worst_heap = []  # (total_error, sample_index)
    best_heap = []  # (-total_error, sample_index)

    print("Running inference (regression) ...")
    for batch_i, (images, joint_values) in enumerate(tqdm(dataloader)):
        images = images.to(device)

        # --- GT ---
        gt_theta_deg = (
            joint_values["base_joint"].to(device).to(torch.float32)
        )  # degrees in [0,180)
        gt_x_bin = joint_values["x"].to(device).to(torch.float32)  # integer bins
        gt_y_bin = joint_values["y"].to(device).to(torch.float32)

        # training reg uses x/100 and y/100 as targets
        gt_x_norm = gt_x_bin / float(args.xy_bin_nbr)
        gt_y_norm = gt_y_bin / float(args.xy_bin_nbr)

        # --- Pred ---
        pred_sincos, pred_x_norm, pred_y_norm = ee_model(images)

        # Some model implementations may `squeeze()` outputs when batch_size == 1,
        # producing 0-d tensors. Force explicit batch dimensions everywhere.
        pred_sincos = pred_sincos.view(-1, 2)
        pred_x_norm = pred_x_norm.view(-1)
        pred_y_norm = pred_y_norm.view(-1)
        gt_theta_deg = gt_theta_deg.view(-1)
        gt_x_bin = gt_x_bin.view(-1)
        gt_y_bin = gt_y_bin.view(-1)

        # Normalize the (sin, cos) vector to unit length before decoding.
        pred_sincos = normalize_2d(pred_sincos)

        # Decode: model predicts sin(2θ), cos(2θ)
        phi = torch.atan2(pred_sincos[:, 0], pred_sincos[:, 1])
        theta_pred_rad = 0.5 * phi
        theta_pred_rad = torch.remainder(theta_pred_rad, torch.pi)  # [0, pi)
        theta_pred_deg = torch.rad2deg(theta_pred_rad)

        angular_error = torch.abs(ang_error_deg_period180(theta_pred_deg, gt_theta_deg))

        # Compare x/y in bin units for interpretability
        pred_x_bin = pred_x_norm * float(args.xy_bin_nbr)
        pred_y_bin = pred_y_norm * float(args.xy_bin_nbr)
        x_error_bins = torch.abs(pred_x_bin - gt_x_bin)
        y_error_bins = torch.abs(pred_y_bin - gt_y_bin)

        total_error = angular_error + x_error_bins + y_error_bins

        # === Stream results to CSV ===
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for j in range(images.size(0)):
                global_idx = selected_indices[batch_i * args.batch_size + j]
                err_val = total_error[j].item()

                writer.writerow(
                    [
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
                    ]
                )

                if args.best_worst:
                    # Worst N
                    if len(worst_heap) < args.gradcam_n:
                        heapq.heappush(worst_heap, (err_val, global_idx))
                    elif err_val > worst_heap[0][0]:
                        heapq.heapreplace(worst_heap, (err_val, global_idx))

                    # Best N (max-heap via negative error)
                    if len(best_heap) < args.gradcam_n:
                        heapq.heappush(best_heap, (-err_val, global_idx))
                    elif err_val < -best_heap[0][0]:
                        heapq.heapreplace(best_heap, (-err_val, global_idx))

        # Free batch memory
        del images, pred_sincos, pred_x_norm, pred_y_norm
        torch.cuda.empty_cache()

    # === GradCAM setup ===
    gradcam_dir = Path(args.output_dir) / "gradcam"

    # Number of prefix tokens prepended before patch tokens
    n_prefix = ee_model.nbr_tokens + ee_model.backbone.n_storage_tokens

    target_layers = {
        # "block05": ee_model.backbone.blocks[5].norm1,
        # "block11": ee_model.backbone.blocks[11].norm1,
        # "block17": ee_model.backbone.blocks[17].norm1,
        "block23": ee_model.backbone.blocks[-1].norm1,  # 23 for large
    }

    def _decode_angle_deg(o):
        """Decode (sin2θ, cos2θ) → θ in degrees, keeping grad flow."""
        sc = normalize_2d(o[0].reshape(-1, 2))
        return torch.rad2deg(
            torch.remainder(0.5 * torch.atan2(sc[:, 0], sc[:, 1]), torch.pi)
        ).reshape(-1, 1)

    # One wrapper per prediction head
    wrapped_heads = [
        (
            "Angle (deg)",
            ModelOutputWrapper(ee_model, _decode_angle_deg),
        ),
        # (
        #     "x_pos",
        #     ModelOutputWrapper(
        #         ee_model,
        #         lambda o: o[1].view(-1, 1),
        #     ),
        # ),
        # (
        #     "y_pos",
        #     ModelOutputWrapper(
        #         ee_model,
        #         lambda o: o[2].view(-1, 1),
        #     ),
        # ),
        (
            "Position (px)",
            ModelOutputWrapper(
                ee_model,
                lambda o: (o[1] + o[2]).view(-1, 1),
            ),
        ),
    ]

    gradcam_targets = [RawScoresOutputTarget()]

    # Determine which indices to pre-compute GradCAM data for
    if args.best_worst:
        worst_sorted = sorted(worst_heap, key=lambda x: x[0], reverse=True)
        best_sorted = sorted([(-e, i) for (e, i) in best_heap], key=lambda x: x[0])
        gradcam_indices = list(
            dict.fromkeys([i for _, i in worst_sorted] + [i for _, i in best_sorted])
        )
    else:
        gradcam_indices = selected_indices

    # Pre-compute predictions and image data for each GradCAM sample once,
    # so these are not re-run inside the CAM type loop.
    print(
        f"Pre-computing sample predictions for {len(gradcam_indices)} GradCAM sample(s) ..."
    )
    sample_data = []
    for sample_idx in tqdm(gradcam_indices, desc="Pre-compute"):
        img, gt = dataset[sample_idx]
        pred_sincos_s, pred_x_s, pred_y_s = ee_model(img.unsqueeze(0).to(device))
        pred_sincos_s = normalize_2d(pred_sincos_s.view(-1, 2))
        phi_s = torch.atan2(pred_sincos_s[:, 0], pred_sincos_s[:, 1])
        theta_pred_deg_s = torch.rad2deg(torch.remainder(0.5 * phi_s, torch.pi))[
            0
        ].item()
        pred_x_norm_s = pred_x_s.view(-1)
        pred_y_norm_s = pred_y_s.view(-1)
        # Detect zero-padding added at the bottom/right by the transform.
        img_np_full = np.clip(img.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0).astype(
            np.float32
        )
        row_max = img_np_full.max(axis=(1, 2))
        col_max = img_np_full.max(axis=(0, 2))
        nonzero_rows = np.where(row_max > 0)[0]
        nonzero_cols = np.where(col_max > 0)[0]
        unpadded_h = int(nonzero_rows[-1]) + 1 if len(nonzero_rows) else img_np_full.shape[0]
        unpadded_w = int(nonzero_cols[-1]) + 1 if len(nonzero_cols) else img_np_full.shape[1]
        sample_data.append(
            {
                "img": img,
                "gt": gt,
                "img_np": img_np_full,
                "unpadded_h": unpadded_h,
                "unpadded_w": unpadded_w,
                "theta_pred_deg": theta_pred_deg_s,
                "pred_x_pix": pred_x_norm_s.item() * img.shape[2],
                "pred_y_pix": pred_y_norm_s.item() * img.shape[1],
                # GT tensors for loss-based GradCAM
                "gt_theta_deg": float(gt["base_joint"]),
                "gt_x_norm": float(gt["x"]) / args.xy_bin_nbr,
                "gt_y_norm": float(gt["y"]) / args.xy_bin_nbr,
            }
        )

    idx_to_sd = {idx: sd for idx, sd in zip(gradcam_indices, sample_data)}

    def make_loss_heads(sd):
        """Build wrapped heads that output per-head MAE w.r.t. GT for loss-based GradCAM."""
        gt_theta_t = torch.tensor(
            sd["gt_theta_deg"], dtype=torch.float32, device=device
        )
        gt_x_t = torch.tensor(sd["gt_x_norm"], dtype=torch.float32, device=device)
        gt_y_t = torch.tensor(sd["gt_y_norm"], dtype=torch.float32, device=device)
        return [
            (
                "\u03b8_err",
                ModelOutputWrapper(
                    ee_model,
                    lambda o, _gt=gt_theta_t: torch.abs(
                        _decode_angle_deg(o).reshape(-1) - _gt
                    ).reshape(-1, 1),
                ),
            ),
            (
                "x_err",
                ModelOutputWrapper(
                    ee_model,
                    lambda o, _gt=gt_x_t: torch.abs(o[1].view(-1) - _gt).reshape(-1, 1),
                ),
            ),
            (
                "y_err",
                ModelOutputWrapper(
                    ee_model,
                    lambda o, _gt=gt_y_t: torch.abs(o[2].view(-1) - _gt).reshape(-1, 1),
                ),
            ),
        ]

    def save_gradcam_figure(sd, cam_class, target_layer, save_path, title, heads=None):
        """Compute and save a GradCAM overlay figure for a single sample."""
        if heads is None:
            heads = wrapped_heads
        img = sd["img"]
        img_tensor = img.unsqueeze(0).to(device)

        def reshape_transform(tensor):
            """Strip prefix tokens and reshape [B, N, C] -> [B, C, H, W]."""
            patch_tokens = tensor[:, n_prefix:, :]
            B, N, C = patch_tokens.shape
            h = img_tensor.shape[2] // ee_model.patch_size
            w = img_tensor.shape[3] // ee_model.patch_size
            return patch_tokens.reshape(B, h, w, C).permute(0, 3, 1, 2)

        cams = []
        with torch.enable_grad():
            for head_name, wrapped_model in heads:
                cam_engine = cam_class(
                    model=wrapped_model,
                    target_layers=[target_layer],
                    reshape_transform=reshape_transform,
                )
                grayscale_cam = cam_engine(
                    input_tensor=img_tensor, targets=gradcam_targets
                )
                cams.append((head_name, grayscale_cam[0]))

        gt = sd["gt"]
        uh, uw = sd["unpadded_h"], sd["unpadded_w"]
        img_np = sd["img_np"][:uh, :uw]
        n_cols = 1 + len(cams)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), facecolor="white")
        for ax in axes:
            ax.set_facecolor("white")

        axes[0].imshow(img_np)
        gt_x_pix = gt["x"] * img.shape[2] / args.xy_bin_nbr
        gt_y_pix = gt["y"] * img.shape[1] / args.xy_bin_nbr
        axes[0].scatter(
            gt_x_pix,
            gt_y_pix,
            c="lime",
            s=40,
            marker="o",
            label="GT",
        )
        axes[0].scatter(
            sd["pred_x_pix"],
            sd["pred_y_pix"],
            c="red",
            s=40,
            marker="x",
            label="Pred",
        )
        axes[0].legend(
            loc="lower right",
            fontsize=16,
            frameon=True,
            edgecolor="black",
            framealpha=0.8,
        )
        axes[0].axis("off")

        for ax_i, (head_name, cam_map) in enumerate(cams):
            overlay = show_cam_on_image(img_np, cam_map[:uh, :uw], use_rgb=True)
            axes[ax_i + 1].imshow(overlay)
            axes[ax_i + 1].set_title(head_name, fontsize=16)
            axes[ax_i + 1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    if args.best_worst:
        # === GradCAM for worst/best top-N ===
        for label, ranked in [("worst", worst_sorted), ("best", best_sorted)]:
            print(f"Saving GradCAM for {len(ranked)} {label} predictions ...")
            for cam_type_name, cam_class in CAM_TYPES.items():
                for layer_name, layer in target_layers.items():
                    cam_dir = (
                        Path(args.output_dir)
                        / f"{label}_predictions"
                        / cam_type_name
                        / layer_name
                    )
                    cam_dir.mkdir(exist_ok=True, parents=True)
                    loss_cam_dir = None
                    if args.worst_loss and label == "worst":
                        loss_cam_dir = (
                            Path(args.output_dir)
                            / f"{label}_predictions_loss"
                            / cam_type_name
                            / layer_name
                        )
                        loss_cam_dir.mkdir(exist_ok=True, parents=True)
                    for rank, (err_val, sample_idx) in enumerate(
                        tqdm(
                            ranked,
                            desc=f"GradCAM [{cam_type_name}/{layer_name}] {label}",
                        )
                    ):
                        sd = idx_to_sd[sample_idx]
                        gt = sd["gt"]
                        title = (
                            f"{label.capitalize()} #{rank} | err={err_val:.2f}\n"
                            f"GT: {gt['base_joint']:.1f}\u00b0 | Pred: {sd['theta_pred_deg']:.1f}\u00b0"
                        )
                        fname = (
                            f"{label}_{rank:03d}_idx{sample_idx}_err{err_val:.2f}.pdf"
                        )
                        save_gradcam_figure(
                            sd, cam_class, layer, cam_dir / fname, title
                        )
                        if loss_cam_dir is not None:
                            save_gradcam_figure(
                                sd,
                                cam_class,
                                layer,
                                loss_cam_dir / fname,
                                title,
                                heads=make_loss_heads(sd),
                            )
    else:
        # === GradCAM on random gradcam_n samples, one folder per CAM type / layer ===
        gradcam_dir.mkdir(exist_ok=True, parents=True)
        print(
            f"Running GradCAM on {len(gradcam_indices)} samples for "
            f"{len(CAM_TYPES)} method(s) x {len(target_layers)} layer(s) ..."
        )
        for cam_type_name, cam_class in CAM_TYPES.items():
            for layer_name, layer in target_layers.items():
                cam_type_dir = gradcam_dir / cam_type_name / layer_name
                cam_type_dir.mkdir(exist_ok=True, parents=True)
                for rank, (sample_idx, sd) in enumerate(
                    tqdm(
                        zip(gradcam_indices, sample_data),
                        total=len(gradcam_indices),
                        desc=f"GradCAM [{cam_type_name}/{layer_name}]",
                    )
                ):
                    gt = sd["gt"]
                    title = f"GT: {gt['base_joint']:.1f}\u00b0 | Pred: {sd['theta_pred_deg']:.1f}\u00b0"
                    save_gradcam_figure(
                        sd,
                        cam_class,
                        layer,
                        cam_type_dir / f"gradcam_{rank:04d}_idx{sample_idx}.pdf",
                        title,
                    )

    # === Plot angular error histogram + error vs GT angle ===
    print("Generating plots ...")
    results = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    angular_errors = results[:, 1]
    gt_angles = results[:, 5]

    mean_err = np.mean(angular_errors)
    std_err = np.std(angular_errors)
    stats_text = f"Mean: {mean_err:.2f}°\nStd: {std_err:.2f}°"

    plt.figure()
    plt.hist(angular_errors, bins=50, edgecolor="black")
    plt.title("Angular Error Distribution (°)")
    plt.xlabel("Error (°)")
    plt.ylabel("Frequency")
    plt.text(
        0.97,
        0.97,
        stats_text,
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.savefig(Path(args.output_dir) / "angular_error_histogram.png", dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(gt_angles, angular_errors, s=10, alpha=0.6)
    plt.xlabel("Ground Truth Base Joint (°)")
    plt.ylabel("Angular Error (°)")
    plt.title("Prediction Error vs Ground Truth Angle")
    plt.text(
        0.97,
        0.97,
        stats_text,
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.savefig(Path(args.output_dir) / "error_vs_gt_angle.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference (regression) for EndEffectorPosePredToken (DINOv3)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (.pt)",
        default="/home/wilah/workspace/EndEffectorTrackingNN/training/dinov3_base_reg_x_y_rot_quarter_res/model_checkpoint.pt",
    )
    parser.add_argument(
        "--dinov3_size",
        type=str,
        choices=["dinov3_vitb16", "dinov3_vitl16", "dinov3_vits16"],
        help="DINOv3 backbone size",
        default="dinov3_vitb16",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset folder (same structure as training)",
        default="/home/wilah/datasets/heshan_october_grapple_data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output folder for results",
        default="results_inference_with_data_aug_reg/dinov3_base_reg_x_y_rot_test_infer_script_quarter_res",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=1.0,
        help="Ground truth precision used in dataset",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--gradcam_n",
        type=int,
        default=100,
        help="Number of GradCAM samples: random subset in normal mode, top-N best+worst in --best_worst mode",
    )
    parser.add_argument(
        "--best_worst",
        action="store_true",
        help="Run inference on all images, then save GradCAM for the top-gradcam_n best and worst predictions",
    )
    parser.add_argument(
        "--worst_loss",
        action="store_true",
        help="When used with --best_worst, also save an additional GradCAM pass for worst predictions "
        "using per-head MAE w.r.t. GT as the target (highlights regions that caused the error)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for sample shuffling"
    )
    parser.add_argument("--num_workers", type=int, default=2)
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
