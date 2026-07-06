import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torchvision.transforms.v2 import InterpolationMode
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from PIL import Image
from pathlib import Path
import os
import argparse
import wandb
import math
import random
import datetime
import tqdm
import multiprocessing
import time
from torch.utils.data import DataLoader, SubsetRandomSampler

import eefdataset
import model_token_dinov3_reg

matplotlib.use("Agg")

os.environ["TORCH_HOME"] = os.path.expanduser("~/.cache2/torch")
torch.hub.set_dir(os.environ["TORCH_HOME"])
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

DEBUG = 0
VERBOSE = 0
COMPUTE_ERROR_IN_TRAINING = True
RUN_VALIDATION = True
DINOV3_REPO_DIR = "dinov3"

DINO_CHECKPOINT_DICT = {
    'dinov3_vitb16': 'dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    'dinov3_vitl16': 'dinov3/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
    'dinov3_vits16': 'dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
}

def normalize_2d(v, eps=1e-8):
    return v / (v.norm(dim=1, keepdim=True) + eps)

def ang_error_deg_period180(pred_deg, gt_deg):
    # both in degrees, return minimal error under 180° periodicity
    d = pred_deg - gt_deg                  # signed difference
    d = torch.remainder(d + 90.0, 180.0)   # shift, wrap to [0,180)
    d = d - 90.0                           # shift back to [-90,90)
    return d

def save_debug_image(image_tensor, joint_values, save_path,
                     pred_x=None, pred_y=None, pred_angle=None,
                     nbr_bins_xy=100):
    """
    image_tensor: (C,H,W) tensor in [0,1]
    joint_values: dict with 'x', 'y', 'base_joint' (ground truth)
    pred_x, pred_y: predicted pixel coordinates (optional)
    pred_angle: predicted base joint angle in degrees (optional)
    save_path: where to save the debug image
    """
    img = image_tensor.permute(1, 2, 0).cpu().numpy()  # (H,W,C)
    
    # Plot image
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    x = joint_values['x'] * image_tensor.shape[2] / nbr_bins_xy
    y = joint_values['y'] * image_tensor.shape[1] / nbr_bins_xy

    # Overlay GT point
    ax.scatter(x, y, c='red', s=40, marker='x', label="GT")

    # Overlay prediction if provided
    if pred_x is not None and pred_y is not None:
        ax.scatter(pred_x, pred_y, c='lime', s=40, marker='o', label="Prediction")

    # Write angles at bottom
    angle_text = f"GT: {joint_values['base_joint']:.1f}°"
    if pred_angle is not None:
        angle_text += f" | Pred: {pred_angle:.1f}°"
    ax.text(
        0.5, 1.02, angle_text,
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=10, color='white', backgroundcolor='black'
    )

    ax.axis('off')
    ax.legend(loc="lower right", fontsize=8, facecolor="black", edgecolor="white", labelcolor="white")
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def half_pixels_resize_and_pad(img, s=np.sqrt(0.5)):
    if hasattr(img, "size"):  # PIL Image
        new_w = round(img.width * s)
        new_h = round(img.height * s)
        img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BILINEAR, antialias=True)
        cur_w, cur_h = img.size
    else:  # Tensor (C,H,W)
        _, h, w = img.shape
        new_h = round(h * s)
        new_w = round(w * s)
        img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BILINEAR, antialias=True)
        _, cur_h, cur_w = img.shape

    # Compute padding
    pad_w = (14 - cur_w % 14) % 14
    pad_h = (14 - cur_h % 14) % 14

    if pad_w or pad_h:
        # Pad format in torchvision = (left, top, right, bottom)
        img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)

    return img

def discover_dataset_folders(main_folder):
    image_dirs = []
    joint_csvs = []
    xy_csvs = []

    main_path = Path(main_folder)
    for sub in main_path.iterdir():
        if sub.is_dir():
            img_dir = sub / "rgb" / "left"
            joint_csv = sub / "calibration_results" / "calibrated_joint_values.csv"
            xy_csv = sub / "pose_analysis" / "projection" / "center_piece_pixel_coordinates.csv"

            if img_dir.exists() and joint_csv.exists():
                image_dirs.append(str(img_dir))
                joint_csvs.append(str(joint_csv))
                if xy_csv.exists():
                    xy_csvs.append(str(xy_csv))

    return image_dirs, joint_csvs, xy_csvs


def get_balanced_indices(dataset, num_samples_per_class):
    base_joint_classes = np.array([eefdataset.quantize_joint(dataset[i][1]['base_joint'], dataset.joint_precision) for i in range(len(dataset))])

    class_indices = defaultdict(list)
    for idx, label in enumerate(base_joint_classes):
        class_indices[label].append(idx)

    # Sample with replacement to get same count per class
    balanced_indices = []
    for label, indices in class_indices.items():
        chosen = np.random.choice(indices, num_samples_per_class, replace=True)
        balanced_indices.extend(chosen)

    np.random.shuffle(balanced_indices)
    return balanced_indices

def train(config=None):
    try:
        with wandb.init(config=config):
            config = wandb.config
            random.seed(config.random_seed)
            log_interval = getattr(config, "log_interval", 50)
            # Log the time to build the datasets
            start_time = datetime.datetime.now()
            transform_train = T.Compose([T.Lambda(lambda img: TF.rotate(img, 180)),
                                        T.Lambda(lambda img: TF.crop(img, top=config.top_cropping, left=config.left_cropping, height=img.height - config.bottom_cropping, width=img.width - config.right_cropping)),
                                        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
                                        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
                                        T.RandomInvert(),
                                        T.RandomPosterize(bits=4),
                                        T.RandomSolarize(threshold=192),
                                        T.RandomAdjustSharpness(sharpness_factor=2),
                                        T.RandomAutocontrast(),
                                        T.RandomEqualize(),
                                        T.Lambda(half_pixels_resize_and_pad),
                                        T.ToTensor(),
                                        T.RandomErasing(p=0.5, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
                                        ])
            train_image_dirs, train_joint_csvs, train_xy_csvs = discover_dataset_folders(config.train_main_folder_path)
            dataset_train = eefdataset.EEFDataset(image_dirs=train_image_dirs, joint_csv_paths=train_joint_csvs,
                                            xy_csv_paths=train_xy_csvs, joint_precision=config.ground_truth_precision,
                                            xy_bin_nbr=100, transform=transform_train)
            
            transform_val = T.Compose([T.Lambda(lambda img: TF.rotate(img, 180)),
                                    T.Lambda(lambda img: TF.crop(img, top=config.top_cropping, left=config.left_cropping, height=img.height - config.bottom_cropping, width=img.width - config.right_cropping)),
                                    T.Lambda(half_pixels_resize_and_pad),
                                    T.ToTensor()])
            val_image_dirs, val_joint_csvs, val_xy_csvs = discover_dataset_folders(config.val_main_folder_path)
            dataset_val = eefdataset.EEFDataset(image_dirs=val_image_dirs, joint_csv_paths=val_joint_csvs,
                                            xy_csv_paths=val_xy_csvs, joint_precision=config.ground_truth_precision,
                                            xy_bin_nbr=100, transform=transform_val)
            
            dataset_train.save_to_csv(".tmp/train_dataset.csv")
            dataset_val.save_to_csv(".tmp/val_dataset.csv")

            end_time = datetime.datetime.now()
            time_to_build = end_time - start_time
            print(f"Time to build datasets: {time_to_build}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            # Load DINOv3
            backbone_model = torch.hub.load(DINOV3_REPO_DIR, config.backbone, source='local', weights=DINO_CHECKPOINT_DICT[config.backbone], force_reload=False)
            for param in backbone_model.parameters():
                param.requires_grad = True

            num_blocks_to_freeze = config.freeze_blocks
            for i, block in enumerate(backbone_model.blocks):
                if i < num_blocks_to_freeze:
                    for param in block.parameters():
                        param.requires_grad = False

            if config.freeze_patch_embed:
                print("Freezing patch embedding.")
                backbone_model.patch_embed.requires_grad = False

            xy_bin_nbr = config.xy_bin_nbr

            # Load model
            ee_model = model_token_dinov3_reg.EndEffectorPosePredToken(backbone_model).to(device)
            decay = []
            no_decay = []

            for name, param in ee_model.named_parameters():
                if param.ndim == 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": 1e-2},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=config.learning_rate,
            )

            criterion_joints = nn.MSELoss()
            criterion_pixel = nn.MSELoss()

            # Create a directory to log checkpoints and results
            os.makedirs("training", exist_ok=True)

            # Use the date and time to create a unique directory
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join("training", f"checkpoint_{timestamp}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            #target_batch_size = config.target_batch_size
            #batch_size = min(config.max_batch_size, target_batch_size)
            target_batch_size = 1
            batch_size = 1
            accumulation_steps = max(1, target_batch_size // batch_size)

            weight_loss_joints = config.weight_ratio_joints
            weight_loss_xy = 1.0 - weight_loss_joints

            cpu_count = multiprocessing.cpu_count()
            train_cpu_count = min(12, cpu_count)
            val_cpu_count = min(12, cpu_count)
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=train_cpu_count, shuffle=True, persistent_workers=True)

            if RUN_VALIDATION:
                dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=val_cpu_count, persistent_workers=True)

            num_training_steps = np.ceil(len(dataloader_train) / log_interval)
            print(f"Number of training steps per epoch: {num_training_steps}")
            num_warmup_steps = int(0.1 * num_training_steps)
            print(f"Number of warmup steps: {num_warmup_steps}")

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=0.01, total_iters=num_warmup_steps
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=num_training_steps - num_warmup_steps
                    )
                ],
                milestones=[num_warmup_steps]
            )

            for epoch in range(config.epochs):
                ee_model.train()
                running_loss = 0
                running_loss_joints = 0
                running_loss_pixel = 0
                img_total = 0
                angular_error_base_joint_total = 0
                x_error_bin_total = 0
                y_error_bin_total = 0
                train_average_error = 0

                running_loss_interval = 0.0
                running_loss_joints_interval = 0.0
                running_loss_pixel_interval = 0.0
                angular_error_interval = 0.0
                x_error_interval = 0.0
                y_error_interval = 0.0
                samples_interval = 0

                for i, (images, joint_values, image_names) in enumerate(tqdm.tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{config.epochs}")):
                    # Start a timer to measure the training step duration
                    step_start_time = time.time()
                    theta_rad = torch.deg2rad(joint_values['base_joint'].to(device).to(torch.float32))
                    base_joint_sin_gt = torch.sin(2.0 * theta_rad)
                    base_joint_cos_gt = torch.cos(2.0 * theta_rad)
                    base_x_gt = joint_values['x'].to(device) / 100.0
                    base_y_gt = joint_values['y'].to(device) / 100.0
                    base_joint_gt = torch.stack([base_joint_sin_gt, base_joint_cos_gt], dim=-1)
                    base_x_gt = base_x_gt.to(torch.float32)
                    base_y_gt = base_y_gt.to(torch.float32)
                    base_joint_gt = base_joint_gt.to(torch.float32)

                    if DEBUG > 2:
                        # Save all images in the batch to disk for debugging
                        for j in range(images.size(0)):
                            img = images[j].cpu().numpy().transpose(1, 2, 0)
                            img = (img * 255).astype(np.uint8)
                            img = Image.fromarray(img)
                            img_path = os.path.join(checkpoint_dir, f"debug_image_epoch{epoch+1}_batch{i+1}_img{j+1}.png")
                            img.save(img_path)

                    images = images.to(device)

                    base_joint_sincos, base_x, base_y = ee_model(images)
                    base_joint_sincos = normalize_2d(base_joint_sincos)

                    # Compute loss
                    loss_joints = criterion_joints(base_joint_sincos, base_joint_gt)
                    loss_pixel = criterion_pixel(base_x, base_x_gt) + criterion_pixel(base_y, base_y_gt)
                    
                    loss = (loss_joints * weight_loss_joints) + (loss_pixel * weight_loss_xy)
                    running_loss += loss.item()
                    running_loss_joints += loss_joints.item()
                    running_loss_pixel += loss_pixel.item()
                    
                    loss.backward()

                    # accumulate interval stats
                    running_loss_interval += loss.item()
                    running_loss_joints_interval += loss_joints.item()
                    running_loss_pixel_interval += loss_pixel.item()

                    # log every N batches
                    if (i + 1) % log_interval == 0:
                        avg_loss_interval = running_loss_interval / log_interval
                        avg_loss_joints_interval = running_loss_joints_interval / log_interval
                        avg_loss_pixel_interval = running_loss_pixel_interval / log_interval
                        mean_ae_interval = angular_error_interval / samples_interval
                        mean_x_interval = x_error_interval / samples_interval
                        mean_y_interval = y_error_interval / samples_interval

                        wandb.log({
                            "train_batch_loss": avg_loss_interval,
                            "train_batch_loss_joints": avg_loss_joints_interval,
                            "train_batch_loss_pixel": avg_loss_pixel_interval,
                            "train_batch_angular_error": mean_ae_interval,
                            "train_batch_x_error": mean_x_interval,
                            "train_batch_y_error": mean_y_interval,
                            "epoch": epoch + 1,
                            "batch": i + 1,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        })

                        print(
                            f"Epoch {epoch+1} | Batch {i+1}/{len(dataloader_train)} "
                            f"| Loss {avg_loss_interval:.4f}"
                        )

                        # reset interval counters
                        running_loss_interval = 0.0
                        running_loss_joints_interval = 0.0
                        running_loss_pixel_interval = 0.0

                    # Debug gradient
                    if VERBOSE > 1:
                        for name, param in ee_model.named_parameters():
                            if param.grad is not None:
                                print(f"Parameter: {name} has gradient.")
                            else:
                                print(f"Parameter: {name}, no gradient computed or parameter unused.")

                    if (i + 1) % accumulation_steps == 0 or (i + 1 == len(dataloader_train)):
                        optimizer.step()
                        optimizer.zero_grad()

                    img_total += images.size(0)

                    if COMPUTE_ERROR_IN_TRAINING:
                        # Predictions
                        phi =torch.atan2(base_joint_sincos[:, 0], base_joint_sincos[:, 1])
                        theta_pred_rad = 0.5 * phi
                        theta_pred_rad = torch.remainder(theta_pred_rad, torch.pi)
                        theta_pred_deg = torch.rad2deg(theta_pred_rad)
                        theta_gt_deg = joint_values['base_joint'].to(device).to(torch.float32)

                        base_x_preds = base_x
                        base_y_preds = base_y

                        angular_error_base_joint = ang_error_deg_period180(theta_pred_deg, theta_gt_deg)
                        angular_error_base_joint_total += torch.abs(angular_error_base_joint).sum().item()

                        x_error = torch.abs(base_x_preds - base_x_gt)
                        y_error = torch.abs(base_y_preds - base_y_gt)
                        x_error_bin_total += x_error.sum().item()
                        y_error_bin_total += y_error.sum().item()
                        angular_error_interval += torch.abs(angular_error_base_joint).sum().item()
                        x_error_interval += x_error.sum().item()
                        y_error_interval += y_error.sum().item()
                        samples_interval += images.size(0)

                        if DEBUG > 0:
                            if (i % 5 == 0) and (images.size(0) > 0):
                                debug_dir = os.path.join(checkpoint_dir, "debug_samples")
                                os.makedirs(debug_dir, exist_ok=True)
                                save_debug_image(
                                    images[0],   # first image in batch
                                    {k: v[0].item() if hasattr(v, 'numpy') or torch.is_tensor(v) else v
                                    for k, v in joint_values.items()},
                                    os.path.join(debug_dir, f"train_epoch{epoch+1}_batch{i+1}.png"),
                                    pred_x=base_x_preds[0].item() / xy_bin_nbr * images.shape[3],   # convert normalized back to pixels
                                    pred_y=base_y_preds[0].item() / xy_bin_nbr * images.shape[2],
                                    pred_angle=theta_pred_deg[0].item()
                                )

                epoch_loss = running_loss / img_total
                running_loss_joints /= img_total
                running_loss_pixel /= img_total
                mean_ae_base_joint_train = angular_error_base_joint_total / img_total
                mean_x_error_bin_train = x_error_bin_total / img_total
                mean_y_error_bin_train = y_error_bin_total / img_total
                train_average_error = ((mean_ae_base_joint_train / config.num_classes) + (mean_x_error_bin_train / xy_bin_nbr) + (mean_y_error_bin_train / xy_bin_nbr)) / 3.0

                scheduler.step()

                angular_error_base_joint_total = 0
                x_error_bin_total = 0
                y_error_bin_total = 0
                mean_ae_base_joint_val = 0
                mean_x_error_bin_val = 0
                mean_y_error_bin_val = 0
                if RUN_VALIDATION:
                    # Validation step
                    ee_model.eval()
                    total_img_count = 0
                    with torch.no_grad():
                        for i, (images, joint_values, image_names) in enumerate(tqdm.tqdm(dataloader_val, desc=f"Validation Epoch {epoch+1}/{config.epochs}")):
                            total_img_count += images.size(0)

                            theta_rad = torch.deg2rad(joint_values['base_joint'].to(device).to(torch.float32))
                            base_joint_sin_gt = torch.sin(2.0 * theta_rad)
                            base_joint_cos_gt = torch.cos(2.0 * theta_rad)
                            base_x_gt = joint_values['x'].to(device) / 100.0
                            base_y_gt = joint_values['y'].to(device) / 100.0
                            base_joint_gt = torch.stack([base_joint_sin_gt, base_joint_cos_gt], dim=-1)
                            base_x_gt = base_x_gt.to(torch.float32)
                            base_y_gt = base_y_gt.to(torch.float32)
                            base_joint_gt = base_joint_gt.to(torch.float32)

                            images = images.to(device)

                            base_joint_sincos, base_x, base_y = ee_model(images)

                            base_x_preds = base_x
                            base_y_preds = base_y
                            
                            base_joint_sincos = normalize_2d(base_joint_sincos)
                            # Predictions
                            phi = torch.atan2(base_joint_sincos[:, 0], base_joint_sincos[:, 1])
                            theta_pred_rad = 0.5 * phi
                            theta_pred_rad = torch.remainder(theta_pred_rad, torch.pi)
                            theta_pred_deg = torch.rad2deg(theta_pred_rad)
                            theta_gt_deg = joint_values['base_joint'].to(device).to(torch.float32)
                            angular_error_base_joint = ang_error_deg_period180(theta_pred_deg, theta_gt_deg)
                            angular_error_base_joint_total += torch.abs(angular_error_base_joint).sum().item()

                            x_error_bin = torch.abs(base_x_preds - base_x_gt)
                            y_error_bin = torch.abs(base_y_preds - base_y_gt)

                            x_error_bin_total += x_error_bin.sum().item()
                            y_error_bin_total += y_error_bin.sum().item()
                            if VERBOSE > 0:
                                print(f"Validation batch processed: {images.size(0)} images.")
                                print(f"x_error_bin: {x_error_bin_total}, y_error_bin: {y_error_bin_total}")

                            if DEBUG > 0:
                                if (i % 5 == 0) and (images.size(0) > 0):
                                    debug_dir = os.path.join(checkpoint_dir, "debug_samples")
                                    os.makedirs(debug_dir, exist_ok=True)
                                    save_debug_image(
                                        images[0],   # first image in batch
                                        {k: v[0].item() if hasattr(v, 'numpy') or torch.is_tensor(v) else v
                                        for k, v in joint_values.items()},
                                        os.path.join(debug_dir, f"val_epoch{epoch+1}_batch{i+1}.png"),
                                        pred_x=base_x_preds[0].item() / xy_bin_nbr * images.shape[3],   # convert normalized back to pixels
                                        pred_y=base_y_preds[0].item() / xy_bin_nbr * images.shape[2],
                                        pred_angle=angular_error_base_joint[0].item()
                                    )

                    print(f"Validation: {total_img_count} images processed.")
                    print(f"x_error_bin_total: {x_error_bin_total}, y_error_pixel_total: {y_error_bin_total}")
                    mean_ae_base_joint_val = angular_error_base_joint_total / len(dataloader_val.dataset)
                    mean_x_error_bin_val = x_error_bin_total / len(dataloader_val.dataset)
                    mean_y_error_bin_val = y_error_bin_total / len(dataloader_val.dataset)

                print(f"[Epoch {epoch+1}/{config.epochs}] Loss: {epoch_loss:.4f} | base_joint AE: {mean_ae_base_joint_val:.4f}")
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "epoch": epoch + 1,
                    "learning_rate": current_lr,
                    "train_angular_error_joint3": mean_ae_base_joint_train,
                    "train_x_error_pixel": mean_x_error_bin_train,
                    "train_y_error_pixel": mean_y_error_bin_train,
                    "loss": epoch_loss,
                    "loss_joints": running_loss_joints,
                    "loss_pixel": running_loss_pixel,
                    "val_angular_error_joint3": mean_ae_base_joint_val,
                    "val_x_error_pixel": mean_x_error_bin_val,
                    "val_y_error_pixel": mean_y_error_bin_val,
                    "train_average_error": train_average_error
                })
                
                # Save model checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ee_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'dinov2_model': config.backbone,
                    'num_classes': config.num_classes,
                }, checkpoint_path)
    finally:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train EndEffectorPosePrediction Model with Token-based Architecture")
    parser.add_argument("--sweep", type=str, help="Sweep ID to use for hyperparameter optimization", required=True)
    args = parser.parse_args()
    sweep_id = args.sweep
    print(sweep_id)

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        try:
            with open("/run/wandb_api_key.txt", "r") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            print("API key not found")
            exit(1)
    if api_key:
        wandb.login(key=api_key)
        print("Logged into wandb successfully.")
    else:
        print("Could not login to wandb. Exiting.")
        raise SystemExit(1)

    wandb.agent(sweep_id, train)