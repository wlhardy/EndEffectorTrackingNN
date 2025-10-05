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
import model_token

matplotlib.use("Agg")

DEBUG = 0
VERBOSE = 0
COMPUTE_ERROR_IN_TRAINING = False
RUN_VALIDATION = True


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

def half_pixels_resize_and_pad(img, s=1/math.sqrt(2)):
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
            # Log the time to build the datasets
            start_time = datetime.datetime.now()
            transform_train = T.Compose([T.Lambda(lambda img: TF.rotate(img, 180)),
                                        T.Lambda(lambda img: TF.crop(img, top=config.top_cropping, left=config.left_cropping, height=img.height - config.bottom_cropping, width=img.width - config.right_cropping)),
                                        #T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
                                        #T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
                                        #T.RandomInvert(),
                                        #T.RandomPosterize(bits=4),
                                        #T.RandomSolarize(threshold=192),
                                        #T.RandomAdjustSharpness(sharpness_factor=2),
                                        #T.RandomAutocontrast(),
                                        #T.RandomEqualize(),
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

            # Load DINOv2
            backbone_model = torch.hub.load('facebookresearch/dinov2', config['backbone'], force_reload=False)
            for param in backbone_model.parameters():
                param.requires_grad = True

            num_blocks_to_freeze = config.freeze_blocks
            for i, block in enumerate(backbone_model.blocks):
                if i < num_blocks_to_freeze:
                    for param in block.parameters():
                        param.requires_grad = False
            
            if config.freeze_pos_embed:
                print("Freezing position embedding.")
                backbone_model.pos_embed.requires_grad = False

            if config.freeze_patch_embed:
                print("Freezing patch embedding.")
                backbone_model.patch_embed.requires_grad = False

            xy_bin_nbr = config.xy_bin_nbr

            # Load model
            ee_model = model_token.EndEffectorPosePredToken(backbone_model, num_classes_joint=config.num_classes, nbr_classes_xy=xy_bin_nbr).to(device)
            optimizer = torch.optim.AdamW(ee_model.parameters(), lr=config.learning_rate)
            
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer,
                                                            total_iters=config.epochs,
                                                            power=config.lr_decay_power)

            criterion_joints = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
            criterion_pixel = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

            # Create a directory to log checkpoints and results
            os.makedirs("training", exist_ok=True)

            # Use the date and time to create a unique directory
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join("training", f"checkpoint_{timestamp}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            current_label_smoothing = config.label_smoothing
            target_batch_size = config.target_batch_size
            batch_size = min(config.max_batch_size, target_batch_size)
            accumulation_steps = max(1, target_batch_size // batch_size)

            cpu_count = multiprocessing.cpu_count()
            train_cpu_count = min(16, cpu_count)
            val_cpu_count = min(16, cpu_count)
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=train_cpu_count, shuffle=True, persistent_workers=True)

            if RUN_VALIDATION:
                dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=val_cpu_count, persistent_workers=True)

            for epoch in range(config.epochs):
                ee_model.train()
                running_loss = 0
                running_loss_joints = 0
                running_loss_pixel = 0
                img_total = 0
                angular_error_base_joint_total = 0
                x_error_bin_total = 0
                y_error_bin_total = 0

                for i, (images, joint_values) in enumerate(tqdm.tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{config.epochs}")):
                    # Start a timer to measure the training step duration
                    step_start_time = time.time()
                    # Reset the gradients
                    optimizer.zero_grad()
                    base_joint_quant = joint_values['base_joint_quant'].to(device)
                    base_x = joint_values['x'].to(device)
                    base_y = joint_values['y'].to(device)

                    if DEBUG > 2:
                        # Save all images in the batch to disk for debugging
                        for j in range(images.size(0)):
                            img = images[j].cpu().numpy().transpose(1, 2, 0)
                            img = (img * 255).astype(np.uint8)
                            img = Image.fromarray(img)
                            img_path = os.path.join(checkpoint_dir, f"debug_image_epoch{epoch+1}_batch{i+1}_img{j+1}.png")
                            img.save(img_path)

                    images = images.to(device)

                    base_joint_logits, base_x_logits, base_y_logits = ee_model(images)

                    # Compute loss
                    loss_joints = criterion_joints(base_joint_logits, base_joint_quant)
                    loss_pixel = criterion_pixel(base_x_logits, base_x) + criterion_pixel(base_y_logits, base_y)
                    
                    loss = (loss_joints * config.weight_loss_joints) + (loss_pixel * config.weight_loss_xy)
                    running_loss += loss.item()
                    running_loss_joints += loss_joints.item()
                    running_loss_pixel += loss_pixel.item()
                    
                    loss.backward()

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
                        base_joint_preds = base_joint_logits.argmax(dim=1)
                        base_x_preds = base_x_logits.argmax(dim=1)
                        base_y_preds = base_y_logits.argmax(dim=1)

                        # Convert to angles
                        base_joint_pred_angles = eefdataset.class_to_angle(base_joint_preds, config.ground_truth_precision, symmetric=True)
                        base_joint_gt_angles = eefdataset.class_to_angle(base_joint_quant, config.ground_truth_precision, symmetric=True)
                        if VERBOSE > 0:
                            print(f"base_joint_pred_angles: {base_joint_pred_angles}")
                            print(f"target_base: {base_joint_quant}")
                            print(f"base_joint_gt_angles: {base_joint_gt_angles}")

                        # Compute angular error
                        angular_error_base_joint = torch.abs(base_joint_pred_angles - base_joint_gt_angles)

                        # Fix the angular error calculation because right now if we predict 0 and GT is 179, we get 179 which is not correct as the error should be 1
                        angular_error_base_joint = torch.where(angular_error_base_joint > 90, 180 - angular_error_base_joint, angular_error_base_joint)

                        # Accumulate for epoch
                        angular_error_base_joint_total += angular_error_base_joint.sum().item()

                        x_error_bin = torch.abs(base_x_preds - base_x)
                        y_error_bin = torch.abs(base_y_preds - base_y)
                        x_error_bin_total += x_error_bin.sum().item()
                        y_error_bin_total += y_error_bin.sum().item()

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
                                    pred_angle=base_joint_pred_angles[0].item()
                                )

                epoch_loss = running_loss / img_total
                running_loss_joints /= img_total
                running_loss_pixel /= img_total
                mean_ae_base_joint_train = angular_error_base_joint_total / img_total
                mean_x_error_bin_train = x_error_bin_total / img_total
                mean_y_error_bin_train = y_error_bin_total / img_total

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
                        for i, (images, joint_values) in enumerate(tqdm.tqdm(dataloader_val, desc=f"Validation Epoch {epoch+1}/{config.epochs}")):
                            total_img_count += images.size(0)

                            base_joint_quant = joint_values['base_joint_quant'].to(device)
                            base_x = joint_values['x'].to(device)
                            base_y = joint_values['y'].to(device)

                            images = images.to(device)

                            if torch.any(base_x > xy_bin_nbr) or torch.any(base_y > xy_bin_nbr):
                                print(f"Warning: Target pixel coordinates exceed xy_bin_nbr in validation.")
                                print(f"target_x: {base_x}, target_y: {base_y}")

                            base_joint_logits, base_x_logits, base_y_logits = ee_model(images)

                            base_joint_preds = base_joint_logits.argmax(dim=1)
                            base_x_preds = base_x_logits.argmax(dim=1)
                            base_y_preds = base_y_logits.argmax(dim=1)
                            base_joint_pred_angles = eefdataset.class_to_angle(base_joint_preds, config.ground_truth_precision, symmetric=True)
                            base_joint_gt_angles = eefdataset.class_to_angle(base_joint_quant, config.ground_truth_precision, symmetric=True)
                            
                            angular_error_base_joint = torch.abs(base_joint_pred_angles - base_joint_gt_angles)
                            angular_error_base_joint = torch.where(angular_error_base_joint > 90, 180 - angular_error_base_joint, angular_error_base_joint)
                            angular_error_base_joint_total += angular_error_base_joint.sum().item()

                            x_error_bin = torch.abs(base_x_preds - base_x)
                            y_error_bin = torch.abs(base_y_preds - base_y)

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
                                        pred_angle=base_joint_pred_angles[0].item()
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
                    "label_smoothing": current_label_smoothing,
                    "train_angular_error_joint3": mean_ae_base_joint_train,
                    "train_x_error_pixel": mean_x_error_bin_train,
                    "train_y_error_pixel": mean_y_error_bin_train,
                    "loss": epoch_loss,
                    "loss_joints": running_loss_joints,
                    "loss_pixel": running_loss_pixel,
                    "val_angular_error_joint3": mean_ae_base_joint_val,
                    "val_x_error_pixel": mean_x_error_bin_val,
                    "val_y_error_pixel": mean_y_error_bin_val
                })

                # Reduce label smoothing
                if current_label_smoothing > 0:
                    current_label_smoothing = current_label_smoothing - (config.label_smoothing * config.reduce_label_smoothing)
                    if current_label_smoothing < 0:
                        current_label_smoothing = 0
                    criterion_joints = nn.CrossEntropyLoss(label_smoothing=current_label_smoothing)
                
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