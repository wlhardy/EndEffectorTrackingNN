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
import wandb
import math
import random
import datetime
import tqdm
import multiprocessing
from torch.utils.data import DataLoader, SubsetRandomSampler

import eefdataset
import model_token

matplotlib.use("Agg")

MAIN_TRAIN_FOLDER = "/home/wilah/workspace/EndEffectorTrackingNN/datasets/train"
VAL_TRAIN_FOLDER = "/home/wilah/workspace/EndEffectorTrackingNN/datasets/val"
GT_PRECISION = 1 # Degrees that the GT will be rounded to (e.g. if set to 5, then a GT of 12 will be set to 10)
EPOCHS = 40
LEARNING_RATE = [1e-7, 1e-8]
BACKBONE = 'dinov2_vitb14_reg'
NUM_CLASSES = math.ceil(360 / GT_PRECISION)
RANDOM_SEED = 42
LABEL_SMOOTHING_MAX = 0.05
LABEL_SMOOTHING_MIN = 0.0
LR_DECAY_POWER = [0.9, 0.95, 0.99, 1.0]
REDUCE_LABEL_SMOOTHING_MIN = 0.0
REDUCE_LABEL_SMOOTHING_MAX = 0.05
SAMPLES_PER_CLASS = 3
LEFT_CROPPING = 398
RIGHT_CROPPING = 856
DEBUG = 1
VERBOSE = 0
WEIGHT_LOSS_JOINTS = 1.0
WEIGHT_LOSS_XY = 1.0
FREEZE_BLOCKS = [0, 2, 4, 6]  # Max number of blocks in the backbone is 11
NBR_TOKEN_PER_TASK = [1]
MAX_BATCH_SIZE = 2
TARGET_BATCH_SIZE = [2]
FREEZE_POS_EMBED = True
FREEZE_PATCH_EMBED = True
RUN_VALIDATION = True

sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "model_type": {"value": "token"},
        "max_batch_size": {"value": MAX_BATCH_SIZE},
        "ground_truth_precision": {"value": GT_PRECISION},
        "epochs": {"value": EPOCHS},
        "random_seed": {"value": RANDOM_SEED},
        "backbone": {"value": BACKBONE},
        "num_classes": {"value": NUM_CLASSES},
        "train_main_folder_path": {"value": MAIN_TRAIN_FOLDER},
        "val_main_folder_path": {"value": VAL_TRAIN_FOLDER},
        "learning_rate": {"values": LEARNING_RATE},
        "label_smoothing": {"distribution": "uniform", "max": LABEL_SMOOTHING_MAX, "min": LABEL_SMOOTHING_MIN},
        "lr_decay_power": {"values": LR_DECAY_POWER},
        "reduce_label_smoothing": {"distribution": "uniform", "max": REDUCE_LABEL_SMOOTHING_MAX, "min": REDUCE_LABEL_SMOOTHING_MIN},
        "samples_per_class": {"value": SAMPLES_PER_CLASS},
        "left_cropping": {"value": LEFT_CROPPING},
        "right_cropping": {"value": RIGHT_CROPPING},
        "top_cropping": {"value": 1},
        "bottom_cropping": {"value": 2},
        "weight_loss_joints": {"value": WEIGHT_LOSS_JOINTS},
        "weight_loss_xy": {"value": WEIGHT_LOSS_XY},
        "target_batch_size": {"values": TARGET_BATCH_SIZE},
        "freeze_blocks": {"values": FREEZE_BLOCKS},
        "nbr_token_per_task": {"values": NBR_TOKEN_PER_TASK},
        "freeze_pos_embed": {"value": FREEZE_POS_EMBED},
        "freeze_patch_embed": {"value": FREEZE_PATCH_EMBED},
    },
}

def save_debug_image(image_tensor, joint_values, save_path,
                     pred_x=None, pred_y=None, pred_angle=None):
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
    
    # Overlay GT point
    ax.scatter(joint_values['x'], joint_values['y'], c='red', s=40, marker='x', label="GT")

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
                                            xy_csv_paths=train_xy_csvs, joint_precision=1, transform=transform_train)
            
            transform_val = T.Compose([T.Lambda(lambda img: TF.rotate(img, 180)),
                                    T.Lambda(lambda img: TF.crop(img, top=config.top_cropping, left=config.left_cropping, height=img.height - config.bottom_cropping, width=img.width - config.right_cropping)),
                                    T.Lambda(half_pixels_resize_and_pad),
                                    T.ToTensor()])
            val_image_dirs, val_joint_csvs, val_xy_csvs = discover_dataset_folders(config.val_main_folder_path)
            dataset_val = eefdataset.EEFDataset(image_dirs=val_image_dirs, joint_csv_paths=val_joint_csvs,
                                            xy_csv_paths=val_xy_csvs, joint_precision=1, transform=transform_val)
            
            dataset_train.save_to_csv("tmp/train_dataset.csv")
            dataset_val.save_to_csv("tmp/val_dataset.csv")

            end_time = datetime.datetime.now()
            time_to_build = end_time - start_time
            print(f"Time to build datasets: {time_to_build}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            # Load DINOv2
            backbone_model = torch.hub.load('facebookresearch/dinov2', config['backbone']).to(device)
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

            # Load model
            ee_model = model_token.EndEffectorPosePredToken(backbone_model, num_classes=config.num_classes, nbr_tokens=config.nbr_token_per_task).to(device)
            optimizer = torch.optim.AdamW(ee_model.parameters(), lr=config.learning_rate)
            
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer,
                                                            total_iters=config.epochs,
                                                            power=config.lr_decay_power)

            criterion_joints = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
            criterion_pixel = nn.MSELoss()

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
            train_cpu_count = min(16, cpu_count)  # Limit to 8 workers to avoid overloading the system
            val_cpu_count = min(8, cpu_count)  # Validation can be lighter
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=train_cpu_count, shuffle=True, persistent_workers=True)

            if RUN_VALIDATION:
                dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=val_cpu_count, persistent_workers=True)

            for epoch in range(config.epochs):
                ee_model.train()
                running_loss = 0
                running_loss_joints = 0
                running_loss_pixel = 0
                img_total = 0
                angular_error_j3_total = 0
                x_error_pixel_total = 0
                y_error_pixel_total = 0

                for i, (images, joint_values) in enumerate(tqdm.tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{config.epochs}")):
                    # Reset the gradients
                    optimizer.zero_grad()

                    if DEBUG > 2:
                        # Save all images in the batch to disk for debugging
                        for j in range(images.size(0)):
                            img = images[j].cpu().numpy().transpose(1, 2, 0)
                            img = (img * 255).astype(np.uint8)
                            img = Image.fromarray(img)
                            img_path = os.path.join(checkpoint_dir, f"debug_image_epoch{epoch+1}_batch{i+1}_img{j+1}.png")
                            img.save(img_path)
                        
                    base_joint_angles = joint_values['base_joint'].numpy()
                    x = joint_values['x'].numpy()
                    y = joint_values['y'].numpy()
                    target_base = torch.tensor([
                        eefdataset.quantize_joint(base, config.ground_truth_precision, symmetric=True) for base in base_joint_angles
                    ], dtype=torch.long).to(device)

                    if torch.any(target_base < 0) or torch.any(target_base >= (config.num_classes / 2)):
                        print("Invalid target detected!")
                        print(f"target_base: {target_base}")
                        print(f"num_classes: {config.num_classes}")
                        exit()

                    images = images.to(device)

                    # Get image width and height
                    image_width, image_height = images.shape[2], images.shape[3]

                    target_x = torch.tensor([
                        (x_val / image_width) for x_val in x
                    ], dtype=torch.float32).to(device)
                    target_y = torch.tensor([
                        (y_val / image_height) for y_val in y
                    ], dtype=torch.float32).to(device)

                    #_, _, j3_logits, _, base_x, base_y = ee_model(images)
                    j3_logits, base_x, base_y = ee_model(images)

                    # Check if j3_logits is within class range
                    if torch.any(j3_logits.argmax(dim=1) < 0) or torch.any(j3_logits.argmax(dim=1) >= (config.num_classes / 2)):
                        print("Invalid j3_logits detected!")
                        print(f"j3_logits: {j3_logits}")
                        print(f"num_classes: {config.num_classes}")
                        exit()

                    # Compute loss
                    loss_joints = criterion_joints(j3_logits, target_base)
                    
                    # Reshape base_x and base_y to match target_x and target_y
                    target_x = target_x.view(-1, 1)
                    target_y = target_y.view(-1, 1)
                    loss_pixel = criterion_pixel(base_x, target_x) + criterion_pixel(base_y, target_y)
                    
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

                    # Predictions
                    j3_preds = j3_logits.argmax(dim=1)

                    # Convert to angles
                    j3_pred_angles = eefdataset.class_to_angle(j3_preds, config.ground_truth_precision, symmetric=True)
                    j3_gt_angles = eefdataset.class_to_angle(target_base, config.ground_truth_precision, symmetric=True)
                    if VERBOSE > 0:
                        print(f"j3_pred_angles: {j3_pred_angles}")
                        print(f"target_base: {target_base}")
                        print(f"j3_gt_angles: {j3_gt_angles}")

                    # Compute angular error
                    angular_error_j3 = torch.abs(j3_pred_angles - j3_gt_angles)

                    # Fix the angular error calculation because right now if we predict 0 and GT is 179, we get 179 which is not correct as the error should be 1
                    angular_error_j3 = torch.where(angular_error_j3 > 90, 180 - angular_error_j3, angular_error_j3)

                    # Accumulate for epoch
                    angular_error_j3_total += angular_error_j3.sum().item()

                    x_error_pixel = torch.abs(base_x - target_x)
                    y_error_pixel = torch.abs(base_y - target_y)
                    x_error_pixel_total += x_error_pixel.sum().item()
                    y_error_pixel_total += y_error_pixel.sum().item()

                    if DEBUG > 0:
                        if (i % 5 == 0) and (images.size(0) > 0):
                            debug_dir = os.path.join(checkpoint_dir, "debug_samples")
                            os.makedirs(debug_dir, exist_ok=True)
                            save_debug_image(
                                images[0],   # first image in batch
                                {k: v[0].item() if hasattr(v, 'numpy') or torch.is_tensor(v) else v
                                for k, v in joint_values.items()},
                                os.path.join(debug_dir, f"train_epoch{epoch+1}_batch{i+1}.png"),
                                pred_x=base_x[0].item() * images.shape[3],   # convert normalized back to pixels
                                pred_y=base_y[0].item() * images.shape[2],
                                pred_angle=j3_pred_angles[0].item()
                            )


                epoch_loss = running_loss / img_total
                running_loss_joints /= img_total
                running_loss_pixel /= img_total
                mean_ae_j3_train = angular_error_j3_total / img_total
                mean_x_error_pixel_train = x_error_pixel_total / img_total
                mean_y_error_pixel_train = y_error_pixel_total / img_total

                scheduler.step()

                angular_error_j3_total = 0
                x_error_pixel_total = 0
                y_error_pixel_total = 0
                mean_ae_j3_val = 0
                mean_x_error_pixel_val = 0
                mean_y_error_pixel_val = 0
                if RUN_VALIDATION:
                    # Validation step
                    ee_model.eval()
                    total_img_count = 0
                    with torch.no_grad():
                        for i, (images, joint_values) in enumerate(tqdm.tqdm(dataloader_val, desc=f"Validation Epoch {epoch+1}/{config.epochs}")):
                            total_img_count += images.size(0)
                            base_joint_angles = joint_values['base_joint'].numpy()
                            x = joint_values['x'].numpy()
                            y = joint_values['y'].numpy()
                            target_base = torch.tensor([
                                eefdataset.quantize_joint(base, config.ground_truth_precision, symmetric=True) for base in base_joint_angles
                            ], dtype=torch.long).to(device)

                            images = images.to(device)

                            # Get image width and height
                            image_width, image_height = images.shape[2], images.shape[3]

                            target_x = torch.tensor([
                                (x_val / image_width) for x_val in x
                            ], dtype=torch.float32).to(device)
                            target_y = torch.tensor([
                                (y_val / image_height) for y_val in y
                            ], dtype=torch.float32).to(device)

                            if torch.any(target_x > 1) or torch.any(target_y > 1):
                                print(f"Warning: Target pixel coordinates exceed 1 in validation.")
                                print(f"target_x: {target_x}, target_y: {target_y}")

                            j3_logits, base_x, base_y = ee_model(images)

                            j3_preds = j3_logits.argmax(dim=1)
                            j3_pred_angles = eefdataset.class_to_angle(j3_preds, config.ground_truth_precision, symmetric=True)
                            j3_gt_angles = eefdataset.class_to_angle(target_base, config.ground_truth_precision, symmetric=True)
                            
                            angular_error_j3 = torch.abs(j3_pred_angles - j3_gt_angles)
                            angular_error_j3 = torch.where(angular_error_j3 > 90, 180 - angular_error_j3, angular_error_j3)
                            angular_error_j3_total += angular_error_j3.sum().item()

                            target_x = target_x.view(-1, 1)
                            target_y = target_y.view(-1, 1)
                            x_error_pixel = torch.abs(base_x - target_x)
                            y_error_pixel = torch.abs(base_y - target_y)

                            if torch.any(x_error_pixel > 1) or torch.any(y_error_pixel > 1):
                                print(f"Warning: Pixel error exceeds 1 in validation for batch size {batch_size}.")
                                print(f"x_error_pixel: {x_error_pixel}, y_error_pixel: {y_error_pixel}")

                            x_error_pixel_total += x_error_pixel.sum().item()
                            y_error_pixel_total += y_error_pixel.sum().item()
                            if VERBOSE > 0:
                                print(f"Validation batch processed: {images.size(0)} images.")
                                print(f"x_error_pixel: {x_error_pixel_total}, y_error_pixel: {y_error_pixel_total}")

                            if DEBUG > 0:
                                if (i % 5 == 0) and (images.size(0) > 0):
                                    debug_dir = os.path.join(checkpoint_dir, "debug_samples")
                                    os.makedirs(debug_dir, exist_ok=True)
                                    save_debug_image(
                                        images[0],   # first image in batch
                                        {k: v[0].item() if hasattr(v, 'numpy') or torch.is_tensor(v) else v
                                        for k, v in joint_values.items()},
                                        os.path.join(debug_dir, f"val_epoch{epoch+1}_batch{i+1}.png"),
                                        pred_x=base_x[0].item() * images.shape[3],   # convert normalized back to pixels
                                        pred_y=base_y[0].item() * images.shape[2],
                                        pred_angle=j3_pred_angles[0].item()
                                    )

                    print(f"Validation: {total_img_count} images processed.")
                    print(f"x_error_pixel_total: {x_error_pixel_total}, y_error_pixel_total: {y_error_pixel_total}")
                    mean_ae_j3_val = angular_error_j3_total / len(dataloader_val.dataset)
                    mean_x_error_pixel_val = x_error_pixel_total / len(dataloader_val.dataset)
                    mean_y_error_pixel_val = y_error_pixel_total / len(dataloader_val.dataset)

                print(f"[Epoch {epoch+1}/{config.epochs}] Loss: {epoch_loss:.4f} | J3 AE: {mean_ae_j3_train:.4f}")
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "epoch": epoch + 1,
                    "learning_rate": current_lr,
                    "label_smoothing": current_label_smoothing,
                    "train_angular_error_joint3": mean_ae_j3_train,
                    "train_x_error_pixel": mean_x_error_pixel_train,
                    "train_y_error_pixel": mean_y_error_pixel_train,
                    "loss": epoch_loss,
                    "loss_joints": running_loss_joints,
                    "loss_pixel": running_loss_pixel,
                    "val_angular_error_joint3": mean_ae_j3_val,
                    "val_x_error_pixel": mean_x_error_pixel_val,
                    "val_y_error_pixel": mean_y_error_pixel_val
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
                    'num_token_per_class' : config.nbr_token_per_task,
                }, checkpoint_path)
    finally:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="EndEffectorPosePred")
    wandb.agent(sweep_id, train, count=40)