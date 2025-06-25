import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from PIL import Image
import os
import wandb
import math
import random
import datetime
import tqdm

import eefdataset
import model_token
from torch.utils.data import DataLoader, SubsetRandomSampler

# Configuration
TRAIN_IMAGE_FOLDER_PATH = "/home/wilah/workspace/EndEffectorTrackingNN/image_subset"
TRAIN_JOINT_GT_FILE_PATH = "/home/wilah/workspace/EndEffectorTracking/output_ground_truth_202504/joint_values.csv"
TRAIN_XY_GT_FILE_PATH = "/home/wilah/workspace/EndEffectorTracking/output_ground_truth_202504/center_piece_pixel_coordinates.csv"  # Path to the ground truth file for XY coordinates
VAL_IMAGE_FOLDER_PATH = "/home/wilah/workspace/SVO_processing/output/20250605-for_william./rgb/left"
VAL_JOINT_GT_FILE_PATH = "/home/wilah/workspace/EndEffectorTracking/output_ground_truth_202506/joint_values.csv"
VAL_XY_GT_FILE_PATH = "/home/wilah/workspace/EndEffectorTracking/output_ground_truth_202506/center_piece_pixel_coordinates.csv"  # Path to the ground truth file for XY coordinates
GT_PRECISION = 1 # Degrees that the GT will be rounded to (e.g. if set to 5, then a GT of 12 will be set to 10)
EPOCHS = 25
LEARNING_RATE = [1e-2, 1e-3, 1e-4, 1e-5]
BACKBONE = 'dinov2_vitb14_reg'
MLP_DIM = 256
NUM_CLASSES = math.ceil(360 / GT_PRECISION)
BATCH_SIZE = [1, 2, 4]
RANDOM_SEED = 42
LABEL_SMOOTHING_MAX = 0.5
LABEL_SMOOTHING_MIN = 0
LR_DECAY_POWER = 0.95
REDUCE_LABEL_SMOOTHING_MIN = 0.1
REDUCE_LABEL_SMOOTHING_MAX = 0.25
SAMPLES_PER_CLASS = 3
LEFT_CROPPING = 398
RIGHT_CROPPING = 856
DEBUG = 0
WEIGHT_LOSS_JOINTS = 1.0
WEIGHT_LOSS_XY = 0.01

sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "model_type": {"value": "token"},
        "batch_size": {"values": BATCH_SIZE},
        "ground_truth_precision": {"value": GT_PRECISION},
        "epochs": {"value": EPOCHS},
        "random_seed": {"value": RANDOM_SEED},
        "backbone": {"value": BACKBONE},
        "mlp_dim": {"value": MLP_DIM},
        "num_classes": {"value": NUM_CLASSES},
        "train_image_folder_path": {"value": TRAIN_IMAGE_FOLDER_PATH},
        "train_joint_ground_truth_file_path": {"value": TRAIN_JOINT_GT_FILE_PATH},
        "train_xy_ground_truth_file_path": {"value": TRAIN_XY_GT_FILE_PATH},
        "val_image_folder_path": {"value": VAL_IMAGE_FOLDER_PATH},
        "val_joint_ground_truth_file_path": {"value": VAL_JOINT_GT_FILE_PATH},
        "val_xy_ground_truth_file_path": {"value": VAL_XY_GT_FILE_PATH},
        "learning_rate": {"values": LEARNING_RATE},
        "label_smoothing": {"distribution": "uniform", "max": LABEL_SMOOTHING_MAX, "min": LABEL_SMOOTHING_MIN},
        "lr_decay_power": {"value": LR_DECAY_POWER},
        "reduce_label_smoothing": {"distribution": "uniform", "max": REDUCE_LABEL_SMOOTHING_MAX, "min": REDUCE_LABEL_SMOOTHING_MIN},
        "samples_per_class": {"value": SAMPLES_PER_CLASS},
        "left_cropping": {"value": LEFT_CROPPING},
        "right_cropping": {"value": RIGHT_CROPPING},
        "top_cropping": {"value": 1},
        "bottom_cropping": {"value": 2},
        "weight_loss_joints": {"value": WEIGHT_LOSS_JOINTS},
        "weight_loss_xy": {"value": WEIGHT_LOSS_XY},
    },
}

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
    with wandb.init(config=config):
        config = wandb.config
        random.seed(config.random_seed)
        transform = T.Compose([T.Lambda(lambda img: TF.rotate(img, 180)),
                               T.Lambda(lambda img: TF.crop(img, top=config.top_cropping, left=config.left_cropping, height=img.height - config.bottom_cropping, width=img.width - config.right_cropping)),
                            T.ToTensor()])
        dataset_train = eefdataset.EEFDataset(image_dir=config.train_image_folder_path, joint_csv_path=config.train_joint_ground_truth_file_path,
                                        xy_csv_path=config.train_xy_ground_truth_file_path, joint_precision=1, transform=transform)
        dataset_val = eefdataset.EEFDataset(image_dir=config.val_image_folder_path, joint_csv_path=config.val_joint_ground_truth_file_path,
                                        xy_csv_path=config.val_xy_ground_truth_file_path, joint_precision=1, transform=transform)
        dataset_train.save_to_csv("tmp/dataset.csv")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load DINOv2
        backbone_model = torch.hub.load('facebookresearch/dinov2', config['backbone']).to(device)
        for param in backbone_model.parameters():
            param.requires_grad = True

        # Load model
        ee_model = model_token.EndEffectorPosePredToken(backbone_model, num_classes=config.num_classes).to(device)
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

        for epoch in range(config.epochs):
            # Get indices
            indices = get_balanced_indices(dataset_train, config.samples_per_class)

            # Sampler for balanced loading
            sampler = SubsetRandomSampler(indices)

            # Create DataLoader
            dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, sampler=sampler)
            dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False)

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

                #_, _, j3_logits, _, base_x, base_y = ee_model(images)
                j3_logits = ee_model(images)

                # Compute loss
                loss_joints = criterion_joints(j3_logits, target_base)
                
                # Reshape base_x and base_y to match target_x and target_y
                target_x = target_x.view(-1, 1)
                target_y = target_y.view(-1, 1)
                #loss_pixel = criterion_pixel(base_x, target_x) + criterion_pixel(base_y, target_y)
                
                loss = (loss_joints * config.weight_loss_joints) #+ (loss_pixel * config.weight_loss_xy)
                running_loss += loss.item()
                running_loss_joints += loss_joints.item()
                #running_loss_pixel += loss_pixel.item()
                
                loss.backward()

                # Debug gradient
                if DEBUG > 0:
                    for name, param in ee_model.named_parameters():
                        if param.grad is not None:
                            print(f"Parameter: {name} has gradient.")
                        else:
                            print(f"Parameter: {name}, no gradient computed or parameter unused.")

                optimizer.step()

                img_total += images.size(0)

                # Predictions
                j3_preds = j3_logits.argmax(dim=1)

                # Convert to angles
                j3_pred_angles = eefdataset.class_to_angle(j3_preds, config.ground_truth_precision)

                # Compute angular error
                angular_error_j3 = torch.abs(j3_pred_angles - joint_values['base_joint'].to(device))

                # Accumulate for epoch
                angular_error_j3_total += angular_error_j3.sum().item()

                x_error_pixel = torch.abs(target_x - target_x)#torch.abs(base_x - target_x)
                y_error_pixel = torch.abs(target_y - target_y)#torch.abs(base_y - target_y)
                x_error_pixel_total += x_error_pixel.sum().item()
                y_error_pixel_total += y_error_pixel.sum().item()


            epoch_loss = running_loss
            mean_ae_j3_train = angular_error_j3_total / img_total
            mean_x_error_pixel_train = x_error_pixel_total / img_total
            mean_y_error_pixel_train = y_error_pixel_total / img_total

            scheduler.step()

            angular_error_j3_total = 0
            x_error_pixel_total = 0
            y_error_pixel_total = 0

            # Validation step
            ee_model.eval()
            total_img_count = 0
            with torch.no_grad():
                for images, joint_values in tqdm.tqdm(dataloader_val, desc=f"Validation Epoch {epoch+1}/{config.epochs}"):
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

                    #_, _, j3_logits, _, base_x, base_y = ee_model(images)
                    j3_logits = ee_model(images)

                    j3_preds = j3_logits.argmax(dim=1)
                    j3_pred_angles = eefdataset.class_to_angle(j3_preds, GT_PRECISION)
                    j3_gt_angles = eefdataset.class_to_angle(target_base, GT_PRECISION)
                    # Fix the angular error calculation because right now if we predict 0 and GT is 179, we get 179 which is not correct as the error should be 1
                    angular_error_j3 = torch.abs(j3_pred_angles - j3_gt_angles)
                    angular_error_j3_total += angular_error_j3.sum().item()

                    target_x = target_x.view(-1, 1)
                    target_y = target_y.view(-1, 1)
                    x_error_pixel = torch.abs(target_x - target_x)#torch.abs(base_x - target_x)
                    y_error_pixel = torch.abs(target_y - target_y)#torch.abs(base_y - target_y)

                    if torch.any(x_error_pixel > 1) or torch.any(y_error_pixel > 1):
                        print(f"Warning: Pixel error exceeds 1 in validation for batch size {config.batch_size}.")
                        print(f"x_error_pixel: {x_error_pixel}, y_error_pixel: {y_error_pixel}")

                    x_error_pixel_total += x_error_pixel.sum().item()
                    y_error_pixel_total += y_error_pixel.sum().item()
                    if DEBUG > 0:
                        print(f"Validation batch processed: {images.size(0)} images.")
                        print(f"x_error_pixel: {x_error_pixel_total}, y_error_pixel: {y_error_pixel_total}")

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
                "total_loss": epoch_loss,
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
                'mlp_dim': config.mlp_dim,
                'num_classes': config.num_classes
            }, checkpoint_path)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="EndEffectorPosePred")
    wandb.agent(sweep_id, train, count=20)