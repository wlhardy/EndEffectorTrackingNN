import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np

IMAGE_FOLDER_PATH = "/home/wilah/workspace/SVO_processing/output/20250605-for_william./rgb/left"
JOINT_GT_FILE_PATH = "/home/wilah/workspace/EndEffectorTracking/output_ground_truth/joint_values.csv"

def get_closest_joint_values(joint_dict, timestamp):
    closest_timestamp = min(joint_dict.keys(), key=lambda t: abs(t - timestamp))
    return joint_dict[closest_timestamp], closest_timestamp

def quantize_joint(joint_value, precision, symmetric=False):
    # Since the angle can be -180 to 180, we normalize it to 0 to 360
    angle = ((joint_value + 180) % 360)
    # For some applications, we might want to treat the angle as symmetric around 180 degrees
    if symmetric and angle >= 180:
        angle = angle - 180
    class_idx = int(round((angle) / precision))
    return class_idx

def class_to_angle(class_idx, precision):
    # Convert class index back to angle (-180 to 180 range)
    return class_idx * precision - 180

class RandomHorizontalFlipWithTarget:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1).item() < self.p:
            image = transforms.functional.hflip(image)
            target['lower_joint'] = -1 * target['lower_joint']
            target['base_joint'] = -1 * target['base_joint']
            
        return image, target

class EEFDataset(Dataset):
    def __init__(self, image_dir, joint_csv_path, xy_csv_path, joint_precision, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.joint_precision = joint_precision
        self.xy_csv_path = xy_csv_path
        self.data = []

        joint_csv_data = pd.read_csv(joint_csv_path)
        joint_dict = {
            row['timestamp']: {
                'telescope_joint': row['telescope_joint'],
                'upper_joint': row['upper_joint'],
                'lower_joint': row['lower_joint'],
                'base_joint': row['base_joint'],
                'gripper_joint': row['gripper_joint']
            }
            for _, row in joint_csv_data.iterrows()
        }

        xy_csv_data = pd.read_csv(xy_csv_path)
        xy_dict = {
            row['timestamp']: {
                'x': row['x'],
                'y': row['y']
            }
            for _, row in xy_csv_data.iterrows()
        }

        already_present = set()
        key_order = ['telescope_joint', 'upper_joint', 'lower_joint', 'base_joint', 'gripper_joint']
        # Pair images with closest joint values
        for file_name in sorted(os.listdir(image_dir)):
            if file_name.endswith(".png"):
                timestamp = int(os.path.splitext(file_name)[0])
                closest_joint_values, closest_timestamp = get_closest_joint_values(joint_dict, timestamp)
                img_path = os.path.join(image_dir, file_name)
                values_tuple = tuple(quantize_joint(float(closest_joint_values[key]),joint_precision) for key in key_order)
                if values_tuple not in already_present:
                    already_present.add(values_tuple)
                    # Load xy coordinates
                    if timestamp in xy_dict:
                        xy_values = xy_dict[timestamp]
                        for key in ['x', 'y']:
                            closest_joint_values[key] = xy_values[key]
                    else:
                        raise ValueError(f"Timestamp {timestamp} not found in xy coordinates CSV.")
                    self.data.append((img_path, closest_joint_values))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, joint_values = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, joint_values
    
    def save_to_csv(self, save_path):
        df_data = []
        for img_path, joint_vals in self.data:
            row = {'image_path': img_path}
            row.update(joint_vals)
            df_data.append(row)
        df = pd.DataFrame(df_data)
        df.to_csv(save_path, index=False)