import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd
import torchvision.transforms as transforms
import pickle
import hashlib
import json
import math
from bisect import bisect_left


def get_closest_dict_value(data_dict, timestamp):
    closest_timestamp = min(data_dict.keys(), key=lambda t: abs(t - timestamp))
    return data_dict[closest_timestamp], closest_timestamp

def quantize_joint(joint_value, precision, symmetric=False):
    # Normalize to [0, 360)
    angle = (joint_value + 180.0) % 360.0

    if symmetric:
        # Map to [0, 180)
        if angle >= 180.0:
            angle -= 180.0
        n_classes = int(round(180.0 / precision))
    else:
        n_classes = int(round(360.0 / precision))

    # Bin index with wrap-around. The 1e-9 avoids edge cases like 179.9999999 -> 180
    bin_f = (angle / precision) + 1e-9
    class_idx = int(math.floor(bin_f)) % n_classes
    return class_idx

def class_to_angle(class_idx, precision, symmetric=False, center=False):
    if symmetric:
        base = 0.0
    else:
        base = -180.0

    offset = (class_idx + 0.5) * precision if center else class_idx * precision
    return base + offset

class EEFDataset(Dataset):
    def __init__(self, image_dirs, joint_csv_paths, xy_csv_paths=None,
                 joint_precision=1, transform=None, cache_dir="cache"):

        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        if isinstance(joint_csv_paths, str):
            joint_csv_paths = [joint_csv_paths]
        if xy_csv_paths is None:
            xy_csv_paths = []
        elif isinstance(xy_csv_paths, str):
            xy_csv_paths = [xy_csv_paths]

        self.image_dirs = image_dirs
        self.joint_precision = joint_precision
        self.transform = transform

        closest_time_threshold = 5e+7
        time_interval = 6e+10

        # --- Create a unique cache identifier ---
        param_dict = {
            "image_dirs": sorted(image_dirs),
            "joint_csv_paths": sorted(joint_csv_paths),
            "xy_csv_paths": sorted(xy_csv_paths),
            "joint_precision": joint_precision,
            "closest_time_threshold": closest_time_threshold,
            "time_interval": time_interval
        }
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = hashlib.md5(json.dumps(param_dict, sort_keys=True).encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_key}_data.pkl")
        meta_path = os.path.join(cache_dir, f"{cache_key}_meta.json")

        # --- Load from cache if available ---
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta == param_dict:
                print(f"Loading cached dataset: {cache_path}")
                with open(cache_path, "rb") as f:
                    self.data = pickle.load(f)
                return

        # --- Otherwise, build dataset ---
        self.data = []
        joint_dict = {}
        for joint_csv_path in joint_csv_paths:
            joint_csv_data = pd.read_csv(joint_csv_path)
            for _, row in joint_csv_data.iterrows():
                joint_dict[row['timestamp']] = {
                    'basemast_to_mast': row['basemast_to_mast'],
                    'mast_to_mainboom': row['mast_to_mainboom'],
                    'stick_to_telescope': row['stick_to_telescope'],
                    'telescope_joint': row['telescope_joint'],
                    'upper_joint': row['upper_joint'],
                    'lower_joint': row['lower_joint'],
                    'base_joint': row['base_joint'],
                    'gripper_joint': row['gripper_joint']
                }

        xy_dict = {}
        if xy_csv_paths:
            for xy_csv_path in xy_csv_paths:
                xy_csv_data = pd.read_csv(xy_csv_path)
                for _, row in xy_csv_data.iterrows():
                    xy_dict[row['timestamp']] = {'x': row['x_center'], 'y': row['y_center']}

        key_order = ['basemast_to_mast', 'mast_to_mainboom', 'stick_to_telescope',
                     'telescope_joint', 'upper_joint', 'lower_joint', 'base_joint', 'gripper_joint']
        already_present = defaultdict(list)

        for image_dir in image_dirs:
            for file_name in sorted(os.listdir(image_dir)):
                if file_name.endswith(".png"):
                    timestamp = int(os.path.splitext(file_name)[0])
                    closest_joint_values, closest_timestamp = get_closest_dict_value(joint_dict, timestamp)

                    if abs(closest_timestamp - timestamp) > closest_time_threshold:
                        print(f"Warning: No close joint values found for {file_name}. Skipping.")
                        continue

                    img_path = os.path.join(image_dir, file_name)
                    values_tuple = tuple(
                        quantize_joint(float(closest_joint_values[key]), joint_precision)
                        for key in key_order
                    )

                    add_entry = False
                    if not already_present[values_tuple]:
                        add_entry = True
                    else:
                        diffs = [abs(timestamp - t) for t in already_present[values_tuple]]
                        if all(diff > time_interval for diff in diffs):
                            add_entry = True

                    if add_entry:
                        already_present[values_tuple].append(timestamp)
                        closest_xy, closest_xy_timestamp = get_closest_dict_value(xy_dict, timestamp)
                        if closest_xy is not None:
                            if abs(closest_xy_timestamp - timestamp) > closest_time_threshold:
                                print(f"Warning: No close XY values found for {file_name} "
                                    f"(diff={abs(closest_xy_timestamp - timestamp)}). Using defaults.")
                                closest_joint_values['x'] = 0
                                closest_joint_values['y'] = 0
                            else:
                                closest_joint_values['x'] = closest_xy['x']
                                closest_joint_values['y'] = closest_xy['y']
                        else:
                            closest_joint_values['x'] = 0
                            closest_joint_values['y'] = 0

                        self.data.append((img_path, closest_joint_values))

        # --- Save to cache ---
        with open(cache_path, "wb") as f:
            pickle.dump(self.data, f)
        with open(meta_path, "w") as f:
            json.dump(param_dict, f)
        print(f"Dataset cached to {cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, joint_values = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        x_norm = joint_values['x'] / orig_w
        y_norm = joint_values['y'] / orig_h
        if self.transform:
            image = self.transform(image)
        if isinstance(image, torch.Tensor):
            _, new_h, new_w = image.shape
        else:
            new_w, new_h = image.size
        x_transf = x_norm * new_w
        y_transf = y_norm * new_h
        joint_values['x'] = x_transf
        joint_values['y'] = y_transf
        return image, joint_values

    def save_to_csv(self, save_path):
        df = pd.DataFrame([{'image_path': img, **vals} for img, vals in self.data])
        df.to_csv(save_path, index=False)