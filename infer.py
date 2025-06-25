import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import math

import eefdataset
import model_mlp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import PIL.Image as Image
import time

# Configuration
CHECKPOINT_PATH = "training/checkpoint_20250607_005332/model_checkpoint.pt"
IMAGE_FOLDER_PATH = "/home/wilah/workspace/SVO_processing/output/20250605-for_william./rgb/left"
GT_FILE_PATH = "/home/wilah/workspace/EndEffectorTracking/output_ground_truth_202506/joint_values.csv"
XY_GT_FILE_PATH = "/home/wilah/workspace/EndEffectorTracking/output_ground_truth_202506/center_piece_pixel_coordinates.csv"
GT_PRECISION = 1 # Degrees that the GT will be rounded to (e.g. if set to 5, then a GT of 12 will be set to 10)
NUM_CLASSES = math.ceil(360 / GT_PRECISION)
BATCH_SIZE = 32
RANDOM_SEED = 42
BACKBONE = 'dinov2_vitl14_reg'
DEBUG = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.serialization.add_safe_globals([model_mlp.EndEffectorPosePred])

# Load the model
def load_model(model_path):
    checkpoint = torch.load(model_path)
    backbone_model = torch.hub.load('facebookresearch/dinov2', checkpoint['dinov2_model']).to(device)
    ee_model = model_mlp.EndEffectorPosePred(backbone_model, num_classes=checkpoint['num_classes'], mpl_dim=checkpoint['mlp_dim']).to(device)
    ee_model.load_state_dict(checkpoint['model_state_dict'])
    ee_model.eval()
    return ee_model

ee_model = load_model(CHECKPOINT_PATH)
ee_model.to(device)

# Load the dataset
transform = T.Compose([T.Lambda(lambda img: TF.rotate(img, 180)),
                    T.Lambda(lambda img: TF.crop(img, top=1, left=398, height=img.height - 2, width=img.width - 856)),
                    T.ToTensor()])
dataset = eefdataset.EEFDataset(image_dir=IMAGE_FOLDER_PATH, joint_csv_path=GT_FILE_PATH, xy_csv_path=XY_GT_FILE_PATH, joint_precision=GT_PRECISION, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize bins for each joint
upper_x = []
lower_x = []
base_x = []
gripper_x = []
upper_y = []
lower_y = []
base_y = []
gripper_y = []

for images, joint_values in dataloader:
    upper_joint_angles = joint_values['upper_joint'].numpy()
    lower_joint_angles = joint_values['lower_joint'].numpy()
    base_joint_angles = joint_values['base_joint'].numpy()
    gripper_joint_angles = joint_values['gripper_joint'].numpy()
    x = joint_values['x'].numpy()
    y = joint_values['y'].numpy()
    target_upper = torch.tensor([
        eefdataset.quantize_joint(upper, GT_PRECISION) for upper in upper_joint_angles
    ], dtype=torch.long).to(device)
    target_lower = torch.tensor([
        eefdataset.quantize_joint(lower, GT_PRECISION) for lower in lower_joint_angles
    ], dtype=torch.long).to(device)
    target_base = torch.tensor([
        eefdataset.quantize_joint(base, GT_PRECISION, symmetric=True) for base in base_joint_angles
    ], dtype=torch.long).to(device)
    target_gripper = torch.tensor([
        eefdataset.quantize_joint(gripper, GT_PRECISION) for gripper in gripper_joint_angles
    ], dtype=torch.long).to(device)

    images = images.to(device)
    image_width, image_height = images.shape[2], images.shape[3]

    target_x = torch.tensor([
        (x_val / image_width) for x_val in x
    ], dtype=torch.float32).to(device)
    target_y = torch.tensor([
        (y_val / image_height) for y_val in y
    ], dtype=torch.float32).to(device)

    # Compute the time taken for inference
    start_time = time.time()
    j1_logits, j2_logits, j3_logits, j4_logits, base_x, base_y = ee_model(images)
    end_time = time.time()
    print(f"Inference time for batch: {end_time - start_time:.4f} seconds")

    j1_preds = j1_logits.argmax(dim=1)
    j2_preds = j2_logits.argmax(dim=1)
    j3_preds = j3_logits.argmax(dim=1)
    j4_preds = j4_logits.argmax(dim=1)

    j1_pred_angles = eefdataset.class_to_angle(j1_preds, GT_PRECISION)
    j2_pred_angles = eefdataset.class_to_angle(j2_preds, GT_PRECISION)
    j3_pred_angles = eefdataset.class_to_angle(j3_preds, GT_PRECISION)
    j4_pred_angles = eefdataset.class_to_angle(j4_preds, GT_PRECISION)

    j1_gt_angles = eefdataset.class_to_angle(target_upper, GT_PRECISION)
    j2_gt_angles = eefdataset.class_to_angle(target_lower, GT_PRECISION)
    j3_gt_angles = eefdataset.class_to_angle(target_base, GT_PRECISION)
    j4_gt_angles = eefdataset.class_to_angle(target_gripper, GT_PRECISION)

    # Compute angular error
    angular_error_j1 = torch.abs(j1_pred_angles - joint_values['upper_joint'].to(device))
    angular_error_j2 = torch.abs(j2_pred_angles - joint_values['lower_joint'].to(device))
    angular_error_j3 = torch.abs(j3_pred_angles - joint_values['base_joint'].to(device))
    angular_error_j4 = torch.abs(j4_pred_angles - joint_values['gripper_joint'].to(device))
    
    # Add angular error to the corresponding bins
    for i in range(len(images)):
        upper_x.append(j1_gt_angles[i].item())
        lower_x.append(j2_gt_angles[i].item())
        base_x.append(j3_gt_angles[i].item())
        gripper_x.append(j4_gt_angles[i].item())

        upper_y.append(angular_error_j1[i].item())
        lower_y.append(angular_error_j2[i].item())
        base_y.append(angular_error_j3[i].item())
        gripper_y.append(angular_error_j4[i].item())
        if angular_error_j3[i].item() > 10 and DEBUG:
            # Show the image
            print(f"High error for base joint: {j3_gt_angles[i].item()} degrees with error {angular_error_j3[i].item()} degrees")
            img_tmp = images[i].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format for visualization
            img_tmp = (img_tmp * 255).astype('uint8')
            img = Image.fromarray(img_tmp)
            img.show()
            # Wait for user input
            input("Press Enter to continue...")

# Create scatter plots for each joint

def plot_scatter(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.xlim(-180, 180)
    plt.ylim(0, 180)
    plt.savefig(f"figs/{title.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')

plot_scatter(upper_x, upper_y, "Upper Joint Angle Prediction Error", "Upper Joint Angle (degrees)", "Angular Error (degrees)")
plot_scatter(lower_x, lower_y, "Lower Joint Angle Prediction Error", "Lower Joint Angle (degrees)", "Angular Error (degrees)")
plot_scatter(base_x, base_y, "Base Joint Angle Prediction Error", "Base Joint Angle (degrees)", "Angular Error (degrees)")
plot_scatter(gripper_x, gripper_y, "Gripper Joint Angle Prediction Error", "Gripper Joint Angle (degrees)", "Angular Error (degrees)")