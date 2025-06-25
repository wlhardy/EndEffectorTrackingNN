import math
import matplotlib.pyplot as plt

import eefdataset

IMAGE_FOLDER_PATH = "/home/wilah/workspace/SVO_processing/output/20250605-for_william./rgb/left"
GT_FILE_PATH = "/home/wilah/workspace/EndEffectorTracking/output_ground_truth_202506/joint_values.csv"
XY_GT_FILE_PATH = "/home/wilah/workspace/EndEffectorTracking/output_ground_truth_202506/center_piece_pixel_coordinates.csv"  # Path to the ground truth file for XY coordinates
GT_PRECISION = 1 # Degrees that the GT will be rounded to (e.g. if set to 5, then a GT of 12 will be set to 10)
NUM_CLASSES = math.ceil(360 / GT_PRECISION)

dataset = eefdataset.EEFDataset(image_dir=IMAGE_FOLDER_PATH, joint_csv_path=GT_FILE_PATH, xy_csv_path=XY_GT_FILE_PATH, joint_precision=GT_PRECISION, transform=None)

upper_joints = []
lower_joints = []
base_joints = []
gripper_joints = []
for image_path, joint_data in dataset.data:
    upper_joints.append(eefdataset.quantize_joint(joint_data['upper_joint'], GT_PRECISION))
    lower_joints.append(eefdataset.quantize_joint(joint_data['lower_joint'], GT_PRECISION))
    base_joints.append(eefdataset.quantize_joint(joint_data['base_joint'], GT_PRECISION))
    gripper_joints.append(eefdataset.quantize_joint(joint_data['gripper_joint'], GT_PRECISION))

# Plot histograms of the values of each joint
plt.hist(upper_joints, bins=NUM_CLASSES, range=(0, NUM_CLASSES), edgecolor='black')
plt.title(f"Upper Joint Angle Histogram at {GT_PRECISION} degree precision")
plt.xlabel("Upper joint angle bin")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("figs/angle_histogram_upper_joint.png", dpi=300, bbox_inches='tight')
plt.close()

plt.hist(lower_joints, bins=NUM_CLASSES, range=(0, NUM_CLASSES), edgecolor='black')
plt.title(f"Lower Joint Angle Histogram at {GT_PRECISION} degree precision")
plt.xlabel("Lower joint angle bin")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("figs/angle_histogram_lower_joint.png", dpi=300, bbox_inches='tight')
plt.close()

plt.hist(base_joints, bins=NUM_CLASSES, range=(0, NUM_CLASSES), edgecolor='black')
plt.title(f"Base Joint Angle Histogram at {GT_PRECISION} degree precision")
plt.xlabel("Base joint angle bin")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("figs/angle_histogram_base_joint.png", dpi=300, bbox_inches='tight')
plt.close()

plt.hist(gripper_joints, bins=NUM_CLASSES, range=(0, NUM_CLASSES), edgecolor='black')
plt.title(f"Gripper Joint Angle Histogram at {GT_PRECISION} degree precision")
plt.xlabel("Gripper joint angle bin")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("figs/angle_histogram_gripper_joint.png", dpi=300, bbox_inches='tight')
plt.close()