import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

input_file = "/home/wilah/workspace/EndEffectorTrackingNN/results_inference_with_data_aug_reg/dinov3_base_reg_x_y_rot_half_res_fixed_x_y/results.csv"
output_file = "/home/wilah/workspace/EndEffectorTrackingNN/results_inference_with_data_aug_reg/dinov3_base_reg_x_y_rot_half_res_fixed_x_y/sorted_output.csv"

with open(input_file, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)

# If your CSV has a header row, separate it
header = rows[0]
data = rows[1:]

# Sort by first column (filename), case-insensitive
data.sort(key=lambda row: row[0].lower())

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print("CSV sorted successfully!")

print("Generating plots ...")
results = np.loadtxt(output_file, delimiter=",", skiprows=1, usecols=(1,5))
angular_errors = results[:, 0]
gt_angles = results[:, 1]

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
plt.savefig("test_sur_sorted_data/angular_error_histogram.png", dpi=150)
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
plt.savefig("test_sur_sorted_data/error_vs_gt_angle.png", dpi=150)
plt.close()