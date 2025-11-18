import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import argparse

def plot_histograms_and_scatter(args):
    # === Plot angular error histogram ===
    print("Generating plots ...")
    results = np.loadtxt(args.csv_path, delimiter=",", skiprows=1)
    angular_errors = results[:, 1]
    gt_angles = results[:, 5]

    # Compute stats
    mean_err = np.mean(angular_errors)
    std_err = np.std(angular_errors)
    stats_text = f"Mean: {mean_err:.2f}°\nStd: {std_err:.2f}°"

    # --- Histogram ---
    plt.figure()
    plt.hist(angular_errors, bins=50, color='skyblue', edgecolor='black')
    plt.title("Angular Error Distribution (°)")
    plt.xlabel("Error (°)")
    plt.ylabel("Frequency")

    # Add stats text (top-right corner)
    plt.text(
        0.97, 0.97, stats_text,
        ha='right', va='top',
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.savefig(Path(args.output_dir) / "angular_error_histogram.png", dpi=150)
    plt.close()

    # --- Scatter plot ---
    plt.figure()
    plt.scatter(gt_angles, angular_errors, s=10, alpha=0.6)
    plt.xlabel("Ground Truth Base Joint (°)")
    plt.ylabel("Angular Error (°)")
    plt.title("Prediction Error vs Ground Truth Angle")

    # Add stats text again
    plt.text(
        0.97, 0.97, stats_text,
        ha='right', va='top',
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.savefig(Path(args.output_dir) / "error_vs_gt_angle.png", dpi=150)
    plt.close()
    print("Inference complete. Results saved to:", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Results from Inference")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with inference results")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where results are saved")
    args = parser.parse_args()

    plot_histograms_and_scatter(args)