"""
Compute angle and position error metrics from an inference results CSV.

CSV columns expected (produced by infer_token_x_y_rot_dinov3_regression.py):
    idx, angular_error_deg, x_error_bins, y_error_bins, total_error,
    gt_base_joint_deg, pred_base_joint_deg,
    gt_x_bin, pred_x_bin_float, gt_y_bin, pred_y_bin_float,
    pred_sin2theta, pred_cos2theta
"""

import numpy as np
from pathlib import Path


def angular_error_period180(gt_deg: np.ndarray, pred_deg: np.ndarray) -> np.ndarray:
    """
    Absolute angular error accounting for 180-degree periodicity.
    Both inputs are first normalized to [0, 180).
    The error is the shortest arc distance on a 180-degree circle.
    """
    gt_norm = gt_deg % 180
    pred_norm = pred_deg % 180
    diff = np.abs(gt_norm - pred_norm)
    # Shortest arc on [0, 180) circle
    return np.minimum(diff, 180.0 - diff)


def euclidean_position_error(
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    pred_x: np.ndarray,
    pred_y: np.ndarray,
) -> np.ndarray:
    """Euclidean distance between ground-truth and predicted (x, y) positions (in percentage)."""
    return np.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)


def print_stats(name: str, errors: np.ndarray) -> None:
    mean, std, max_ = np.mean(errors), np.std(errors), np.max(errors)
    print(f" {name}: {mean:.2f}$_{{\\pm{std:.2f}}}$ & {max_:.2f}")


RESULT_FILES = [
    "outputs/dinov2_base_reg_x_y_rot_half_stats",
    "outputs/dinov3_small_reg_x_y_rot_half_stats",
    "outputs/dinov3_base_reg_x_y_rot_half_stats",
    "outputs/dinov3_large_reg_x_y_rot_half_stats",
]

# RESULT_FILES = [
#     "outputs/dinov3_base_reg_x_y_rot_full_stats",
#     "outputs/dinov3_base_reg_x_y_rot_half_stats",
#     "outputs/dinov3_base_reg_x_y_rot_quarter_stats",
#     "outputs/dinov3_base_reg_x_y_rot_eighth_stats",
# ]


def process(path: str) -> None:
    # Resolve path – accept both a direct CSV file and a results directory.
    csv_path = Path(path)
    if csv_path.is_dir():
        csv_path = csv_path / "results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    print(f"\n{'#' * 60}")
    print(f"  {path}")
    print(f"{'#' * 60}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    gt_angle   = data["gt_base_joint_deg"]
    pred_angle = data["pred_base_joint_deg"]
    gt_x       = data["gt_x_norm"] * 100
    gt_y       = data["gt_y_norm"] * 100
    pred_x     = data["pred_x_norm"] * 100
    pred_y     = data["pred_y_norm"] * 100

    # --- Angle error ---
    angle_errors = angular_error_period180(gt_angle, pred_angle)
    print_stats("Angular Error (period-180, after normalization to [0,180))", angle_errors)

    # --- Position error ---
    pos_errors = euclidean_position_error(gt_x, gt_y, pred_x, pred_y)
    print_stats("Position Error (Euclidean, in percentage)", pos_errors)


if __name__ == "__main__":
    for result_path in RESULT_FILES:
        process(result_path)
    print()
