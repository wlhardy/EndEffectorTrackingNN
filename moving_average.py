import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# =========================
# 1. Load CSV
# =========================
csv_path = "/home/wilah/workspace/EndEffectorTrackingNN/sorted_output.csv"
df = pd.read_csv(csv_path)

def ang_error_deg_period180(pred_deg, gt_deg):
    # both in degrees, return minimal error under 180° periodicity
    d = pred_deg - gt_deg                  # signed difference
    d = np.remainder(d + 90.0, 180.0)   # shift, wrap to [0,180)
    d = d - 90.0                           # shift back to [-90,90)
    return d

# =========================
# 2. Sort by timestamp
# =========================
# Extract timestamp from image_name
df["timestamp"] = df["image_name"].str.replace(".png", "", regex=False).astype(np.int64)
df = df.sort_values("timestamp").reset_index(drop=True)

# =========================
# 3. Optimize EMA Alpha
# =========================

def apply_ema(df, alpha):
    df_temp = df.copy()

    # ---- Linear signals ----
    df_temp["pred_x_bin_float_filtered"] = (
        df_temp["pred_x_bin_float"].ewm(alpha=alpha, adjust=False).mean()
    )

    df_temp["pred_y_bin_float_filtered"] = (
        df_temp["pred_y_bin_float"].ewm(alpha=alpha, adjust=False).mean()
    )

    # ---- Angular signal (unit circle smoothing) ----
    theta_rad = np.deg2rad(df_temp["pred_base_joint_deg"])
    cos_vals = np.cos(theta_rad)
    sin_vals = np.sin(theta_rad)

    cos_ema = pd.Series(cos_vals).ewm(alpha=alpha, adjust=False).mean()
    sin_ema = pd.Series(sin_vals).ewm(alpha=alpha, adjust=False).mean()

    theta_filtered = np.arctan2(sin_ema, cos_ema)
    df_temp["pred_base_joint_deg_filtered"] = np.rad2deg(theta_filtered)

    return df_temp


def compute_total_rmse(df_temp):
    # Angular
    ang_error = ang_error_deg_period180(
        df_temp["pred_base_joint_deg_filtered"],
        df_temp["gt_base_joint_deg"]
    )

    # X
    x_error = df_temp["pred_x_bin_float_filtered"] - df_temp["gt_x_bin"]

    # Y
    y_error = df_temp["pred_y_bin_float_filtered"] - df_temp["gt_y_bin"]

    rmse_ang = np.sqrt(np.mean(ang_error**2))
    rmse_x = np.sqrt(np.mean(x_error**2))
    rmse_y = np.sqrt(np.mean(y_error**2))

    # You can weight them if needed
    total_rmse = rmse_ang + rmse_x + rmse_y

    return total_rmse, rmse_ang, rmse_x, rmse_y


# ---- Grid Search ----
alpha_candidates = np.linspace(0.01, 0.99, 50)

best_alpha = None
best_score = np.inf
best_breakdown = None

for alpha in alpha_candidates:
    df_test = apply_ema(df, alpha)
    total_rmse, rmse_ang, rmse_x, rmse_y = compute_total_rmse(df_test)

    if total_rmse < best_score:
        best_score = total_rmse
        best_alpha = alpha
        best_breakdown = (rmse_ang, rmse_x, rmse_y)

print("\n==== Optimal Alpha Search ====")
print(f"Best alpha: {best_alpha:.4f}")
print(f"Total RMSE: {best_score:.6f}")
print(f"Angular RMSE: {best_breakdown[0]:.6f}")
print(f"X RMSE: {best_breakdown[1]:.6f}")
print(f"Y RMSE: {best_breakdown[2]:.6f}")

# Apply best alpha permanently
df = apply_ema(df, best_alpha)


# =========================
# 4. Recompute Errors
# =========================

# Angular error
df["angular_error_before"] = ang_error_deg_period180(df["pred_base_joint_deg"], df["gt_base_joint_deg"])
df["angular_error_after"] = ang_error_deg_period180(df["pred_base_joint_deg_filtered"], df["gt_base_joint_deg"])

# X error
df["x_error_before"] = df["pred_x_bin_float"] - df["gt_x_bin"]
df["x_error_after"] = df["pred_x_bin_float_filtered"] - df["gt_x_bin"]

# Y error
df["y_error_before"] = df["pred_y_bin_float"] - df["gt_y_bin"]
df["y_error_after"] = df["pred_y_bin_float_filtered"] - df["gt_y_bin"]

# =========================
# 5. Metrics Function
# =========================
def compute_metrics(error_series):
    mae = np.mean(np.abs(error_series))
    rmse = np.sqrt(np.mean(error_series ** 2))
    return mae, rmse

# =========================
# 6. Compare Before vs After
# =========================
metrics = {}

for label in ["angular", "x", "y"]:
    mae_before, rmse_before = compute_metrics(df[f"{label}_error_before"])
    mae_after, rmse_after = compute_metrics(df[f"{label}_error_after"])

    metrics[label] = {
        "MAE_before": mae_before,
        "MAE_after": mae_after,
        "RMSE_before": rmse_before,
        "RMSE_after": rmse_after
    }

# =========================
# 7. Print Results
# =========================
for k, v in metrics.items():
    print(f"\n==== {k.upper()} ERROR ====")
    print(f"MAE   before: {v['MAE_before']:.6f}")
    print(f"MAE   after : {v['MAE_after']:.6f}")
    print(f"RMSE  before: {v['RMSE_before']:.6f}")
    print(f"RMSE  after : {v['RMSE_after']:.6f}")

# =========================
# 8. Create Output Folder
# =========================
output_dir = os.path.join(os.path.dirname(csv_path), "plots_moving_average")
os.makedirs(output_dir, exist_ok=True)

time = df["timestamp"]

# =========================
# 9. Angular Error Over Time
# =========================
plt.figure(figsize=(12, 5))
plt.plot(time, df["angular_error_before"], label="Before MA", alpha=0.7)
plt.plot(time, df["angular_error_after"], label="After MA", alpha=0.7)
plt.title("Angular Error Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Angular Error (deg)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "angular_error_over_time.png"), dpi=300)
plt.close()

# =========================
# 10. X Error Over Time
# =========================
plt.figure(figsize=(12, 5))
plt.plot(time, df["x_error_before"], label="Before MA", alpha=0.7)
plt.plot(time, df["x_error_after"], label="After MA", alpha=0.7)
plt.title("X Error Over Time")
plt.xlabel("Timestamp")
plt.ylabel("X Error (bins)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "x_error_over_time.png"), dpi=300)
plt.close()

# =========================
# 11. Y Error Over Time
# =========================
plt.figure(figsize=(12, 5))
plt.plot(time, df["y_error_before"], label="Before MA", alpha=0.7)
plt.plot(time, df["y_error_after"], label="After MA", alpha=0.7)
plt.title("Y Error Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Y Error (bins)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "y_error_over_time.png"), dpi=300)
plt.close()

# =========================
# 12. X Prediction vs GT
# =========================
plt.figure(figsize=(12, 5))
plt.plot(time, df["gt_x_bin"], label="GT X", linewidth=2)
plt.plot(time, df["pred_x_bin_float"], label="Pred X Before", alpha=0.6)
plt.plot(time, df["pred_x_bin_float_filtered"], label="Pred X After", alpha=0.8)
plt.title("X Position Over Time")
plt.xlabel("Timestamp")
plt.ylabel("X (bins)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "x_position_over_time.png"), dpi=300)
plt.close()

# =========================
# 13. Y Prediction vs GT
# =========================
plt.figure(figsize=(12, 5))
plt.plot(time, df["gt_y_bin"], label="GT Y", linewidth=2)
plt.plot(time, df["pred_y_bin_float"], label="Pred Y Before", alpha=0.6)
plt.plot(time, df["pred_y_bin_float_filtered"], label="Pred Y After", alpha=0.8)
plt.title("Y Position Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Y (bins)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "y_position_over_time.png"), dpi=300)
plt.close()

# =========================
# 14. 2D Trajectory Plot (Before MA)
# =========================
plt.figure(figsize=(6, 6))
plt.plot(df["gt_x_bin"], df["gt_y_bin"], label="GT", linewidth=2)
plt.plot(df["pred_x_bin_float"], df["pred_y_bin_float"], label="Prediction Before", alpha=0.7)
plt.title("2D Trajectory (Before Moving Average)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "trajectory_before_ma.png"), dpi=300)
plt.close()

# =========================
# 15. 2D Trajectory Plot (After MA)
# =========================
plt.figure(figsize=(6, 6))
plt.plot(df["gt_x_bin"], df["gt_y_bin"], label="GT", linewidth=2)
plt.plot(df["pred_x_bin_float_filtered"], df["pred_y_bin_float_filtered"],
         label="Prediction After", alpha=0.7)
plt.title("2D Trajectory (After Moving Average)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "trajectory_after_ma.png"), dpi=300)
plt.close()

print(f"\nPlots saved to: {output_dir}")