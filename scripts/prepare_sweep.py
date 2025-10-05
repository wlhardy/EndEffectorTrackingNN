import wandb
import datetime
import math
import os

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
WEIGHT_LOSS_JOINTS = 1.0
WEIGHT_LOSS_XY = 1.0
FREEZE_BLOCKS = [0, 2, 4, 6]  # Max number of blocks in the backbone is 11
MAX_BATCH_SIZE = 20
TARGET_BATCH_SIZE = [20]
FREEZE_POS_EMBED = True
FREEZE_PATCH_EMBED = True
XY_BIN_NBR = 100
MAIN_TRAIN_FOLDER = "./datasets/val"
VAL_TRAIN_FOLDER = "./datasets/val"

sweep_config = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "loss"},
    "early_terminate": {
        "eta": 5,
        "max_iter": 40,
        "s": 2,
        "type": "hyperband",
    },
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
        "freeze_pos_embed": {"value": FREEZE_POS_EMBED},
        "freeze_patch_embed": {"value": FREEZE_PATCH_EMBED},
        "xy_bin_nbr": {"value": XY_BIN_NBR}
    },
}

if __name__ == "__main__":
    # TODO: Login to wandb is required in a few scripts, so we should probably make a utility function for this
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        try:
            with open("/run/wandb_api_key.txt", "r") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            print("API key not found")
            exit(1)
    if api_key:
        wandb.login(key=api_key)
        print("Logged into wandb successfully.")
    else:
        print("Could not login to wandb. Exiting.")
        raise SystemExit(1)

    sweep_id = wandb.sweep(sweep_config, project="EndEffectorPosePred")
    
    # Write the sweep ID to a file with timestamp for use in slurm.sh
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f".tmp/sweep_id_{timestamp}.txt", "w") as f:
        f.write(sweep_id)
    print(f"Sweep ID: {sweep_id}")