python infer_token_x_y_rot_dinov3_regression.py \
    --checkpoint /home/wilah/workspace/EndEffectorTrackingNN/training/checkpoint_20260211_150713/model_checkpoint.pt \
    --dataset /home/wilah/datasets/heshan_october_grapple_data \
    --output_dir results_inference_with_data_aug_reg \
    --precision 1 \
    --top_n 50