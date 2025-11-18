python infer_token_x_y_rot.py \
    --checkpoint /home/wilah/workspace/EndEffectorTrackingNN/training/checkpoint_20251110_150106/model_checkpoint.pt \
    --dataset /home/wilah/datasets/heshan_october_grapple_data \
    --output_dir results_inference_with_data_aug \
    --precision 1 \
    --top_n 50