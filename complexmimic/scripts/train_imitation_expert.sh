set -e

timestamp=$(date +"%Y-%m-%d_%H-%M")
exp_name="imitation_expert"
log_dir="./logs"
log_file="${log_dir}/train_${exp_name}_trumans_${timestamp}.log"

mkdir -p "$log_dir"

echo "[INFO] Starting imitation expert training..."
echo "[INFO] Log file: $log_file"

export CUDA_VISIBLE_DEVICES=3
python -u ./complexmimic/run_teacher.py \
    learning=im_imitation_expert \
    exp_name=${exp_name} \
    env=env_train_imitation_expert \
    headless=True \
    epoch=-1 \
    2>&1 | tee "$log_file"

echo "[INFO] Training finished"