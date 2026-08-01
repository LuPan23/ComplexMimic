set -e

timestamp=$(date +"%Y-%m-%d_%H-%M")
exp_name="interaction_expert"
log_dir="./logs"
log_file="${log_dir}/${exp_name}_train_trumans_${timestamp}.log"

mkdir -p "$log_dir"

echo "[INFO] Starting interaction expert training..."
echo "[INFO] Log file: $log_file"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u ./complexmimic/run_teacher.py \
    learning=im_debug \
    exp_name=${exp_name} \
    env=env_debug \
    headless=True \
    2>&1 | tee "$log_file"

echo "[INFO] Training finished"