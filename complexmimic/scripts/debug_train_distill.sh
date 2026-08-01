set -e

timestamp=$(date +"%Y-%m-%d_%H-%M")
exp_name="general_policy"
log_dir="./logs"
log_file="${log_dir}/${exp_name}_train_trumans_${timestamp}.log"

mkdir -p "$log_dir"

echo "[INFO] Starting interaction expert training..."
echo "[INFO] Log file: $log_file"

export CUDA_VISIBLE_DEVICES=0
python -u ./complexmimic/run_student.py \
    learning=im_distill_debug \
    exp_name=${exp_name} \
    env=env_distill_debug \
    headless=True \
    2>&1 | tee "$log_file"

echo "[INFO] Training finished"