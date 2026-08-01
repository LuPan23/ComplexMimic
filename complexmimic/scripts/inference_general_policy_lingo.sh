set -e

timestamp=$(date +"%Y-%m-%d_%H-%M")
exp_name="general_policy"
log_dir="./logs"
log_file="${log_dir}/inference_${exp_name}_lingo_${timestamp}.log"

mkdir -p "$log_dir"

echo "[INFO] Starting testing..."
echo "[INFO] Log file: $log_file"

export CUDA_VISIBLE_DEVICES=0
python -u ./complexmimic/run_student.py \
    learning=im_distillation \
    exp_name=${exp_name} \
    env=env_inference_lingo \
    epoch=4500 \
    headless=True \
    test=True \
    im_eval=True \
    collect_dataset=True \
    2>&1 | tee "$log_file"

echo "[INFO] Training finished"