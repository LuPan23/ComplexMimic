set -e

timestamp=$(date +"%Y-%m-%d_%H-%M")
exp_name="interaction_expert"
log_dir="./logs"
log_file="${log_dir}/inference_${exp_name}_trumans_${timestamp}.log"

mkdir -p "$log_dir"

echo "[INFO] Starting testing..."
echo "[INFO] Log file: $log_file"

export CUDA_VISIBLE_DEVICES=0
python -u ./complexmimic/run_teacher.py\
    learning=im_interaction_expert \
    exp_name=${exp_name} \
    env=env_inference_trumans \
    epoch=15000 \
    headless=True \
    test=True \
    im_eval=True \
    collect_dataset=True \
    2>&1 | tee "$log_file"

echo "[INFO] Evaluation finished"