#!/bin/bash
# lambda sweep: same prompt/seed/ref as the atlas, no captures (videos only).
# Answers the open question "is the content collapse monotonic in lambda?"
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd /scratch/IITB/ai-at-ieor/23b0702/Wan2.2
REF=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/ref_videos/sandstorm.mp4
FINAL=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/outputs/lambda-sweep
mkdir -p "$FINAL" /tmp/wan_lambda
for LAM in 0.05 0.10; do
  echo "=== lambda=$LAM $(date) ==="
  OUT=/tmp/wan_lambda/lam_$LAM
  mkdir -p "$OUT"
  time python generate.py --task ti2v-5B \
    --ckpt_dir /scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B \
    --size "1280*704" --frame_num 61 --sample_steps 50 \
    --offload_model True --convert_model_dtype --t5_cpu \
    --prompt "car on city street" --base_seed 42 \
    --ref_video "$REF" --lambda_ref $LAM \
    --save_file "$OUT/video.mp4"
  echo "rc=$?"
  cp -v "$OUT/video.mp4" "$FINAL/lambda_$LAM.mp4" 2>&1 | tail -1
done
echo "=== SWEEP DONE $(date) ==="
ls -la "$FINAL"
