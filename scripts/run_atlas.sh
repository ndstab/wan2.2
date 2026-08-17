#!/bin/bash
# T1.1 latent-inspection atlas runner.
#   $1 = run name (e.g. baseline-20260814_203000)
#   $2 = ref video path (optional; omit for baseline)
#   $3 = lambda_ref (default 0.5)
#
# Captures (~8.5 GB) go to NODE-LOCAL /tmp because the ai-at-ieor group quota on
# /scratch is exhausted (3T/3T as of 2026-08-14). Only small artifacts are copied
# back to /scratch. Captures die with the job allocation - analyse before then.
set -uo pipefail
NAME="$1"; REF="${2:-}"; LAM="${3:-0.5}"
LOCAL=/tmp/wan_atlas/$NAME
FINAL=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/outputs/latent-inspection/$NAME
mkdir -p "$LOCAL" "$FINAL"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd /scratch/IITB/ai-at-ieor/23b0702/Wan2.2

ARGS=(--task ti2v-5B
  --ckpt_dir /scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B
  --size "1280*704" --frame_num 61 --sample_steps 50
  --offload_model True --convert_model_dtype --t5_cpu
  --prompt "car on city street" --base_seed 42
  --capture_blocks 0,5,10,15,20,25,29
  --capture_timesteps 0,2,4,6,8,10,12,14,16,18,20,30,40,49
  --capture_dir "$LOCAL"
  --save_file "$LOCAL/video.mp4")
if [ -n "$REF" ]; then ARGS+=(--ref_video "$REF" --lambda_ref "$LAM"); fi

echo "=== START $(date) on $(hostname) ==="
echo "local=$LOCAL  final=$FINAL  ref=${REF:-none}  lambda=$LAM"
df -h /tmp | tail -1
time python generate.py "${ARGS[@]}"
rc=$?
echo "=== generate exit=$rc at $(date) ==="
ls -la "$LOCAL"
cp -v "$LOCAL/video.mp4" "$FINAL/" 2>&1 | tail -1
df -h /tmp | tail -1
echo "=== DONE rc=$rc ==="
