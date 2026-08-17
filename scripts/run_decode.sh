#!/bin/bash
# Decode a subset of captured block residuals to PNG, both arms.
# PNGs stay on node-local /tmp (scratch group quota is full).
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd /scratch/IITB/ai-at-ieor/23b0702/Wan2.2
CKPT=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B
BLOCKS=0,5,10,15,20,25,29
STEPS=0,10,20,49
for ARM in baseline-20260814_202759 injected-20260814_204155; do
  echo "=== DECODE $ARM $(date) ==="
  python decode_latent.py \
    --captures /tmp/wan_atlas/$ARM/captures.pt \
    --ckpt_dir $CKPT \
    --output_dir /tmp/wan_atlas/decoded/$ARM \
    --only_blocks $BLOCKS --only_timesteps $STEPS
  echo "=== rc=$? ==="
  ls /tmp/wan_atlas/decoded/$ARM | wc -l
done
echo "=== ALL DONE $(date) ==="
du -sh /tmp/wan_atlas/decoded/*
