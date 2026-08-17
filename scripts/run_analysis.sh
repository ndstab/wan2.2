#!/bin/bash
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd /scratch/IITB/ai-at-ieor/23b0702/Wan2.2
mkdir -p /tmp/wan_atlas/analysis
echo "=== START $(date) ==="
time python analyze_captures.py \
  --baseline /tmp/wan_atlas/baseline-20260814_202759/captures.pt \
  --injected /tmp/wan_atlas/injected-20260814_204155/captures.pt \
  --output_dir /tmp/wan_atlas/analysis --no_plots
echo "=== EXIT $? at $(date) ==="
ls -la /tmp/wan_atlas/analysis
