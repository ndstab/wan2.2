#!/bin/bash
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd /scratch/IITB/ai-at-ieor/23b0702/Wan2.2
REF=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/ref_videos/sandstorm.mp4
echo "=== START $(date) ==="
python compute_metrics.py --manifest /tmp/wan_night/manifest.csv --ref $REF \
  --out /tmp/wan_night/metrics.csv
echo "=== rc=$? ==="
cp /tmp/wan_night/metrics.csv /scratch/IITB/ai-at-ieor/23b0702/Wan2.2/outputs/overnight/metrics.csv 2>/dev/null && echo copied
