#!/bin/bash
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
echo "=== START $(date) ==="
python ~/shrink.py
echo "=== rc=$? $(date) ==="
du -sh /tmp/wan_atlas/decoded_small
ls /tmp/wan_atlas/decoded_small/*/ | wc -l
