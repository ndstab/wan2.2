#!/bin/bash
# T4.4 gap-fill, 2026-09-01.
#   (1) same-session RANDOM control at lambda=0.15 -- removes the "historical
#       Aug-15 code state" confound from the one lambda where inverted comes
#       closest to random.
#   (2) INVERTED at lambda=0.20 -- the untested gap between inv 0.15 (works)
#       and inv 0.25 (total collapse).
# Everything else held identical to run_inv_gen.sh: same prompt, seed, ref,
# resolution, frame count, steps, shift, guide scale.
set -u
cd /scratch/IITB/ai-at-ieor/23b0702/Wan2.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

CKPT=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B
REF=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/ref_videos/sandstorm.mp4
INV=/home/IITB/ai-at-ieor/23b0702/ref_inv_sandstorm.pt
P="car on city street"
OUT=~/gen_inv
mkdir -p $OUT

echo "=== (1) RANDOM control lambda=0.15 (same session) ==="
python -u generate.py --task ti2v-5B --size "1280*704" --frame_num 61 \
  --ckpt_dir $CKPT --prompt "$P" --base_seed 42 \
  --ref_video $REF --lambda_ref 0.15 \
  --save_file $OUT/rand_lam0.15.mp4 || echo "FAILED rand 0.15"

echo "=== (2) INVERTED lambda=0.20 (gap fill) ==="
python -u generate.py --task ti2v-5B --size "1280*704" --frame_num 61 \
  --ckpt_dir $CKPT --prompt "$P" --base_seed 42 \
  --ref_video $REF --lambda_ref 0.20 --ref_inversion $INV \
  --save_file $OUT/inv_lam0.20.mp4 || echo "FAILED inv 0.20"

echo "=== T4.4 GAPFILL DONE ==="
