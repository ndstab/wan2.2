#!/bin/bash
# Fair test of the last-blocks hypothesis: subsets need a HIGHER lambda to
# deliver comparable total injection. Matched-lambda comparison is biased
# toward all-blocks by construction.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd /scratch/IITB/ai-at-ieor/23b0702/Wan2.2
CK=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B
REF=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/ref_videos/sandstorm.mp4
FIN=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/outputs/subset-lambda
mkdir -p "$FIN" /tmp/wan_sl
run(){ N=$1; BLK=$2; LAM=$3; O=/tmp/wan_sl/$N; mkdir -p $O
  echo "### $N blocks=$BLK lam=$LAM $(date +%H:%M:%S)"
  python generate.py --task ti2v-5B --ckpt_dir $CK --size "1280*704" \
    --frame_num 61 --sample_steps 50 --offload_model True --convert_model_dtype \
    --t5_cpu --prompt "car on city street" --base_seed 42 \
    --ref_video $REF --lambda_ref $LAM --inject_blocks "$BLK" \
    --save_file $O/video.mp4 > $O/log.txt 2>&1
  echo "   rc=$? $(date +%H:%M:%S)"; cp $O/video.mp4 $FIN/$N.mp4 2>/dev/null && echo "   copied"; }
run b29_l030   "29" 0.30
run b29_l050   "29" 0.50
run b29_l080   "29" 0.80
run b2729_l030 "27,28,29" 0.30
run b2729_l050 "27,28,29" 0.50
run b2529_l040 "25,26,27,28,29" 0.40
echo "===== DONE $(date) ====="
ls -la $FIN
