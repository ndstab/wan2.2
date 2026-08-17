#!/bin/bash
# Overnight queue: T1.2 block subsets at the NEW operating point (lambda=0.10),
# plus prompt-generality checks. ~11 runs x ~9.6 min.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd /scratch/IITB/ai-at-ieor/23b0702/Wan2.2
CK=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B
REF=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/ref_videos/sandstorm.mp4
FIN=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/outputs/overnight
mkdir -p "$FIN" /tmp/wan_night

run () {  # $1=name  $2=prompt  $3=lambda(0=baseline)  $4=inject_blocks("" = all)
  local NAME="$1" PROMPT="$2" LAM="$3" BLK="${4:-}"
  local O=/tmp/wan_night/$NAME
  mkdir -p "$O"
  echo "### $NAME | lam=$LAM | blocks=${BLK:-all} | $(date +%H:%M:%S)"
  local A=(--task ti2v-5B --ckpt_dir $CK --size "1280*704" --frame_num 61
           --sample_steps 50 --offload_model True --convert_model_dtype --t5_cpu
           --prompt "$PROMPT" --base_seed 42 --save_file "$O/video.mp4")
  if [ "$LAM" != "0" ]; then A+=(--ref_video "$REF" --lambda_ref "$LAM"); fi
  if [ -n "$BLK" ]; then A+=(--inject_blocks "$BLK"); fi
  python generate.py "${A[@]}" > "$O/log.txt" 2>&1
  echo "   rc=$? $(date +%H:%M:%S)"
  cp "$O/video.mp4" "$FIN/$NAME.mp4" 2>/dev/null && echo "   copied" || echo "   COPY FAILED (stays on /tmp)"
}

P1="car on city street"
echo "===== BLOCK SUBSETS at lambda=0.10 ====="
run blk_29        "$P1" 0.10 "29"
run blk_27-29     "$P1" 0.10 "27,28,29"
run blk_25-29     "$P1" 0.10 "25,26,27,28,29"
run blk_15-20     "$P1" 0.10 "15,16,17,18,19,20"
run blk_10-20     "$P1" 0.10 "10,11,12,13,14,15,16,17,18,19,20"

echo "===== PROMPT GENERALITY at lambda=0.10, all blocks ====="
P2="a truck driving on a highway"
P3="people walking on a city sidewalk"
P4="an empty parking lot with tall buildings"
run p2_base "$P2" 0;    run p2_inj "$P2" 0.10
run p3_base "$P3" 0;    run p3_inj "$P3" 0.10
run p4_base "$P4" 0;    run p4_inj "$P4" 0.10

echo "===== METRICS ====="
B=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/outputs
M=/tmp/wan_night/manifest.csv
: > $M
echo "lambda_0.05,$B/lambda-sweep/lambda_0.05.mp4,$B/latent-inspection/baseline-20260814_202759/video.mp4" >> $M
echo "lambda_0.10,$B/lambda-sweep/lambda_0.10.mp4,$B/latent-inspection/baseline-20260814_202759/video.mp4" >> $M
echo "lambda_0.15,$B/lambda-sweep/lambda_0.15.mp4,$B/latent-inspection/baseline-20260814_202759/video.mp4" >> $M
echo "lambda_0.25,$B/lambda-sweep/lambda_0.25.mp4,$B/latent-inspection/baseline-20260814_202759/video.mp4" >> $M
echo "lambda_0.35,$B/lambda-sweep/lambda_0.35.mp4,$B/latent-inspection/baseline-20260814_202759/video.mp4" >> $M
echo "lambda_0.50,$B/latent-inspection/injected-20260814_204155/video.mp4,$B/latent-inspection/baseline-20260814_202759/video.mp4" >> $M
for n in blk_29 blk_27-29 blk_25-29 blk_15-20 blk_10-20; do
  echo "$n,/tmp/wan_night/$n/video.mp4,$B/latent-inspection/baseline-20260814_202759/video.mp4" >> $M
done
for n in p2 p3 p4; do
  echo "${n}_inj,/tmp/wan_night/${n}_inj/video.mp4,/tmp/wan_night/${n}_base/video.mp4" >> $M
done
python compute_metrics.py --manifest $M --ref $REF --out /tmp/wan_night/metrics.csv
cp /tmp/wan_night/metrics.csv "$FIN/metrics.csv" 2>/dev/null && echo "metrics copied"
echo "===== ALL DONE $(date) ====="
