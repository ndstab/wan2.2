#!/bin/bash
# Showcase: window-gated K/V injection with first-person WeatherWeaver references.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate torch
export PYTHONUNBUFFERED=1
T=/scratch/IITB/ai-at-ieor/23b0702/t5; O=$T/results/showcase
export TMPDIR=$T/tmp XDG_CACHE_HOME=$T/hf_home TORCH_HOME=$T/hf_home HF_HOME=$T/hf_home
cd /scratch/IITB/ai-at-ieor/23b0702/Wan2.2
CK=/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B
P="first-person dashcam view driving down a city street"
gen() { n=$1; shift; [ -f $O/$n.mp4 ] && return; echo "=== $n $(date +%T)"
  python generate.py --task ti2v-5B --ckpt_dir $CK --size "1280*704" --frame_num 61 --sample_steps 50 \
    --offload_model True --convert_model_dtype --t5_cpu --prompt "$P" --base_seed 42 --save_file $O/$n.mp4 "$@" > $O/$n.log 2>&1
  echo "    rc=$?"; }
gen base
for W in rain snow fog storm; do
  gen ${W}_early --ref_video $T/refs/$W.mp4 --lambda_ref 0.10 --inject_sigma 0.9,1.0
  gen ${W}_all   --ref_video $T/refs/$W.mp4 --lambda_ref 0.10
done
echo SHOWCASE_DONE
