#!/bin/bash
# T5.5: paired ground-truth eval of window-gated reference K/V injection.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate torch
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=/scratch/IITB/ai-at-ieor/23b0702/t5
export TMPDIR=$T/tmp XDG_CACHE_HOME=$T/hf_home TORCH_HOME=$T/hf_home HF_HOME=$T/hf_home; mkdir -p $TMPDIR
cd /scratch/IITB/ai-at-ieor/23b0702/Wan2.2
for SH in "$@"; do
  python paired_eval_kv.py --ckpt_dir /scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B \
    --shards "$T/data/scene0/$SH.tar" --ref_pool "$T/data/scene0/*.tar" --out $T/results/evkv \
    --lambdas 0.1,0.2 --windows "0,1;0.9,1;0.8,1;0,0.9" --leak "" --pixel --save_videos 1
done
echo EVKV_ALL_DONE
