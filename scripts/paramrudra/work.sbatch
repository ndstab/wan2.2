#!/bin/bash
#SBATCH -A ai-at-ieor
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH -J t5work
#SBATCH -o /scratch/IITB/ai-at-ieor/23b0702/t5/logs/work_%j.out
T=/scratch/IITB/ai-at-ieor/23b0702/t5
nvidia-smi --query-gpu=name,memory.total --format=csv
# Two processes share the 80 GB A100: (A) eval shards 0000+0001, (B) showcase then eval 0002.
$T/run_evkv.sh 0000 0001 > $T/logs/evkv_A.log 2>&1 &
( $T/run_showcase.sh > $T/logs/showcase.log 2>&1; $T/run_evkv.sh 0002 > $T/logs/evkv_B.log 2>&1 ) &
wait
echo WORK_DONE
