#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 4-00:00:00
#SBATCH --job-name=jup
#SBATCH -o jupyter_%j.log
 
# === Dynamically set nodelist if provided ===
if [ ! -z "$NODELIST" ]; then
    # Requeue job with node constraint if SLURM_NNODES not set (avoid loop)
    if [ -z "$SLURM_NNODES" ]; then
        echo "Re-submitting with --nodelist=$NODELIST"
        sbatch --nodelist=$NODELIST "$0"
        exit 0
    fi
fi
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
 
# === Set port and IP ===
PORT=2222
IP=$(hostname -i)
 
echo "Jupyter starting on $IP:$PORT"
 
# === Start Jupyter ===
jupyter notebook --no-browser --port=$PORT --ip=$IP
