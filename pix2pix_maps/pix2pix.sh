#!/bin/bash
#SBATCH --job-name=maps-cond
#SBATCH --time=24:00:00
#SBATCH --account=rrg-ravanelm
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lihaoyuwilson@gmail.com
#SBATCH --output=logs/%x-%j.out

# --- PRE-RUN CHECK ---
# Create the logs directory locally so SLURM can write the output file
mkdir -p logs

set -x
set -euo pipefail

# ------------------- CONFIG -------------------
MAPS_TAR="$HOME/project/datasets/maps.tar.gz"
CODE_DIR="/home/lihaoyu/Pix2Pix-Implementation/pix2pix_maps"

# Using a distinct name for the unconditional run
OUT_ROOT="/home/lihaoyu/scratch/results/pix2pix/job_${SLURM_JOB_ID}_conditional"
# ----------------------------------------------

# 1. Activate Environment
source "$HOME/myenv/bin/activate"

# 2. Setup Data in TMPDIR
cd "$SLURM_TMPDIR"
cp -v "$MAPS_TAR" .
tar -xzf "$(basename "$MAPS_TAR")"
# Assumes the tar extracted to a folder named 'maps'. 
DATA_DIR="$SLURM_TMPDIR/maps"

# 3. Setup Code
mkdir -p pix2pix_run
cp -rv "$CODE_DIR"/* pix2pix_run/
cd pix2pix_run

# 4. Create Output Directory explicitly to prevent errors
mkdir -p "$OUT_ROOT"

# 5. Run Training
# Note: conditional=False for the ablation study
python train.py receipt.yaml \
    --data_folder="$DATA_DIR" \
    --output_folder="$OUT_ROOT" \
    --batch_size=16 \
    --lr=0.0002 \
    --lambda_L1=100.0 \
    --number_of_epochs=200 \
    --conditional=True

echo "Training completed successfully."