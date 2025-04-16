#!/bin/bash

#SBATCH --job-name=cl_sam2_ddp
#SBATCH --nodes=3
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --output=/scratch/gokuladethya.cse.nitt/fyp/slurm-loop-%j.out
#SBATCH --time=6-00:00:00

WATCH_DIR="/scratch/gokuladethya.cse.nitt/fyp/jobs"
PROCESSED_DIR="/scratch/gokuladethya.cse.nitt/fyp/processed_jobs"

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

echo "Using head node: $head_node ($head_node_ip)"
source /scratch/gokuladethya.cse.nitt/miniconda3/etc/profile.d/conda.sh
conda activate fyp

export LOGLEVEL=INFO
export TORCH_RUN_RDZV_TIMEOUT=360000
export TORCH_DISTRIBUTED_DEBUG=INFO
export WANDB_API_KEY=283c41dda88b658ba85c2d8ee7d37230f3341d8c
export WANDB_MODE=online

echo "Infinite job monitor running. Watching for scripts in $WATCH_DIR..."

while true; do
    for script in "$WATCH_DIR"/*.sh; do
        [ -e "$script" ] || continue  # No .sh file found

        script_name=$(basename "$script")

        # Skip if already processed
        if [ -f "$PROCESSED_DIR/$script_name" ]; then
            continue
        fi

        echo "[$(date)] Found new script: $script_name"
        chmod +x "$script"
        echo "Running $script_name with srun..."
        srun --nodes=3 --ntasks=6 bash "$script"

        # Mark as processed
        mv "$script" "$PROCESSED_DIR/$script_name"
        echo "[$(date)] Finished $script_name"
    done
    sleep 10  # Check every 10 seconds
done
