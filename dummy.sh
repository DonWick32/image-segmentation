#!/bin/bash

echo "STARTING DUMMY JOB: $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
sleep 30  # Simulate some work
echo "ENDING DUMMY JOB: $(date)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"