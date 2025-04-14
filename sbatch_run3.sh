#!/bin/bash

#SBATCH --job-name=cl_sam2_ddp
#SBATCH --nodes=3
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --output=/scratch/gokuladethya.cse.nitt/fyp/slurm-%j.out
#SBATCH --time=4-00:00:00

echo "Allocated Gokul node: jobid:"
squeue -a | grep gok
echo "------------------------------------"

# Get hostnames of allocated nodes
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
echo "SLURM_NODELIST: $SLURM_JOB_NODELIST"
scontrol show hostnames $SLURM_JOB_NODELIST

nodes_array=($nodes)
head_node=${nodes_array[0]}

# Get IP address of the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"

export LOGLEVEL=INFO

# Setup environment
conda init bash
source /scratch/gokuladethya.cse.nitt/miniconda3/etc/profile.d/conda.sh

conda activate fyp
export WANDB_API_KEY=283c41dda88b658ba85c2d8ee7d37230f3341d8c
# Diagnostics
srun python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
srun nvidia-smi

# Launch training
echo "Launching torchrun..."
ls /scratch/gokuladethya.cse.nitt/image-segmentation/

export TORCH_RUN_RDZV_TIMEOUT=360000
export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCHELASTIC_ENABLE_FILE_TIMER=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=0


   srun torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  /scratch/gokuladethya.cse.nitt/image-segmentation/train_single_domain.py \
  --domain "regular" \
  --learning_rate 0.0001 \
  --epochs 20 \
  --evaluate_every_n_epochs 2 \
  --dataset.max_frames 4 \
  --dataset.max_frame_interval_skip 3 \
  --lora.decoder false \
  --lora.image_encoder true \
  --lora.rank 8

    srun torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  /scratch/gokuladethya.cse.nitt/image-segmentation/train_single_domain.py \
  --domain "bg_change" \
  --learning_rate 0.0001 \
  --epochs 20 \
  --evaluate_every_n_epochs 2 \
  --dataset.max_frames 4 \
  --dataset.max_frame_interval_skip 3 \
  --lora.decoder false \
  --lora.image_encoder true \
  --lora.rank 8


  srun torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  /scratch/gokuladethya.cse.nitt/image-segmentation/train_single_domain.py \
  --domain "smoke" \
  --learning_rate 0.0001 \
  --epochs 20 \
  --evaluate_every_n_epochs 2 \
  --dataset.max_frames 4 \
  --dataset.max_frame_interval_skip 3 \
  --lora.decoder false \
  --lora.image_encoder true \
  --lora.rank 8

    srun torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  /scratch/gokuladethya.cse.nitt/image-segmentation/train_single_domain.py \
  --domain "low_brightness" \
  --learning_rate 0.0001 \
  --epochs 20 \
  --evaluate_every_n_epochs 2 \
  --dataset.max_frames 4 \
  --dataset.max_frame_interval_skip 3 \
  --lora.decoder false \
  --lora.image_encoder true \
  --lora.rank 8



# CL-baseline-naive, CL-KD Loss, CL-Kmean

  # srun torchrun \
#   --nnodes=3 \
#   --nproc_per_node=2 \
#   --rdzv_id=$RANDOM \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$head_node_ip:29500 \
#   /scratch/gokuladethya.cse.nitt/image-segmentation/train_single_domain.py \
#   --domain "smoke" \
#   --learning_rate 0.0001 \
#   --epochs 100 \
#   --evaluate_every_n_epochs 2 \
#   --dataset.max_frames 4 \
#   --dataset.max_frame_interval_skip 3 \
#   --lora.decoder false \
#   --lora.image_encoder true \
#   --lora.rank 8
 


 ############ ALL THE PARAMETERS FOR CL-KMEAN/CL-Regularization/Hybrid TRAINING FOR REFERENCE ############



# srun torchrun \
#   --nnodes=3 \
#   --nproc_per_node=2 \
#   --rdzv_id=$RANDOM \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$head_node_ip:29500 \
#   /scratch/gokuladethya.cse.nitt/image-segmentation/train_cl_kmean.py \
#   --notes "CL-Kmean" \
#   --batch_size 2 \
#   --num_workers 0 \
#   --learning_rate 0.0001 \
#   --epochs 5 \
#   --distributed true \
#   --model.checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
#   --dataset.max_frames 4 \
#   --dataset.max_frame_interval_skip 3 \
#   --loss_weights.loss_mask 20 \
#   --loss_weights.loss_dice 1 \
#   --loss_weights.loss_iou 1 \
#   --loss_weights.loss_class 1 \
#   --loss_config.supervise_all_iou true \
#   --loss_config.iou_use_l1_loss true \
#   --loss_config.pred_obj_scores true \
#   --loss_config.focal_gamma_obj_score 0.0 \
#   --loss_config.focal_alpha_obj_score -1.0 \
#   --cl_config.evaluate_every_n_epochs 10 \
#   --cl_config.knowledge_distillation 0.1 \
#   --cl_kmean.reset_lora true \
#   --cl_kmean.knowledge_distillation false \
#   --lora.decoder false \
#   --lora.image_encoder true \
#   --lora.rank 8

 ############ ALL THE PARAMETERS FOR SINGLE DOMAIN TRAINING FOR REFERENCE ############



# srun torchrun \
#   --nnodes=3 \
#   --nproc_per_node=2 \
#   --rdzv_id=$RANDOM \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$head_node_ip:29500 \
#   /scratch/gokuladethya.cse.nitt/image-segmentation/train_single_domain.py \
#   --domain "smoke" \
#   --notes "Single domain smoke training" \
#   --batch_size 2 \
#   --num_workers 0 \
#   --learning_rate 0.0001 \
#   --epochs 100 \
#   --distributed true \
#   --evaluate_every_n_epochs 2 \
#   --dataset.max_frames 4 \
#   --dataset.max_frame_interval_skip 3 \
#   --loss_weights.loss_mask 20 \
#   --loss_weights.loss_dice 1 \
#   --loss_weights.loss_iou 1 \
#   --loss_weights.loss_class 1 \
#   --loss_config.supervise_all_iou true \
#   --loss_config.iou_use_l1_loss true \
#   --loss_config.pred_obj_scores true \
#   --loss_config.focal_gamma_obj_score 0.0 \
#   --loss_config.focal_alpha_obj_score -1.0 \
#   --lora.decoder false \
#   --lora.image_encoder true \
#   --lora.rank 8
 