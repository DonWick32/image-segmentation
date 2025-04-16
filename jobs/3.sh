torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  /scratch/gokuladethya.cse.nitt/image-segmentation/train_cl.py \
  --notes "CL-KD Loss" \
  --cl_config.knowledge_distillation 0.1 \
  --cl_kmean.reset_lora false \
  --cl_kmean.knowledge_distillation true
