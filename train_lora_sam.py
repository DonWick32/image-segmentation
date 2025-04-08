import torch
import torch.distributed as dist
import torch.distributed
from torch.utils.data import DataLoader, DistributedSampler
from collections import defaultdict
from tqdm import tqdm
import os
import wandb
from datetime import timedelta

from sam2_dataset import get_train_dataloader
from sam2_model import get_model, _load_checkpoint

from SAM2.sam2.training.loss_fns import MultiStepMultiMasksAndIous
from lora_qkv import wrap_decoder_lora, wrap_image_encoder_lora, custom_save_lora_parameters, custom_load_lora_parameters
from omegaconf import OmegaConf
import gc
from evaluate import run_eval
from utils import Logger


def seed_everything(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_distributed():
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=5))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    node_rank = int(os.environ.get("NODE_RANK", 0))  # Default to 0 if not set

    # Global rank (unique ID for each process)
    global_rank = dist.get_rank()

    # Local rank (which GPU within a node)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # This is set by torchrun

    # World size (total number of processes across all nodes)
    world_size = dist.get_world_size()

    print(f"Node Rank: {node_rank}, Global Rank: {global_rank}, Local Rank (GPU): {local_rank}, World Size: {world_size}")


def cleanup_distributed():
    dist.destroy_process_group()

def is_main_process():
    return (dist.get_rank() == 0)

setup_distributed()

rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
device = torch.device(f"cuda:{local_rank}")


config = OmegaConf.load("lora_sam_config.yaml")

if is_main_process():
    logger = Logger(config, wandb_log=True)
    wandb.init(
        project="CL-SAM2",
        config=OmegaConf.to_container(config),
        notes=config.notes,
        entity=config.entity,
        mode="online"
    )
    track_file_Type = [".py", ".sh", ".yaml", "ipynb", ".json", ".txt"]
    wandb.run.log_code(".", include_fn=lambda path: (
        any([path.endswith(file_type) for file_type in track_file_Type])
        and ("wandb" not in path)
        and (config.output_dir not in path)
    ))
    run_id = wandb.run.id

    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
        
    os.mkdir(os.path.join(config.output_dir, run_id))
else:
    run_id = None
    

torch.distributed.barrier()

run_id = run_id if run_id else "ddp_run"

PATH = config.dataset.path
TEST_PATH = os.path.join(PATH, "SegSTRONGC_test/test/9/")
TEST_VIDS = [os.path.join(TEST_PATH, i) for i in sorted(os.listdir(TEST_PATH), key=lambda x: int(x))]
VAL_VIDS = TEST_VIDS[:1]
TEST_VIDS = TEST_VIDS[1:]

print("Test videos:", TEST_VIDS)
print("Val videos:", VAL_VIDS)


weight_dict = config.loss_weights
loss_cfg = config.loss_config
criterion = MultiStepMultiMasksAndIous(
        weight_dict=weight_dict,
        supervise_all_iou=loss_cfg.supervise_all_iou,
        iou_use_l1_loss=loss_cfg.iou_use_l1_loss,
        pred_obj_scores=loss_cfg.pred_obj_scores,
        focal_gamma_obj_score=loss_cfg.focal_gamma_obj_score,
        focal_alpha_obj_score=loss_cfg.focal_alpha_obj_score,
    )

model = get_model(config).train()
_load_checkpoint(model, config.model.checkpoint)

for param in model.parameters():
    param.requires_grad = False

if config.lora.decoder:
    wrap_decoder_lora(model, config.lora.rank)
if config.lora.image_encoder:
    wrap_image_encoder_lora(model, config.lora.rank)

model.to(device)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

if is_main_process():
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable LoRA parameters:", trainable_param_count)
    logger.log({"trainable_lora_params": trainable_param_count}, epoch_end_log=False)

trainable_params = [param for name, param in model.named_parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

DOMAINS = ['regular', 'blood', 'bg_change', 'smoke', 'low_brightness']

val_performance = {i:[] for i in DOMAINS}
train_performance = {i:[] for i in DOMAINS}
test_performance = {i:[] for i in DOMAINS}
log_metrics_history = {}


def train():
        if is_main_process():
            print(f"===================Training===================")

        train_loader, val_loader = get_train_dataloader(config)
        torch.distributed.barrier()
                
        # Wrap with DistributedSampler

        for epoch in range(0, config.epochs):
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
            
            model.train()
            model.module.training = True

            for batch in tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch+1}"):
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                losses = criterion(output, batch.masks)

                if is_main_process():
                    for k, v in losses.items():
                        logger.log({f"metric/train_loss_{k}": v.item(), "epoch": epoch + 1})

                loss_key, core_loss = losses.popitem()
                core_loss.backward()
                optimizer.step()

                del losses, batch, core_loss, output
                torch.cuda.empty_cache()
                gc.collect()

            with torch.no_grad():
                model.eval()
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    losses = criterion(output, batch.masks)

                    if is_main_process():
                        for k, v in losses.items():
                            logger.log({f"metric/val_loss_{k}": v.item(), "epoch": epoch + 1})

                    del losses, batch, output
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            if epoch%config.evaluate_every_n_epochs == 0:
                if is_main_process():
                    for domain_prev in DOMAINS:
                        print(f"Evaluating prev domain: {domain_prev} performance")
                        perf = run_eval(model.module, VAL_VIDS[0], domain_prev, os.path.join(config.dataset.annotation_path, f"test.json"))
                        for k, v in perf.items():
                            logger.log({f"metric/{domain_prev}/{k}": v})
                        print(f"Performance of {domain_prev} domain: {perf}")
                                
                torch.distributed.barrier()

                    
            torch.distributed.barrier()
            if is_main_process():
                custom_save_lora_parameters(model.module, os.path.join(config.output_dir, run_id, f"lora.pth"))
                

            logger.log_epoch_average()

if __name__ == '__main__':
    seed_everything()
    train()
    cleanup_distributed()
