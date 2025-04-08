import torch
import torch.distributed as dist
import torch.distributed
from torch.utils.data import DataLoader, DistributedSampler
from collections import defaultdict
from tqdm import tqdm
import os
import wandb
from datetime import timedelta

from sam2_dataset import get_dataloader
from sam2_model import get_model, _load_checkpoint

from SAM2.sam2.training.loss_fns import MultiStepMultiMasksAndIous
from lora_qkv import wrap_decoder_lora, wrap_image_encoder_lora, custom_save_lora_parameters, custom_load_lora_parameters
from omegaconf import OmegaConf
import gc
from evaluate import run_eval
from utils import calculate_forgetting, insert_perf, rm_output_keys, Logger

from torch.distributed.elastic.multiprocessing.errors import record

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

DOMAINS = ['regular', 'blood', 'bg_change', 'smoke', 'low_brightness']

config = OmegaConf.load("config.yaml")

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
TRAIN_PATH = os.path.join(PATH, "SegSTRONGC_val/val/1/")
TRAIN_VIDS = [os.path.join(TRAIN_PATH, i) for i in sorted(os.listdir(TRAIN_PATH), key=lambda x: int(x))]
TEST_VIDS = [os.path.join(TEST_PATH, i) for i in sorted(os.listdir(TEST_PATH), key=lambda x: int(x))]
VAL_VIDS = TEST_VIDS[:1]
TEST_VIDS = TEST_VIDS[1:]

print("Test videos:", TEST_VIDS)
print("Train videos:", TRAIN_VIDS)
print("Val videos:", VAL_VIDS)

monitor_vids = {'train': TRAIN_VIDS[0], 'val': VAL_VIDS[0]}

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

weight_dict['loss_iou'] = 0
kd_criterion = MultiStepMultiMasksAndIous(
        weight_dict=weight_dict,
        supervise_all_iou=loss_cfg.supervise_all_iou,
        iou_use_l1_loss=loss_cfg,
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

val_performance = {i:[] for i in DOMAINS}
train_performance = {i:[] for i in DOMAINS}
test_performance = {i:[] for i in DOMAINS}
log_metrics_history = {}


prev_domain = None
@record
def train():
    for domain_idx, domain in enumerate(DOMAINS):
        torch.distributed.barrier()
        if is_main_process():
            print(f"===================Training on domain: {domain}===================")

        train_loader, val_loader = get_dataloader(domain, config)

        # Wrap with DistributedSampler

        for epoch in range(0, config.epochs):
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
                    
            model.module.training = False
            if (epoch == config.epochs - 1) or (epoch % config.cl_config.evaluate_every_n_epochs == 0):
                for perf_list, type_ in zip([val_performance, train_performance], ['val', 'train']):
                    print(f"Evaluating {type_} performance from current domain {domain}")
                    perf_total = {}
                    for domain_prev in DOMAINS[:domain_idx+1]:
                        print(f"Evaluating prev domain: {domain_prev} performance")
                        annot_file = "val" if type_ == "train" else "test"
                        perf = run_eval(model.module, monitor_vids[type_], domain, os.path.join(config.dataset.annotation_path, f"{annot_file}.json"))
                        perf_total[domain_prev] = perf
                        for k, v in perf.items():
                            logger.log({f"{type_}_perf/{domain_prev}/{k}": v})
                        print(f"Performance of {domain_prev} domain: {perf}")
                    insert_perf(perf_list, perf_total)
                    calculate_forgetting(perf_list, domain_idx, config, logger, tag=type_)
                    
            torch.distributed.barrier()
            model.train()
            model.module.training = True

            for batch in tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch+1}"):
                batch = batch.to(device)
                
                if prev_domain is not None:
                    with torch.no_grad():
                        custom_save_lora_parameters(model.module, os.path.join(config.output_dir, run_id, f"curr_lora_{domain}.pth"))
                        custom_load_lora_parameters(model.module, os.path.join(config.output_dir, run_id, f"lora_{prev_domain}.pth"))
                        output_old = model(batch)
                        output_old = torch.stack([output_old[i]['multistep_pred_masks_high_res'].squeeze() for i in range(len(output))], 0)
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        custom_load_lora_parameters(model.module, os.path.join(config.output_dir, run_id, f"curr_lora_{domain}.pth"))
                
                optimizer.zero_grad()
                
                output = model(batch)
                rm_output_keys(output)
                gc.collect()
                torch.cuda.empty_cache()
                
                losses = criterion(output, batch.masks)
                kd_loss = kd_criterion(output, output_old)
                losses['kd_loss'] = kd_loss['core_loss']
    

                if is_main_process():
                    for k, v in losses.items():
                        logger.log({f"metric/train_loss_{k}": v.item(), "epoch": epoch + 1})
                        
                    for k, v in kd_loss.items():
                        logger.log({f"metric/train_loss_kd_{k}": v.item(), "epoch": epoch + 1})

                core_loss = losses['core_loss']
                core_loss = config.loss_weights.knowledge_distillation * kd_loss + (1-config.loss_weights.knowledge_distillation) * core_loss
                core_loss.backward()
                optimizer.step()

                del losses, batch, core_loss, output, kd_loss, output_old
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
                            print(k, v.item())

                    del losses, batch, output
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            torch.distributed.barrier()
            if is_main_process():
                custom_save_lora_parameters(model.module, os.path.join(config.output_dir, run_id, f"lora_{domain}.pth"))
                
                model.module.training = False

                if epoch == config.epochs - 1:
                    print(f"Evaluating test performance from current domain {domain}")
                    perf_total = {}
                    for domain_prev in DOMAINS[:domain_idx+1]:
                        print(f"Evaluating prev domain: {domain_prev} performance")
                        perf_total[domain_prev] = []
                        for i, vids in enumerate(TEST_VIDS):
                            perf = run_eval(model.module, vids, domain, os.path.join(config.dataset.annotation_path, "test.json"))
                            perf_total[domain_prev].append(perf)
                            for k, v in perf.items():
                                logger.log({f"test_perf/{domain_prev}/vid_{i}/{k}": v})
                            print(f"{vids} Performance of {domain_prev} domain: {perf}")
                    insert_perf(test_performance, perf_total)
                    calculate_forgetting(test_performance, domain_idx, config, logger)
                    
                logger.log_epoch_average()
                    
    if is_main_process():
        print(f"Saving wts of {domain} domain")
        custom_save_lora_parameters(model.module, os.path.join(config.output_dir, run_id, f"lora_{domain}.pth"))
        prev_domain = domain
    

if __name__ == '__main__':
    seed_everything()
    try:
        train()
    except Exception as e:
        print(f"Rank {dist.get_rank()} failed: {e}")

    cleanup_distributed()
