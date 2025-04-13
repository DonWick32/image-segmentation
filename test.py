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
from utils import calculate_forgetting, insert_perf, rm_output_keys, Logger, override_config_with_args

from torch.distributed.elastic.multiprocessing.errors import record

DOMAINS = ['smoke', 'blood', 'low_brightness', 'bg_change', 'regular']


config = OmegaConf.load("config.yaml")
config.dataset.path = "../segstrong"

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

model = get_model(config).to('cuda').train()
_load_checkpoint(model, config.model.checkpoint)

model.training = False
annot_file = "val"
perf = run_eval(model, monitor_vids['train'], 'smoke', os.path.join(config.dataset.annotation_path, f"{annot_file}.json"))
print(perf)