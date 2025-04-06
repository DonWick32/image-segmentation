

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

PATH = "../segstrong"
TEST_PATH = os.path.join(PATH, "SegSTRONGC_test/test/9/")
VAL_PATH = os.path.join(PATH, "SegSTRONGC_val/val/1/")

test_vids = sorted(os.listdir(TEST_PATH), key=lambda x: int(x))
val_vids = sorted(os.listdir(VAL_PATH), key=lambda x: int(x))

print(test_vids, val_vids)

set_ = []

DOMAINS = ['blood', 'bg_change', 'regular', 'smoke', 'low_brightness']

for vids in [test_vids, val_vids]:
    temp_set = {}
    for domain in DOMAINS+['ground_truth']:
        temp_set[domain] = []
        for path in vids:
            path = os.path.join(TEST_PATH, path)
            temp_path = os.path.join(path, domain)
            
            for view in ['left', 'right']:
                temp_set[domain].append([])
                for img in os.listdir(os.path.join(temp_path, view)):
                    temp_set[domain][-1].append(os.path.join(temp_path, view, img))
                    
                temp_set[domain][-1] = sorted(temp_set[domain][-1], key=lambda x: int(x.split('/')[-1].split('.')[0]))
                
    set_.append(temp_set)
    
test_set, train_set = tuple(set_)
test_gt, train_gt = test_set['ground_truth'], train_set['ground_truth']
del test_set['ground_truth'], train_set['ground_truth']
        


# In[6]:


import cv2
from PIL import Image as PILImage
from iopath.common.file_io import g_pathmgr



import logging
import random
from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset
from SAM2.sam2.training.utils.data_utils import Frame, Object, VideoDatapoint


class VOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        videos,
        gt_frames,
    ):
        self.vidoes = videos
        self.gt_frames = gt_frames
        self._transforms = transforms
        self.cache_gt = {}

    def _get_datapoint(self, idx):
        # start_frame = np.random.randint(0, len(self.video_dataset[idx]) - 10)
        # end_frame = np.random.randint(start_frame + 5, len(self.video_dataset[idx]))
        start_frame, end_frame = 0, 3 #len(self.vidoes[idx])-1
        video = self.vidoes[idx][start_frame:end_frame]
        gt_frames = self.gt_frames[idx][start_frame:end_frame]
        frame_num_list = list(range(start_frame, end_frame))
        
        images = []
        frame_idx = 0
        rgb_images = load_images(video)
        
        have_sampled = False
        prev_connected_comp = None
        for gt_mask, frame_num in tqdm(zip(gt_frames, frame_num_list)):
            w, h = rgb_images[frame_num].size
            images.append(
                Frame(
                    data=rgb_images[frame_num],
                    objects=[],
                )
            )
            
            if f"{gt_mask}_{frame_num}" in self.cache_gt:
                objects = self.cache_gt[f"{gt_mask}_{frame_num}"]
            else:
                mask = cv2.imread(gt_mask, 0)
                objects = cv2.connectedComponents(mask)
                self.cache_gt[f"{gt_mask}_{frame_num}"] = deepcopy(objects)
                
            #     print("mask")
            #     plt.imshow(mask)
            #     plt.show()
                
            # print("init")
            # plt.imshow(objects[1])
            # plt.show()
            objects = objects[1]
            class_, class_count = np.unique(objects, return_counts=True)
            bg_class = class_[np.where(class_count == np.max(class_count))[0][0]]
            class_ = class_.tolist()
            
            if bg_class!=0:
                objects+=1
                objects[objects==0] = 5
                objects[objects==bg_class] = 0
                objects[objects==5] = bg_class
                bg_class = 0
                
            class_.remove(bg_class)
            
            
            if not have_sampled:
                # sample = np.random.choice(2)+1 # 1 or 2
                sample = 2
                sampled_objs = np.random.choice(class_, sample, replace=False)
                have_sampled = True
            else:
                ### calc iou and make sure obj id doesnt change due to connected comp random label allocation
                
                prev_connected_comp_temp = deepcopy(prev_connected_comp)
                objects_temp = deepcopy(objects)
                prev_connected_comp[prev_connected_comp!=1] = 0
                objects[objects!=1] = 0
                iou1 = np.logical_and(prev_connected_comp, objects)
        
                prev_connected_comp = deepcopy(prev_connected_comp_temp)
                objects = deepcopy(objects_temp)
                
                prev_connected_comp[prev_connected_comp==1] = 3
                prev_connected_comp[prev_connected_comp==2] = 1
                prev_connected_comp[prev_connected_comp==3] = 2
                prev_connected_comp[prev_connected_comp!=1] = 0
                objects[objects!=1] = 0
                
                iou2 = np.logical_and(prev_connected_comp, objects)
                objects = deepcopy(objects_temp)
                
                if iou2.sum() > iou1.sum():
                    print("swapped!!")
                    objects[objects==1] = 3
                    objects[objects==2] = 1
                    objects[objects==3] = 2
                    
                    
            prev_connected_comp = objects
            # print("final")
            # plt.imshow(objects)
            # plt.show()
            
            for objs in sampled_objs:
                mask = np.where(objects == objs, 1, 0).astype(np.uint8)
                images[frame_idx].objects.append(
                    Object(
                        object_id=objs-1,
                        frame_index=frame_num,
                        segment= torch.Tensor(mask),
                    )
                )
            
            frame_idx += 1

            
        datapoint = VideoDatapoint(
            frames=images,
            video_id=idx,
            size=(h, w),
        )

        datapoint = self._transforms(datapoint)
        return datapoint

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.vidoes)


def load_images(frames):
    all_images = []
    for path in tqdm(frames):
        with g_pathmgr.open(path, "rb") as fopen:
            all_images.append(PILImage.open(fopen).convert("RGB"))

    return all_images



from SAM2.sam2.training.dataset.transforms import (
    ComposeAPI,
    RandomHorizontalFlip,
    RandomAffine,
    RandomResizeAPI,
    ColorJitter,
    RandomGrayscale,
    ToTensorAPI,
    NormalizeAPI
)


train_transform = ComposeAPI([
    RandomHorizontalFlip(consistent_transform=True),
    RandomAffine(
        degrees=25,
        shear=20,
        image_interpolation="bilinear",
        consistent_transform=True,
    ),
    ColorJitter(
        consistent_transform=True,
        brightness=0.1,
        contrast=0.03,
        saturation=0.03,
        hue=None,
    ),
    RandomResizeAPI(
        sizes=1024,
        square=True,
        consistent_transform=True,
    ),
    ToTensorAPI(),
    NormalizeAPI(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


dataset = VOSDataset(transforms=train_transform, 
                     videos=test_set['blood'], 
                     gt_frames=test_gt)




from SAM2.sam2.sam2.modeling.sam2_base import SAM2Base
from SAM2.sam2.sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from SAM2.sam2.sam2.modeling.backbones.hieradet import Hiera
from SAM2.sam2.sam2.modeling.position_encoding import PositionEmbeddingSine
from SAM2.sam2.sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from SAM2.sam2.sam2.modeling.sam.transformer import RoPEAttention
from SAM2.sam2.sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from SAM2.sam2.sam2.modeling.position_encoding import PositionEmbeddingSine
from SAM2.sam2.sam2.modeling.backbones.hieradet import Hiera


trunk = Hiera(
    embed_dim=112,
    num_heads=2
)

position_encoding = PositionEmbeddingSine(
    num_pos_feats=256,
    normalize=True,
    scale=None,
    temperature=10000
)

neck = FpnNeck(
    position_encoding=position_encoding,
    d_model=256,
    backbone_channel_list=[896, 448, 224, 112],
    fpn_top_down_levels=[2, 3],
    fpn_interp_model='nearest'
)

image_encoder = ImageEncoder(
    trunk=trunk,
    neck=neck,
    scalp=1
)

self_attention = RoPEAttention(
    rope_theta=10000.0,
    feat_sizes=[64, 64],
    embedding_dim=256,
    num_heads=1,
    downsample_rate=1,
    dropout=0.1
)

cross_attention = RoPEAttention(
    rope_theta=10000.0,
    rope_k_repeat=True,
    feat_sizes=[64, 64],
    embedding_dim=256,
    num_heads=1,
    downsample_rate=1,
    dropout=0.1,
    kv_in_dim=64
)

memory_attention_layer = MemoryAttentionLayer(
    activation="relu",
    dim_feedforward=2048,
    dropout=0.1,
    pos_enc_at_attn=False,
    d_model=256,
    pos_enc_at_cross_attn_keys=True,
    pos_enc_at_cross_attn_queries=False,
    self_attention=self_attention,
    cross_attention=cross_attention
)

memory_attention = MemoryAttention(
    d_model=256,
    pos_enc_at_input=True,
    layer=memory_attention_layer,
    num_layers=4
)

position_encoding = PositionEmbeddingSine(
    num_pos_feats=64,
    normalize=True,
    scale=None,
    temperature=10000
)

mask_downsampler = MaskDownSampler(
    kernel_size=3,
    stride=2,
    padding=1
)

fuser_layer = CXBlock(
    dim=256,
    kernel_size=7,
    padding=3,
    layer_scale_init_value=1e-6,
    use_dwconv=True
)

fuser = Fuser(
    layer=fuser_layer,
    num_layers=2
)

memory_encoder = MemoryEncoder(
    out_dim=64,
    position_encoding=position_encoding,
    mask_downsampler=mask_downsampler,
    fuser=fuser
)


from SAM2.sam2.training.model.sam2 import SAM2Train
import importlib
import SAM2.sam2.training.utils.data_utils
importlib.reload(SAM2.sam2.training.model.sam2)
from SAM2.sam2.training.model.sam2 import SAM2Train
model = SAM2Train(
    image_encoder,
    memory_attention=memory_attention,
    memory_encoder=memory_encoder,
    num_maskmem=7,
    image_size=1024,
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    use_mask_input_as_output_without_sam=True,
    directly_add_no_mem_embed=True,
    use_high_res_features_in_sam=True,
    multimask_output_in_sam=True,
    iou_prediction_use_sigmoid=True,
    use_obj_ptrs_in_encoder=True,
    add_tpos_enc_to_obj_ptrs=False,
    only_obj_ptrs_in_the_past_for_eval=True,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    fixed_no_obj_ptr=True,
    multimask_output_for_tracking=True,
    use_multimask_token_for_obj_ptr=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    use_mlp_for_obj_ptr_proj=True,
    compile_image_encoder=False).to('cuda')
    
        # prob_to_use_pt_input_for_train=0.0,
        # prob_to_use_pt_input_for_eval=0.0,
        # prob_to_use_box_input_for_train=0.0,
        # prob_to_use_box_input_for_eval=0.0,
        # num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        # num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        # rand_frames_to_correct_for_train=False,
        # rand_frames_to_correct_for_eval=False,
        # num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        # num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        # rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        # rand_init_cond_frames_for_eval=False,
        # add_all_frames_to_correct_as_cond=False,
        # num_correction_pt_per_frame=7,
        # pt_sampling_for_eval="center",
        # prob_to_sample_from_gt_for_train=0.0,
        # use_act_ckpt_iterative_pt_sampling=False,
        # forward_backbone_per_frame_for_eval=False,
        # freeze_image_encoder=False)


import importlib
import SAM2.sam2.training.utils.data_utils
importlib.reload(SAM2.sam2.training.utils.data_utils)
from SAM2.sam2.training.utils.data_utils import collate_fn

from torch.utils.data import DataLoader
from functools import partial


dataset = VOSDataset(transforms=train_transform, 
                     videos=test_set['blood'], 
                     gt_frames=test_gt)


train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    collate_fn=partial(collate_fn, dict_key='all'),
    pin_memory=False,
)

batch = next(iter(train_loader))
batch = batch.to(
                "cuda"
            ) 


# In[13]:


model.training = True
with torch.no_grad():
    output = model(batch)
targets = batch.masks


# In[30]:


from SAM2.sam2.training.loss_fns import MultiStepMultiMasksAndIous


        
weight_dict = {'loss_mask': 20,
                'loss_dice': 1,
                'loss_iou': 1,
                'loss_class': 1}

criterion = MultiStepMultiMasksAndIous(weight_dict=weight_dict)
loss = criterion(output, targets)


# In[31]:


loss


# In[32]:


criterion = MultiStepMultiMasksAndIous(weight_dict=weight_dict, supervise_all_iou= True,
                                    iou_use_l1_loss = True,
                                    pred_obj_scores = True,
                                    focal_gamma_obj_score = 0.0,
                                    focal_alpha_obj_score= -1.0,)

loss = criterion(output, targets)
loss


# In[35]:


len(loss)


# In[28]:


loss_key, loss = loss.popitem()
loss_key, loss

