# sam2_dataset.py
from SAM.sam2.training.utils.data_utils import Frame, Object, VideoDatapoint
from torchvision.datasets.vision import VisionDataset
from PIL import Image as PILImage
from iopath.common.file_io import g_pathmgr
from functools import partial
from SAM.sam2.training.utils.data_utils import collate_fn
import numpy as np
import os, cv2
from tqdm.auto import tqdm
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from SAM.sam2.training.dataset.transforms import (
    ComposeAPI, RandomHorizontalFlip, RandomAffine,
    RandomResizeAPI, ColorJitter, ToTensorAPI,
    NormalizeAPI
)


class VOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        videos,
        gt_frames,
        max_frames=3,
        min_frames=1,
        max_frame_interval_skip=3, ## min: 1, choose every nth frame
    ):
        self.vidoes = videos
        self.partition_vids = []
        for vid_n in range(len(videos)):
            for i in range(0, 300-max_frames+1):
                self.partition_vids.append([vid_n, i])
            
        self.gt_frames = gt_frames
        self._transforms = transforms
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.max_frame_interval_skip = max_frame_interval_skip
        self.cache_gt = {}

    def _get_datapoint(self, idx):
        # start_frame = np.random.randint(0, len(self.video_dataset[idx]) - 10)
        # end_frame = np.random.randint(start_frame + 5, len(self.video_dataset[idx]))
        vid_n, start_frame = self.partition_vids[idx] #297
        total_frames = self.max_frames
        max_interval = min(self.max_frame_interval_skip, (300 - start_frame) // total_frames)
        frame_skip = np.random.randint(1, max_interval+1)
        end_frame = start_frame + total_frames * frame_skip
        
        video = self.vidoes[vid_n][start_frame:end_frame:frame_skip]
        gt_frames = self.gt_frames[vid_n][start_frame:end_frame:frame_skip]
        
        ## randomize the direction of the video
        if np.random.random() > 0.5:
            video = video[::-1]
            gt_frames = gt_frames[::-1]
        
        frame_num_list = list(range(0, end_frame-start_frame))
        
        images = []
        frame_idx = 0
        rgb_images = load_images(video)
        
        have_sampled = False
        prev_connected_comp = None

        
        for gt_mask, frame_num in zip(gt_frames, frame_num_list):
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
                sampled_objs = np.random.choice([1,2], sample, replace=False)
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
        return 10


def load_images(frames):
    all_images = []
    for path in frames:
        with g_pathmgr.open(path, "rb") as fopen:
            all_images.append(PILImage.open(fopen).convert("RGB"))

    return all_images


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


val_transform = ComposeAPI([
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




def get_set(config):
    PATH = config.dataset.path
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

    val_set = {}
    val_gt = []
    for domain in test_set:
        val_set[domain] = test_set[domain][:2]
        test_set[domain] = test_set[domain][2:]

    val_gt = test_gt[:2]
    test_gt = test_gt[2:]

    
    return test_set, train_set, test_gt, train_gt, val_set, val_gt


def get_dataloader(domain, config):

    test_set, train_set, test_gt, train_gt, val_set, val_gt = get_set(config)

    train_dataset = VOSDataset(
        transforms=train_transform, 
        videos=train_set[domain], 
        gt_frames=train_gt,
        max_frames=config.dataset.max_frames,
        min_frames=config.dataset.min_frames,
        max_frame_interval_skip=config.dataset.max_frame_interval_skip,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn, dict_key='all'),
        pin_memory=False,
        sampler=DistributedSampler(train_dataset, shuffle=True) if config.distributed else None,
    )
    
    val_dataset = VOSDataset(
        transforms=val_transform, 
        videos=val_set[domain], 
        gt_frames=val_gt,
        max_frames=config.dataset.max_frames,
        min_frames=config.dataset.min_frames,
        max_frame_interval_skip=config.dataset.max_frame_interval_skip,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn, dict_key='all'),
        pin_memory=False,
        sampler=DistributedSampler(val_dataset, shuffle=False) if config.distributed else None,
    )
    
    return train_loader, val_loader
    
    