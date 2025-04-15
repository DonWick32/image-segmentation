# %%
import os
import numpy as np
import torch
from PIL import Image
import cv2 
from ultralytics import YOLO
from matplotlib import pyplot as plt
import json
import pickle
import logging
from tqdm.auto import tqdm
import time
import gc
import shutil
import requests
from torchvision.ops import box_convert, nms

from SAM2.sam2.sam2.build_sam import build_sam2_video_predictor
from groundingdino.util.inference import load_model, load_image, predict

# %%
# !sudo mount -t drvfs E: /mnt/g
# mogrify -format jpg *.png && rm *.png

# %%
RUN_DESCRIPTION = "sam2 with grounding dino box prompts. only first frame boxes are used. no tta. boxes were precomputed by using nms and other shit"
SPLITS_TO_RUN = ["test", "val", "train"]
TOTAL_FRAMES_PER_VIDEO = 300

# dataset paths
DATASET_ROOT_PATH = "data/raw"
VAL_PATH = DATASET_ROOT_PATH + "/SegSTRONGC_val/val"
TEST_PATH = DATASET_ROOT_PATH + "/SegSTRONGC_test/test"
TRIAN_PATH = DATASET_ROOT_PATH + "/SegSTRONGC_train/train"
USE_GOKUL_SPLIT = False

VAL_FOLDERS = {"1": ["0", "1", "2"]}
TEST_FOLDERS = {"9": ["0", "1", "2"]} 
TRAIN_FOLDERS = {"3": ["0", "2"], "4": ["0", "1", "2"], "5": ["0", "2"], "7": ["0", "1"], "8": ["1", "2"]}

GOKUL_VAL_FOLDERS = {"9": ["0", "1"]}
GOKUL_TEST_FOLDERS = {"9": ["2"]}
GOKUL_TRAIN_FOLDERS = {"1": ["0", "1", "2"]}

# domains (ground truth is 'ground_truth')
VAL_DOMAINS = ['bg_change', 'blood', 'low_brightness', 'regular', 'smoke']
TEST_DOMAINS = ['bg_change', 'blood', 'low_brightness', 'regular', 'smoke']
TRAIN_DOMAINS = ['regular']

# prompts
SHOULD_USE_MANUAL_PROMPT = False
SHOULD_USE_BOX_PROMPT = True
SHOULD_SAMPLE_GROUND_TRUTH = False
SHOULD_VISUALIZE_PROMPTS = True
REFRESH_PROMPTS = False
NUM_POS_POINTS_PER_TOOL = 5
NUM_NEG_POINTS_PER_TOOL = 5
PROMPTS_ROOT_PATH = "data/prompts"
PROMPTING_STRATEGIES = ["first", "all", "dynamic"]
PROMPTING_STRATEGY = PROMPTING_STRATEGIES[0]
MODEL_PROMPT_CAPTION = "tool"
MODEL_PROMPT_BOX_THRESHOLD = 0.2
MODEL_PROMPT_TEXT_THRESHOLD = 0.25
MODEL_PROMPT_NMS_THRESHOLD = 0.3
MODEL_PROMPT_AREA_THRESHOLD = 0.9

# test time adaptation
SHOULD_PERFROM_CYCLIC_TTA = False

# results
SAVE_RUN_MASK_LOGITS = False
SAVE_IMAGES_ONCE = True
BASE_RESULTS_DIR = "data/results"
MASKS_DIR = BASE_RESULTS_DIR + "/masks"

# models
MODELS = ["sam2.1_hiera_base_plus", "yolo11x-seg", "groundingdino-swinb"]
INFERENCE_MODEL = MODELS[0]
PROMPT_MODEL = MODELS[2]
CHECKPOINTS = {
    "sam2.1_hiera_base_plus": "checkpoints/sam2.1_hiera_base_plus.pt",
    "yolo11x-seg": "checkpoints/yolo11x-seg.pt",
    "groundingdino-swinb": "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
}
MODEL_CONFIGS = {
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "yolo11x-seg": None,
    "groundingdino-swinb": "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
}

# logs
LOG_DIR =  "logs"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1360663640046571580/mNocZ3tLWiUVaMQTnOWqVRJU-HdI9onQuw0Wcr1xn8ZxRdvY51kuf9IcZ2qxRIBE21-x"

# %%
def get_image_paths(path, domain, is_left, num_images=300):
    stereo_dir = "left" if is_left else "right"
    image_paths = []

    for i in range(num_images):
        image_paths.append(path + "/" + domain + "/" + stereo_dir + "/" + str(i) + ".png")

    return image_paths

def get_video_paths(base_path, folders, video_folders):
    return [f"{base_path}/{folder}/{video}" 
            for folder in folders 
            for video in video_folders[folder]]

# %%
if not USE_GOKUL_SPLIT:
    val_video_folders_path = get_video_paths(VAL_PATH, VAL_FOLDERS.keys(), VAL_FOLDERS)
    test_video_folders_path = get_video_paths(TEST_PATH, TEST_FOLDERS.keys(), TEST_FOLDERS)
    train_video_folders_path = get_video_paths(TRIAN_PATH, TRAIN_FOLDERS.keys(), TRAIN_FOLDERS)
else:
    val_video_folders_path = get_video_paths(TEST_PATH, GOKUL_VAL_FOLDERS.keys(), GOKUL_VAL_FOLDERS)  
    test_video_folders_path = get_video_paths(TEST_PATH, GOKUL_TEST_FOLDERS.keys(), GOKUL_TEST_FOLDERS)
    train_video_folders_path = get_video_paths(VAL_PATH, GOKUL_TRAIN_FOLDERS.keys(), GOKUL_TRAIN_FOLDERS)

if "val" not in SPLITS_TO_RUN:
    val_video_folders_path = []
if "test" not in SPLITS_TO_RUN:
    test_video_folders_path = []
if "train" not in SPLITS_TO_RUN:
    train_video_folders_path = []

inference_model_checkpoint = CHECKPOINTS[INFERENCE_MODEL]
inference_model_config = MODEL_CONFIGS[INFERENCE_MODEL]

prompt_model_checkpoint = CHECKPOINTS[PROMPT_MODEL]
prompt_model_config = MODEL_CONFIGS[PROMPT_MODEL]

logging.basicConfig(filename=LOG_DIR + f'/{INFERENCE_MODEL}.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='a')
LOGGER = logging.getLogger()

PROMPTS_FOLDER = PROMPTS_ROOT_PATH
PROMPTS_FOLDER += "/manual" if SHOULD_USE_MANUAL_PROMPT else "/auto"
PROMPTS_FOLDER += "/box" if SHOULD_USE_BOX_PROMPT else "/point"
if SHOULD_USE_BOX_PROMPT:
    PROMPTS_FOLDER += "/ground_truth" if SHOULD_SAMPLE_GROUND_TRUTH else "/groundingdino"
else:
    PROMPTS_FOLDER += f"/{NUM_POS_POINTS_PER_TOOL}-pos_{NUM_NEG_POINTS_PER_TOOL}-neg"

if not SHOULD_USE_MANUAL_PROMPT and SHOULD_USE_BOX_PROMPT and not SHOULD_SAMPLE_GROUND_TRUTH:
    prompt_model = load_model(prompt_model_config, prompt_model_checkpoint)

print("val_video_folders_path", val_video_folders_path)
print("test_video_folders_path", test_video_folders_path)
print("train_video_folders_path", train_video_folders_path)
print("inference_model_name", INFERENCE_MODEL)
print("prompts_folder", PROMPTS_FOLDER)

# %%
def calculate_iou(TP, FP, FN):
    return TP / (TP + FP + FN)

def calculate_dsc(TP, FP, FN):
    return 2 * TP / (2 * TP + FP + FN)

def calculate_miou(pred_masks, gt_masks):
    ious = []
    for i in range(len(pred_masks)):
        TP = np.logical_and(pred_masks[i], gt_masks[i])
        FP = np.logical_and(pred_masks[i], np.logical_not(gt_masks[i]))
        FN = np.logical_and(np.logical_not(pred_masks[i]), gt_masks[i])

        iou = calculate_iou(np.sum(TP), np.sum(FP), np.sum(FN))
        ious.append(iou)
    
    return np.mean(ious)

def calculate_mdsc(pred_masks, gt_masks):
    dscs = []
    for i in range(len(pred_masks)):
        TP = np.logical_and(pred_masks[i], gt_masks[i])
        FP = np.logical_and(pred_masks[i], np.logical_not(gt_masks[i]))
        FN = np.logical_and(np.logical_not(pred_masks[i]), gt_masks[i])

        dsc = calculate_dsc(np.sum(TP), np.sum(FP), np.sum(FN))
        dscs.append(dsc)
    
    return np.mean(dscs)

# %%
def manual_annotate(frame_path):
    annotations = {0: []}
    current_tool = 0
    is_positive = True

    window_name = "Manual Annotation of Frame -" + str(frame_path)
    cv2.namedWindow(window_name)

    def handle_mouse_click(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if annotations[current_tool]["points"] is None:
                annotations[current_tool]["points"] = []

            if annotations[current_tool]["labels"] is None:
                annotations[current_tool]["labels"] = []
            
            annotations[current_tool]["points"].append([x, y])
            annotations[current_tool]["labels"].append(1 if is_positive else 0)

            if is_positive:
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            else:
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        cv2.imshow(window_name, frame)

    cv2.setMouseCallback(window_name, handle_mouse_click)
    frame = cv2.imread(frame_path)
    original_frame = frame.copy()

    while True:
        frame = original_frame.copy()
        for tool in annotations:
            if annotations[tool]["points"] is not None:
                for point, label in zip(annotations[tool]["points"], annotations[tool]["labels"]):
                    if label == 1:
                        cv2.circle(frame, (point[0], point[1]), 10, (0, 255, 0), -1)
                    else:
                        cv2.circle(frame, (point[0], point[1]), 10, (0, 0, 255), -1)
                
        display_text = f"Tool: {current_tool}, Mode: {'Positive' if is_positive else 'Negative'}"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):
            current_tool += 1
            is_positive = True
            annotations[current_tool] = []
        elif key == ord("p"):
            is_positive = not is_positive
        elif key == ord("c"):
            annotations[current_tool] = []
        elif key == ord("s"):
            break

    cv2.destroyWindow(window_name)
    return annotations

def auto_box_ground_truth_annotate(frame_path):
    ground_truth_mask_path = None
    annotations = {}
    # annotations[current_tool].append({
    #             "x": x,
    #             "y": y,
    #             "label": 1 if is_positive else 0
    #         })

    for domain in VAL_DOMAINS:
        if domain in frame_path:
            ground_truth_mask_path = frame_path.replace(domain, "ground_truth")
            break
    
    if ground_truth_mask_path is None:
        raise ValueError("Ground truth path not found.")
    else:
        ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
        ground_truth_mask = (ground_truth_mask > 0).astype(np.bool_)
        _, labels = cv2.connectedComponents(ground_truth_mask.astype(np.uint8))

        # get unique labels, and get count for each label, then sort by count in descending order
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_labels = unique_labels[np.argsort(-counts)]

        background_label = sorted_labels[0]
        first_tool_label = sorted_labels[1]

        if len(sorted_labels) > 2:
            second_tool_label = sorted_labels[2]
        else:
            second_tool_label = -1

        #for each tool, get the centroid, and add num_auto_points - 1 sampled random points
        annotations[0] = []
        annotations[1] = []

        for label, obj_id in [(first_tool_label, 0), (second_tool_label, 1)]:

            if label == -1:
                continue

            label_indices = np.where(labels == label)
            all_points = []

            for i in range(len(label_indices[0])):
                x = int(label_indices[1][i])
                y = int(label_indices[0][i])
                all_points.append((x, y))

            rightmost = max(all_points, key=lambda x: x[0])[0]
            leftmost = min(all_points, key=lambda x: x[0])[0]
            topmost = min(all_points, key=lambda x: x[1])[1]
            bottommost = max(all_points, key=lambda x: x[1])[1]

            x1 = leftmost
            y1 = topmost
            x2 = rightmost
            y2 = bottommost

            annotations[obj_id]["bbox"] = {
                "box": [x1, y1, x2, y2],
                "conf": 1.0
            }

        return annotations

def auto_box_model_annotate(frame_path):
    # print("path: ", frame_path)
    annotations = {}
    img_src, img = load_image(frame_path)

    boxes, logits, _ = predict(
        model=prompt_model,
        image=img,
        caption=MODEL_PROMPT_CAPTION,
        box_threshold=MODEL_PROMPT_BOX_THRESHOLD,
        text_threshold=MODEL_PROMPT_TEXT_THRESHOLD
    )

    h, w, _ = img_src.shape
    all_boxes_xyxy = box_convert(boxes=boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy")
    all_conf = logits
    all_indices = np.arange(len(all_boxes_xyxy))

    nms_indices = nms(all_boxes_xyxy, all_conf, MODEL_PROMPT_NMS_THRESHOLD).numpy().tolist()

    for i in range(len(nms_indices)):
        box = all_boxes_xyxy[nms_indices[i]]

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        area = (x2 - x1) * (y2 - y1)
        if area > MODEL_PROMPT_AREA_THRESHOLD * (w * h):
            if len(nms_indices) > 1:
                nms_indices.pop(i)
            break

    indices_to_remove = []
    for i in range(len(nms_indices)):
        for j in range(i + 1, len(nms_indices)):
            box1 = all_boxes_xyxy[nms_indices[i]]
            box2 = all_boxes_xyxy[nms_indices[j]]

            box1_x1 = box1[0]
            box1_y1 = box1[1]
            box1_x2 = box1[2]
            box1_y2 = box1[3]

            box2_x1 = box2[0]
            box2_y1 = box2[1]
            box2_x2 = box2[2]
            box2_y2 = box2[3]

            area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
            area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

            x_overlap = max(0, min(box1_x2, box2_x2) - max(box1_x1, box2_x1))
            y_overlap = max(0, min(box1_y2, box2_y2) - max(box1_y1, box2_y1))
            intersection = x_overlap * y_overlap

            union = area1 + area2 - intersection

            larger_area = max(area1, area2)

            if union * MODEL_PROMPT_AREA_THRESHOLD <= larger_area:
                if area1 < area2:
                    if nms_indices[i] not in indices_to_remove:
                        if len(set(nms_indices) - set(indices_to_remove)) > 1:
                            indices_to_remove.append(nms_indices[i])
                else:
                    if nms_indices[j] not in indices_to_remove:
                        if len(set(nms_indices) - set(indices_to_remove)) > 1:
                            indices_to_remove.append(nms_indices[j])
    
    nms_indices = list(set(nms_indices) - set(indices_to_remove))
    final_indices = nms_indices.copy()

    if len(final_indices) > 0:
        final_boxes = all_boxes_xyxy[final_indices]
        final_conf = all_conf[final_indices]

        sort_indices = np.argsort(final_boxes[:, 0])
        final_boxes = final_boxes[sort_indices]
        final_conf = final_conf[sort_indices]

        for i in range(min(len(final_boxes), 2)):
            x1, y1, x2, y2 = map(int, final_boxes[i])
            conf = final_conf[i]

            if i not in annotations:
                annotations[i] = {}

            annotations[i]["bbox"] = {
                "box": [x1, y1, x2, y2],
                "conf": float(conf)
            }

    return annotations

def auto_point_annotate(frame_path):
    ground_truth_mask_path = None
    annotations = {}
    # annotations[current_tool].append({
    #             "x": x,
    #             "y": y,
    #             "label": 1 if is_positive else 0
    #         })

    for domain in VAL_DOMAINS:
        if domain in frame_path:
            ground_truth_mask_path = frame_path.replace(domain, "ground_truth")
            break
    
    if ground_truth_mask_path is None:
        raise ValueError("Ground truth path not found.")
    else:
        ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
        ground_truth_mask = (ground_truth_mask > 0).astype(np.bool_)
        _, labels = cv2.connectedComponents(ground_truth_mask.astype(np.uint8))

        # get unique labels, and get count for each label, then sort by count in descending order
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_labels = unique_labels[np.argsort(-counts)]

        background_label = sorted_labels[0]
        first_tool_label = sorted_labels[1]

        if len(sorted_labels) > 2:
            second_tool_label = sorted_labels[2]
        else:
            second_tool_label = -1

        #for each tool, get the centroid, and add num_auto_points - 1 sampled random points
        annotations[0] = []
        annotations[1] = []

        for label, obj_id in [(first_tool_label, 0), (second_tool_label, 1)]:

            if label == -1:
                continue
            
            mask = labels == label
            mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            if annotations[obj_id]["points"] is None:
                annotations[obj_id]["points"] = []

            if annotations[obj_id]["labels"] is None:
                annotations[obj_id]["labels"] = []

            annotations[obj_id]["points"].append([cx, cy])
            annotations[obj_id]["labels"].append(1)

            label_indices = np.where(labels == label)
            random_indices = np.random.choice(len(label_indices[0]), NUM_POS_POINTS_PER_TOOL - 1, replace=False)
            for i in random_indices:
                x = int(label_indices[1][i])
                y = int(label_indices[0][i])
                if ground_truth_mask[y, x]:
                    annotations[obj_id]["points"].append([x, y])
                    annotations[obj_id]["labels"].append(1)

            label_indices = np.where(labels == background_label)
            # print("HIII ", label_indices)
            random_indices = np.random.choice(len(label_indices[0]), NUM_NEG_POINTS_PER_TOOL, replace=False)
            for i in random_indices:
                x = int(label_indices[1][i])
                y = int(label_indices[0][i])
                if not ground_truth_mask[y, x]:
                    annotations[obj_id]["points"].append([x, y])
                    annotations[obj_id]["labels"].append(0)

    return annotations

def annotate_frames(sub_dirs, domains, split, num_frames=300):
    print("Annotating split: ", split)

    all_annotations = {}

    for sub_dir in tqdm(sub_dirs):
        for domain in tqdm(domains):
            for frame in tqdm(range(num_frames // 2)):
                left_video_frames_path = sub_dir + "/" + domain + "/left"
                right_video_frames_path = sub_dir + "/" + domain + "/right"

                first_left_frame = left_video_frames_path + "/" + str(frame) + ".png"
                first_right_frame = right_video_frames_path + "/" + str(frame) + ".png"

                last_left_frame = left_video_frames_path + "/" + str(TOTAL_FRAMES_PER_VIDEO - frame - 1) + ".png"
                last_right_frame = right_video_frames_path + "/" + str(TOTAL_FRAMES_PER_VIDEO - frame - 1) + ".png"

                if SHOULD_USE_MANUAL_PROMPT:
                    left_annotations = manual_annotate(first_left_frame)
                    right_annotations = manual_annotate(first_right_frame)

                    left_reverse_annotations = manual_annotate(last_left_frame)
                    right_reverse_annotations = manual_annotate(last_right_frame)
                else:
                    if SHOULD_USE_BOX_PROMPT:
                        if SHOULD_SAMPLE_GROUND_TRUTH:
                            left_annotations = auto_box_ground_truth_annotate(first_left_frame)
                            right_annotations = auto_box_ground_truth_annotate(first_right_frame)

                            left_reverse_annotations = auto_box_ground_truth_annotate(last_left_frame)
                            right_reverse_annotations = auto_box_ground_truth_annotate(last_right_frame)
                        else:
                            left_annotations = auto_box_model_annotate(first_left_frame)
                            right_annotations = auto_box_model_annotate(first_right_frame)

                            left_reverse_annotations = auto_box_model_annotate(last_left_frame)
                            right_reverse_annotations = auto_box_model_annotate(last_right_frame)
                    else:
                        left_annotations = auto_point_annotate(first_left_frame)
                        right_annotations = auto_point_annotate(first_right_frame)

                        left_reverse_annotations = auto_point_annotate(last_left_frame)
                        right_reverse_annotations = auto_point_annotate(last_right_frame)

                if sub_dir + "/" + domain + "/left" not in all_annotations:
                    all_annotations[sub_dir + "/" + domain + "/left"] = {}

                if sub_dir + "/" + domain + "/right" not in all_annotations:
                    all_annotations[sub_dir + "/" + domain + "/right"] = {}

                all_annotations[sub_dir + "/" + domain + "/left"][frame] = left_annotations
                all_annotations[sub_dir + "/" + domain + "/right"][frame] = right_annotations
                all_annotations[sub_dir + "/" + domain + "/left"][TOTAL_FRAMES_PER_VIDEO - frame - 1] = left_reverse_annotations
                all_annotations[sub_dir + "/" + domain + "/right"][TOTAL_FRAMES_PER_VIDEO - frame - 1] = right_reverse_annotations

                annotation_file = PROMPTS_FOLDER + f"/{split}.json"

                os.makedirs(PROMPTS_FOLDER, exist_ok=True)
                with open(annotation_file, "w") as f:
                    json.dump(all_annotations, f)

            print(f"Domain annotated: {domain}")
        print(f"Subdir annotated: {sub_dir}")

# %%
if REFRESH_PROMPTS:
    if PROMPTING_STRATEGY == "first":
        if "val" in SPLITS_TO_RUN:
            annotate_frames(val_video_folders_path, VAL_DOMAINS, "val", 2)
        if "test" in SPLITS_TO_RUN:
            annotate_frames(test_video_folders_path, TEST_DOMAINS, "test", 2)
        if "train" in SPLITS_TO_RUN:
            annotate_frames(train_video_folders_path, TRAIN_DOMAINS, "train", 2)
    elif PROMPTING_STRATEGY == "all":
        if "val" in SPLITS_TO_RUN:
            annotate_frames(val_video_folders_path, VAL_DOMAINS, "val", TOTAL_FRAMES_PER_VIDEO)
        if "test" in SPLITS_TO_RUN:
            annotate_frames(test_video_folders_path, TEST_DOMAINS, "test", TOTAL_FRAMES_PER_VIDEO)
        if "train" in SPLITS_TO_RUN:
            annotate_frames(train_video_folders_path, TRAIN_DOMAINS, "train", TOTAL_FRAMES_PER_VIDEO)
    elif PROMPTING_STRATEGY == "dynamic":
        pass
    else:
        raise ValueError("Invalid prompting strategy")
    
    for split in SPLITS_TO_RUN:
        if SHOULD_VISUALIZE_PROMPTS:
            all_annotations = {}
            annotation_file = PROMPTS_FOLDER + f"/{split}.json"
            with open(annotation_file, "r") as f:
                all_annotations = json.load(f)

            annotation_visualization_path = PROMPTS_FOLDER + f"/{split}_visualization"

            os.makedirs(annotation_visualization_path, exist_ok=True)

            for path in all_annotations:
                for frame in all_annotations[path]:
                    annotations = all_annotations[path][frame]
                    ground_truth_path = path + "/" + str(frame) + ".png"

                    for domain in VAL_DOMAINS:
                        if domain in ground_truth_path:
                            ground_truth_path = ground_truth_path.replace(domain, "ground_truth")
                            break
                    
                    # print("GROUNDDDD TRUTH PATH: ", ground_truth_path)
                    ground_truth_image = cv2.imread(ground_truth_path)
                    # print("ground_truth_image.shape: ", ground_truth_image.shape)
                    for tool in annotations:
                        if "bbox" in annotations[tool]:
                            box = annotations[tool]["bbox"]["box"]
                            cv2.rectangle(ground_truth_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            conf = annotations[tool]["bbox"]["conf"]
                            cv2.putText(ground_truth_image, f"obj-{tool} conf: {conf:.2f}", (box[0], box[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        elif "points" in annotations[tool]:
                            points = annotations[tool]["points"]
                            labels = annotations[tool]["labels"]
                            for point, label in zip(points, labels):
                                if tool == 0:
                                    if label == 1:
                                        cv2.circle(ground_truth_image, (point[0], point[1]), 10, (0, 255, 0), -1)
                                    else:
                                        cv2.circle(ground_truth_image, (point[0], point[1]), 10, (0, 0, 255), -1)
                                else:
                                    if label == 1:
                                        cv2.circle(ground_truth_image, (point[0], point[1]), 10, (255, 0, 0), -1)
                                    else:
                                        cv2.circle(ground_truth_image, (point[0], point[1]), 10, (255, 255, 0), -1)
                        else:
                            print("Sadge, no annotations found for this tool: ", tool, "ground truth path: ", ground_truth_path)

                    infix_path = path.split(f"SegSTRONGC_{split}/")[-1]
                    final_path = annotation_visualization_path + "/" + infix_path + "/" + str(frame) + ".jpg"

                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    cv2.imwrite(annotation_visualization_path + "/" + infix_path + "/" + str(frame) + ".jpg", ground_truth_image)
else:
    print("Annotations already exist. Skipping annotation process.")

# %%
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# %%
def run_inference(inference_model, frames_path, split, is_reverse, forward_pass_path=None):
    mask_storage_data = {}
    predicted_masks = []
    
    if inference_model == "sam2.1_hiera_base_plus":
        print(f"Loading annotations for split: {split}")
        try:
            with open(PROMPTS_FOLDER + f"/{split}.json", "r") as f:
                annotations = json.load(f)
            print(f"Successfully loaded annotations for {len(annotations)} items")
        except Exception as e:
            print(f"Error loading annotations: {e}")
            LOGGER.error(f"Error loading annotations for split {split}: {e}")
            return
 
        if annotations is None:
            print("No annotations found for split", split)
            LOGGER.warning(f"No annotations found for split {split}")
            return None, None
        
        start = time.time()
        print(f"Initializing SAM for video...")
        sam2_predictor = build_sam2_video_predictor(inference_model_config, inference_model_checkpoint, device=device)

        inference_state = sam2_predictor.init_state(
            video_path = frames_path,
        )
        end = time.time()
        print(f"Initialization took {end - start:.2f} seconds.")
        LOGGER.info(f"SAM initialization for {forward_pass_path} took {end - start:.2f} seconds.")

        current_annotations = annotations[forward_pass_path]
        print(f"Found {len(current_annotations)} objects with annotations.")
        LOGGER.info(f"Processing {len(current_annotations)} objects with annotations for {forward_pass_path}")
        
        n_points = 0
        n_boxes = 0

        frame_range = range(TOTAL_FRAMES_PER_VIDEO)
        for frame in tqdm(frame_range):
            if PROMPTING_STRATEGY == "first" and frame > 0:
                break

            print(f"Processing annotations for frame {frame}...")
            print(len(current_annotations), " annotations found.")
            # print keys in frame_annotations
            print("KEYSSSSS: ", current_annotations.keys())

            if is_reverse:
                frame_annotations = current_annotations[str(TOTAL_FRAMES_PER_VIDEO - frame - 1)]
            else:
                frame_annotations = current_annotations[str(frame)]

            for tool in tqdm(frame_annotations, desc=f"Processing annotations for objects"):
                if "bbox" in frame_annotations[tool]:
                    n_boxes += 1
                    box = np.array(frame_annotations[tool]["bbox"]["box"], dtype=np.float32)
                    _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                        inference_state = inference_state,
                        frame_idx = frame,
                        obj_id = int(tool),
                        box = box
                    )
                elif "points" in frame_annotations[tool]:
                    n_points += len(frame_annotations[tool]["points"])
                    points = np.array(frame_annotations[tool]["points"], dtype=np.float32)
                    labels = np.array(frame_annotations[tool]["labels"], dtype=np.int32)

                    _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                        inference_state = inference_state,
                        frame_idx = frame,
                        obj_id = int(tool),
                        points = points,
                        labels = labels
                    )
                else:
                    print("No annotations found for object", tool, " in frame ", frame)
                    LOGGER.warning(f"No annotations found for object {tool} in frame {frame}")

        print(f"Added {n_points} annotation points across all objects.")
        LOGGER.info(f"Added {n_points} annotation points across all objects for {forward_pass_path}")

        print("Starting mask propagation...")
        start = time.time()
        video_segments = {}

        n_frames = 0
        infix_path = frames_path.split(f"SegSTRONGC_{split}/")[-1]

        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(
            inference_state, 
            save_mask_logits = SAVE_RUN_MASK_LOGITS, 
            infix_path = infix_path
        ):
            n_frames += 1
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        end = time.time()
        prop_time = end - start

        print(f"Mask propagation completed in {prop_time:.2f} seconds.")
        LOGGER.info(f"Mask propagation for {forward_pass_path} took {prop_time:.2f} seconds.")

        print("Processing predicted masks...")
        for frame_idx, obj_dict in tqdm(video_segments.items(), desc="Processing video frames"):
            # it should have shape (1080, 1920)
            # mask_storage_data[frame_idx] = []
            overall_mask = np.zeros((1080, 1920), dtype=bool)

            for obj_id, mask_array in obj_dict.items():
                # mask_storage_data[frame_idx].append({
                #     obj_id: mask_array
                # })
                
                overall_mask = np.logical_or(overall_mask, mask_array.squeeze())

            predicted_masks.append(overall_mask)
            mask_storage_data[frame_idx] = overall_mask

        sam2_predictor.reset_state(inference_state)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Empied CUDA cache.")
        print("SAM state reset.")
    elif inference_model == "yolo11x-seg":
        start = time.time()
        print(f"Initializing Yolo for video...")
        yolo_model = YOLO(inference_model_checkpoint)
        end = time.time()
        print(f"Initialization took {end - start:.2f} seconds.")
        LOGGER.info(f"Yolo initialization for {frames_path} took {end - start:.2f} seconds.")

        print("Starting mask propagation...")
        start = time.time()

        total_images = len(os.listdir(frames_path))
        print(f"Total images in video: {total_images}")
        LOGGER.info(f"Total images in video: {total_images}")

        video_result = []

        for i in tqdm(range(total_images), desc="Propagating video frames"):
            frame_path = frames_path + f"/{i}.png"
            # video_result.append(yolo_model(frame_path, max_det = 2))
            video_result.append(yolo_model(frame_path))

        end = time.time()
        prop_time = end - start

        print(f"Mask propagation completed in {prop_time:.2f} seconds.")
        LOGGER.info(f"Mask propagation for {frames_path} took {prop_time:.2f} seconds.")

        print("Processing predicted masks...")

        for frame_idx, frame_result in tqdm(enumerate(video_result), desc="Processing video frames"):
            # it should have shape (1080, 1920)
            # mask_storage_data[frame_idx] = []
            overall_mask = np.zeros((1080, 1920), dtype=bool)

            for result in frame_result:
                if result.masks is None:
                    # mask_storage_data[frame_idx].append({
                    #     0: overall_mask
                    # })
                    continue

                for mask_id, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy()
                    reshaped_mask = cv2.resize(mask_np, (1920, 1080), interpolation=cv2.INTER_NEAREST)

                    # mask_storage_data[frame_idx].append({
                    #     mask_id: reshaped_mask
                    # })

                    overall_mask = np.logical_or(overall_mask, reshaped_mask)

            predicted_masks.append(overall_mask)
            mask_storage_data[frame_idx] = overall_mask
    else:
        raise ValueError("Invalid model name.")
    
    return mask_storage_data, predicted_masks

# %%
import pandas as pd
import os
import datetime

def process_video(frames_path, sub_dir, domain, split, is_left):
    print("="*50)
    print(f"Processing video: {frames_path}")
    print(f"Domain: {domain}, Split: {split}, Camera: {'left' if is_left else 'right'}")
    LOGGER.info(f"Processing video: {frames_path} (Domain: {domain}, Split: {split}, Camera: {'left' if is_left else 'right'})")
    stereo_dir = "left" if is_left else "right"
    ground_truth_masks_path = sub_dir + "/ground_truth/" + stereo_dir

    overall_start = time.time()
    mask_storage_data, predicted_masks = run_inference(INFERENCE_MODEL, frames_path, split, False, frames_path)

    if SHOULD_PERFROM_CYCLIC_TTA:
        temp_video_frames_path = "data/temp"
        if not os.path.exists(temp_video_frames_path):
            os.makedirs(temp_video_frames_path)

        total_files = len(os.listdir(frames_path))
        for filename in os.listdir(frames_path):
            if filename.endswith(".png"):
                original_index = int(filename.split('.')[0])
                new_index = total_files - 1 - original_index
                new_filename = f"{new_index}.png"
                shutil.copy(os.path.join(frames_path, filename), os.path.join(temp_video_frames_path, new_filename))

        reverse_mask_storage_data, predicted_reverse_masks = run_inference(INFERENCE_MODEL, temp_video_frames_path, split, True, frames_path)
        predicted_reverse_masks = predicted_reverse_masks[::-1]

        if SAVE_IMAGES_ONCE:
            save_dir = f"data/results/{INFERENCE_MODEL}/visualizations"
            os.makedirs(save_dir, exist_ok=True)
            
            for i in range(len(predicted_masks)):
                # Get frame path and load original image
                frame_path = os.path.join(frames_path, f"{i}.png")
                original_img = cv2.imread(frame_path)
                
                # Load ground truth mask
                gt_mask = cv2.imread(os.path.join(ground_truth_masks_path, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask > 0).astype(np.uint8) * 255
                
                # Convert predicted masks to uint8
                forward_mask = predicted_masks[i].astype(np.uint8) * 255
                reverse_mask = predicted_reverse_masks[i].astype(np.uint8) * 255
                
                # Create visualization grid
                h, w = original_img.shape[:2]
                grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
                
                # Place images in grid
                grid[:h, :w] = original_img  # Original
                grid[:h, w:] = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)  # Ground truth
                grid[h:, :w] = cv2.cvtColor(forward_mask, cv2.COLOR_GRAY2BGR)  # Forward mask
                grid[h:, w:] = cv2.cvtColor(reverse_mask, cv2.COLOR_GRAY2BGR)  # Reverse mask
                
                # Add labels
                labels = ['Original', 'Ground Truth', 'Forward Mask', 'Reverse Mask']
                positions = [(10, 30), (w+10, 30), (10, h+30), (w+10, h+30)]
                
                for label, pos in zip(labels, positions):
                    cv2.putText(grid, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                # Save the grid
                save_path = os.path.join(save_dir, f"frame_{i:04d}.png")
                cv2.imwrite(save_path, grid)
                
                print(f"Saved visualization images to {save_dir}")

        SAVE_IMAGES_ONCE = False

        for i in range(len(predicted_masks)):
            predicted_masks[i] = np.logical_or(predicted_masks[i], predicted_reverse_masks[i])

        shutil.rmtree(temp_video_frames_path)
        print(f"Applied TTA to {len(predicted_masks)} masks.")
    
    print(f"Generated {len(predicted_masks)} masks for evaluation.")
    LOGGER.info(f"Generated {len(predicted_masks)} masks for {frames_path}")

    # Save the masks
    masks_split_dir = MASKS_DIR + f"/{INFERENCE_MODEL}" + f"/{split}"
    if not os.path.exists(masks_split_dir):
        os.makedirs(masks_split_dir)

    # masks_file = masks_split_dir + f"/{frames_path.replace('/', '-')}.pkl"

    # data = {}
    # data[frames_path] = mask_storage_data

    # with open(masks_file, "wb") as f:
    #     pickle.dump(data, f)
    # print(f"Masks saved to {masks_file}")

    print("Loading ground truth masks for evaluation...")
    ground_truth_masks = []
    for i in range(len(predicted_masks)):
        ground_truth_mask = cv2.imread(ground_truth_masks_path + "/" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
        ground_truth_mask = (ground_truth_mask > 0).astype(np.bool_)
        ground_truth_masks.append(ground_truth_mask)
    print(f"Loaded {len(ground_truth_masks)} ground truth masks.")

    print("Calculating evaluation metrics...")
    start = time.time()
    miou = calculate_miou(predicted_masks, ground_truth_masks)
    mdsc = calculate_mdsc(predicted_masks, ground_truth_masks)
    end = time.time()
    eval_time = end - start
    print(f"Time taken for metrics calculation: {eval_time:.2f} seconds.")
    LOGGER.info(f"Time taken for metrics calculation: {eval_time:.2f} seconds.")

    print(f"Mean IoU for {sub_dir}/{domain}/{stereo_dir}: {miou:.4f}")
    print(f"Mean DSC for {sub_dir}/{domain}/{stereo_dir}: {mdsc:.4f}")

    LOGGER.info(f"Mean IoU for {sub_dir}/{domain}/{stereo_dir}: {miou:.4f}")
    LOGGER.info(f"Mean DSC for {sub_dir}/{domain}/{stereo_dir}: {mdsc:.4f}")

    results_file = BASE_RESULTS_DIR + f"/{INFERENCE_MODEL}" + f"/{split}.json"
    if os.path.exists(results_file):
        print(f"Loading existing results file: {results_file}")
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        print(f"Creating new results file: {results_file}")
        all_results = {}
    
    all_results[frames_path] = {
        "miou": miou,
        "mdsc": mdsc
    }

    with open(results_file, "w") as f:
        json.dump(all_results, f)
    print(f"Results saved to {results_file}")

    overall_end = time.time()
    total_time = overall_end - overall_start
    print(f"Processing video took {total_time:.2f} seconds.")
    LOGGER.info(f"Results for {sub_dir}/{domain}/{stereo_dir} saved.")
    LOGGER.info(f"Processing video took {total_time:.2f} seconds.")
    print("="*50)

    return miou, mdsc

def process_split(sub_dirs, domains, split):
    print("="*80)
    print(f"Running inference for split: {split}")
    LOGGER.info(f"Using Model: {INFERENCE_MODEL}")
    LOGGER.info(f"Annotation mode: {'manual' if SHOULD_USE_MANUAL_PROMPT else 'auto'}")
    LOGGER.info(f"Performing tta: {'yes' if SHOULD_PERFROM_CYCLIC_TTA else 'no'}")
    LOGGER.info(f"Using box prompt: {'yes' if SHOULD_USE_BOX_PROMPT else 'no'}")
    LOGGER.info(f"Using ground truth for box prompt: {'yes' if SHOULD_SAMPLE_GROUND_TRUTH else 'no'}")
    LOGGER.info(f"Refreshing prompts: {'yes' if REFRESH_PROMPTS else 'no'}")
    LOGGER.info(f"Prompting strategy: {PROMPTING_STRATEGY}")
    LOGGER.info(f"Saving run mask logits: {'yes' if SAVE_RUN_MASK_LOGITS else 'no'}")
    LOGGER.info(f"Saving images once: {'yes' if SAVE_IMAGES_ONCE else 'no'}")
    LOGGER.info(f"Visualizing prompts: {'yes' if SHOULD_VISUALIZE_PROMPTS else 'no'}")
    
    if not SHOULD_USE_BOX_PROMPT:
        LOGGER.info(f"Number of positive point annotations per tool: {NUM_POS_POINTS_PER_TOOL}")
        LOGGER.info(f"Number of negative point annotations per tool: {NUM_NEG_POINTS_PER_TOOL}")

    print("="*80)
    LOGGER.info(f"----------------Running inference for split {split}-------------")
    overall_start = time.time()
    
    print(f"Processing {len(sub_dirs)} sub-directories and {len(domains)} domains")
    LOGGER.info(f"Processing {len(sub_dirs)} sub-directories and {len(domains)} domains for split {split}")
        
    sub_dir_results = {}
    for sub_dir in tqdm(sub_dirs, desc=f"Processing sub-directories"):
        print("\n" + "-"*60)
        print(f"Processing sub-directory: {sub_dir}")
        LOGGER.info(f"Processing sub-directory: {sub_dir}")
        domain_results = {}
        for domain in tqdm(domains, desc=f"Processing domains"):
            print(f"\nProcessing domain: {domain}")
            LOGGER.info(f"Processing domain: {domain} in {sub_dir}")
            left_video_frames_path = sub_dir + "/" + domain + "/left"
            right_video_frames_path = sub_dir + "/" + domain + "/right"

            print(f"Processing left camera video...")
            left_miou, left_msdc = process_video(left_video_frames_path, sub_dir, domain, split, True)
            
            print(f"Processing right camera video...")
            right_miou, right_msdc = process_video(right_video_frames_path, sub_dir, domain, split, False)

            overall_miou = (left_miou + right_miou) / 2
            overall_msdc = (left_msdc + right_msdc) / 2

            print(f"\nResults for {sub_dir}/{domain}:")
            print(f"  Left: IoU={left_miou:.4f}, DSC={left_msdc:.4f}")
            print(f"  Right: IoU={right_miou:.4f}, DSC={right_msdc:.4f}")
            print(f"  Overall: IoU={overall_miou:.4f}, DSC={overall_msdc:.4f}")
            
            LOGGER.info(f"Results for {sub_dir}/{domain}: Left IoU={left_miou:.4f}, Right IoU={right_miou:.4f}, Overall IoU={overall_miou:.4f}")

            domain_results[domain] = {
                "left_miou": left_miou,
                "left_msdc": left_msdc,
                "right_miou": right_miou,
                "right_msdc": right_msdc,
                "overall_miou": overall_miou,
                "overall_msdc": overall_msdc
            }

        sub_dir_results[sub_dir] = domain_results

    content = ""

    print("\n" + "="*60)
    print(f"SUMMARY RESULTS FOR SPLIT: {split}")
    print("="*60)
    LOGGER.info(f"SUMMARY RESULTS FOR SPLIT: {split}")
    LOGGER.info(f'Description: {RUN_DESCRIPTION}')
    LOGGER.info(f"Using Model: {INFERENCE_MODEL}")
    LOGGER.info(f"Annotation mode: {'manual' if SHOULD_USE_MANUAL_PROMPT else 'auto'}")
    LOGGER.info(f"Performing tta: {'yes' if SHOULD_PERFROM_CYCLIC_TTA else 'no'}")
    LOGGER.info(f"Using box prompt: {'yes' if SHOULD_USE_BOX_PROMPT else 'no'}")
    LOGGER.info(f"Using ground truth for box prompt: {'yes' if SHOULD_SAMPLE_GROUND_TRUTH else 'no'}")
    LOGGER.info(f"Refreshing prompts: {'yes' if REFRESH_PROMPTS else 'no'}")
    LOGGER.info(f"Prompting strategy: {PROMPTING_STRATEGY}")
    LOGGER.info(f"Saving run mask logits: {'yes' if SAVE_RUN_MASK_LOGITS else 'no'}")
    LOGGER.info(f"Saving images once: {'yes' if SAVE_IMAGES_ONCE else 'no'}")
    LOGGER.info(f"Visualizing prompts: {'yes' if SHOULD_VISUALIZE_PROMPTS else 'no'}")
    
    if not SHOULD_USE_BOX_PROMPT:
        LOGGER.info(f"Number of positive point annotations per tool: {NUM_POS_POINTS_PER_TOOL}")
        LOGGER.info(f"Number of negative point annotations per tool: {NUM_NEG_POINTS_PER_TOOL}")


    content += f"SUMMARY RESULTS FOR SPLIT: {split}\n"
    content += f"Description: {RUN_DESCRIPTION}\n"
    content += f"Using Model: {INFERENCE_MODEL}\n"
    content += f"Annotation mode: {'manual' if SHOULD_USE_MANUAL_PROMPT else 'auto'}\n"
    content += f"Performing tta: {'yes' if SHOULD_PERFROM_CYCLIC_TTA else 'no'}\n"
    content += f"Using box prompt: {'yes' if SHOULD_USE_BOX_PROMPT else 'no'}\n"
    content += f"Using ground truth for box prompt: {'yes' if SHOULD_SAMPLE_GROUND_TRUTH else 'no'}\n"
    content += f"Refreshing prompts: {'yes' if REFRESH_PROMPTS else 'no'}\n"
    content += f"Prompting strategy: {PROMPTING_STRATEGY}\n"
    content += f"Saving run mask logits: {'yes' if SAVE_RUN_MASK_LOGITS else 'no'}\n"
    content += f"Saving images once: {'yes' if SAVE_IMAGES_ONCE else 'no'}\n"
    content += f"Visualizing prompts: {'yes' if SHOULD_VISUALIZE_PROMPTS else 'no'}\n"

    if not SHOULD_USE_BOX_PROMPT:
        content += f"Number of positive point annotations per tool: {NUM_POS_POINTS_PER_TOOL}\n"
        content += f"Number of negative point annotations per tool: {NUM_NEG_POINTS_PER_TOOL}\n"

    # Domain-wise results
    print("\nDomain-wise Results:")
    LOGGER.info("Domain-wise Results:")

    content += "\nDomain-wise Results:\n"
    domain_results_data = {}
    for domain in domains:
        left_mious = [sub_dir_results[sub_dir][domain]["left_miou"] for sub_dir in sub_dirs]
        right_mious = [sub_dir_results[sub_dir][domain]["right_miou"] for sub_dir in sub_dirs]
        overall_mious = [sub_dir_results[sub_dir][domain]["overall_miou"] for sub_dir in sub_dirs]

        left_msdcs = [sub_dir_results[sub_dir][domain]["left_msdc"] for sub_dir in sub_dirs]
        right_msdcs = [sub_dir_results[sub_dir][domain]["right_msdc"] for sub_dir in sub_dirs]
        overall_msdcs = [sub_dir_results[sub_dir][domain]["overall_msdc"] for sub_dir in sub_dirs]

        print(f"\nDomain: {domain}")
        print(f"  Left Frame IoU: {np.mean(left_mious):.4f}")
        print(f"  Right Frame IoU: {np.mean(right_mious):.4f}")
        print(f"  Overall IoU: {np.mean(overall_mious):.4f}")
        print(f"  Left Frame DSC: {np.mean(left_msdcs):.4f}")
        print(f"  Right Frame DSC: {np.mean(right_msdcs):.4f}")
        print(f"  Overall DSC: {np.mean(overall_msdcs):.4f}")
        
        LOGGER.info(f"Domain {domain} - Left IoU: {np.mean(left_mious):.4f}, Right IoU: {np.mean(right_mious):.4f}, Overall IoU: {np.mean(overall_mious):.4f}")
        LOGGER.info(f"Domain {domain} - Left DSC: {np.mean(left_msdcs):.4f}, Right DSC: {np.mean(right_msdcs):.4f}, Overall DSC: {np.mean(overall_msdcs):.4f}")

        content += f"\nDomain: {domain}\n"
        content += f"  Left Frame IoU: {np.mean(left_mious):.4f}\n"
        content += f"  Right Frame IoU: {np.mean(right_mious):.4f}\n"
        content += f"  Overall IoU: {np.mean(overall_mious):.4f}\n"
        content += f"  Left Frame DSC: {np.mean(left_msdcs):.4f}\n"
        content += f"  Right Frame DSC: {np.mean(right_msdcs):.4f}\n"
        content += f"  Overall DSC: {np.mean(overall_msdcs):.4f}\n"

        
        domain_results_data[domain] = {
            "left_miou": np.mean(left_mious),
            "right_miou": np.mean(right_mious),
            "overall_miou": np.mean(overall_mious),
            "left_mdsc": np.mean(left_msdcs),
            "right_mdsc": np.mean(right_msdcs),
            "overall_mdsc": np.mean(overall_msdcs)
        }

    # Overall results across all domains and sub-dirs
    left_mious = [np.mean([sub_dir_results[sub_dir][domain]["left_miou"] for domain in domains]) for sub_dir in sub_dirs]
    right_mious = [np.mean([sub_dir_results[sub_dir][domain]["right_miou"] for domain in domains]) for sub_dir in sub_dirs]
    overall_mious = [np.mean([sub_dir_results[sub_dir][domain]["overall_miou"] for domain in domains]) for sub_dir in sub_dirs]

    left_msdcs = [np.mean([sub_dir_results[sub_dir][domain]["left_msdc"] for domain in domains]) for sub_dir in sub_dirs]
    right_msdcs = [np.mean([sub_dir_results[sub_dir][domain]["right_msdc"] for domain in domains]) for sub_dir in sub_dirs]
    overall_msdcs = [np.mean([sub_dir_results[sub_dir][domain]["overall_msdc"] for domain in domains]) for sub_dir in sub_dirs]

    content += "-------------------------------------------------------------------------"

    print("\n" + "-"*60)
    print("FINAL RESULTS ACROSS ALL DOMAINS AND SUB-DIRECTORIES:")
    print(f"  Left Frame IoU: {np.mean(left_mious):.4f}")
    print(f"  Right Frame IoU: {np.mean(right_mious):.4f}")
    print(f"  Overall IoU: {np.mean(overall_mious):.4f}")
    print(f"  Left Frame DSC: {np.mean(left_msdcs):.4f}")
    print(f"  Right Frame DSC: {np.mean(right_msdcs):.4f}")
    print(f"  Overall DSC: {np.mean(overall_msdcs):.4f}")

    LOGGER.info("FINAL RESULTS ACROSS ALL DOMAINS AND SUB-DIRECTORIES:")
    LOGGER.info(f"Left Frame IoU: {np.mean(left_mious):.4f}")
    LOGGER.info(f"Right Frame IoU: {np.mean(right_mious):.4f}")
    LOGGER.info(f"Overall IoU: {np.mean(overall_mious):.4f}")
    LOGGER.info(f"Left Frame DSC: {np.mean(left_msdcs):.4f}")
    LOGGER.info(f"Right Frame DSC: {np.mean(right_msdcs):.4f}")
    LOGGER.info(f"Overall DSC: {np.mean(overall_msdcs):.4f}")

    content += "FINAL RESULTS ACROSS ALL DOMAINS AND SUB-DIRECTORIES:\n"
    content += f"Left Frame IoU: {np.mean(left_mious):.4f}\n"
    content += f"Right Frame IoU: {np.mean(right_mious):.4f}\n"
    content += f"Overall IoU: {np.mean(overall_mious):.4f}\n"
    content += f"Left Frame DSC: {np.mean(left_msdcs):.4f}\n"
    content += f"Right Frame DSC: {np.mean(right_msdcs):.4f}\n"
    content += f"Overall DSC: {np.mean(overall_msdcs):.4f}\n"

    overall_end = time.time()
    total_time = overall_end - overall_start
    print(f"\nTotal time taken for split {split}: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    LOGGER.info(f"Total time taken for split {split}: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*80)

    content += f"\nTotal time taken for split {split}: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n"
    
    # Save results to CSV
    
    csv_file = f'data/results/results.csv'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        'timestamp': timestamp,
        'desc': RUN_DESCRIPTION,
        'model': INFERENCE_MODEL,
        'split': split,
        'annotation_mode': 'manual' if SHOULD_USE_MANUAL_PROMPT else 'auto',
        'num_pos_points': NUM_POS_POINTS_PER_TOOL if not SHOULD_USE_BOX_PROMPT else 'N/A',
        'num_neg_points': NUM_NEG_POINTS_PER_TOOL if not SHOULD_USE_BOX_PROMPT else 'N/A',
        'tta': 'yes' if SHOULD_PERFROM_CYCLIC_TTA else 'no',
        'overall_left_miou': np.mean(left_mious),
        'overall_right_miou': np.mean(right_mious),
        'overall_miou': np.mean(overall_mious),
        'overall_left_mdsc': np.mean(left_msdcs),
        'overall_right_mdsc': np.mean(right_msdcs),
        'overall_mdsc': np.mean(overall_msdcs),
        'total_time_seconds': total_time,
        'total_time_minutes': total_time/60
    }
    
    # Add domain-specific results
    for domain in domains:
        data[f'{domain}_left_miou'] = domain_results_data[domain]['left_miou']
        data[f'{domain}_right_miou'] = domain_results_data[domain]['right_miou']
        data[f'{domain}_overall_miou'] = domain_results_data[domain]['overall_miou']
        data[f'{domain}_left_mdsc'] = domain_results_data[domain]['left_mdsc']
        data[f'{domain}_right_mdsc'] = domain_results_data[domain]['right_mdsc']
        data[f'{domain}_overall_mdsc'] = domain_results_data[domain]['overall_mdsc']
    
    # Convert to DataFrame for a single row
    df_new = pd.DataFrame([data])
    
    # Check if file exists and append, or create new
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        df_new.to_csv(csv_file, index=False)
    
    print(f"Results saved to CSV file: {csv_file}")
    LOGGER.info(f"Results saved to CSV file: {csv_file}")

    requests.post(DISCORD_WEBHOOK_URL, { "content": content, "username" : f"{split}-runner" })

for split in SPLITS_TO_RUN:
    if "val" == split:
        print("Running inference for validation split")
        requests.post(DISCORD_WEBHOOK_URL, { "content": "Running inference for validation split", "username" : "val-runner"  })
        process_split(val_video_folders_path, VAL_DOMAINS, "val")
    elif "test" == split:
        print("Running inference for test split")
        requests.post(DISCORD_WEBHOOK_URL, { "content": "Running inference for test split", "username" : "test-runner" })
        process_split(test_video_folders_path, TEST_DOMAINS, "test")
    elif "train" == split:
        print("Running inference for train split")
        requests.post(DISCORD_WEBHOOK_URL, { "content": "Running inference for train split", "username" : "train-runner"  })
        process_split(train_video_folders_path, TRAIN_DOMAINS, "train")

# %%
# domains = ['bg_change', 'blood', 'low_brightness', 'regular', 'smoke']
# annotations = None
# with open('data/annotations/auto/val.json', 'r') as f:
#     annotations = json.load(f)

# path = "data/masks/sam2.1_hiera_base_plus/val/data-raw-SegSTRONGC_val-val-1-2-bg_change-right.pkl"
# with open(path, 'rb') as f:
#     mass = pickle.load(f)
#     for video_path, video_data in mass.items():
#         # print(video_path, video_data)
#         for frame_id, frame_data in video_data.items():
#             overall_mask = np.zeros((1080, 1920), dtype=bool)
#             for data in frame_data:
#                 for object_id, mask in data.items():
#                     overall_mask = np.logical_or(overall_mask, mask[0])

#             ground_truth_masks_path = video_path
#             for domain in domains:
#                 if domain in video_path:
#                     ground_truth_masks_path = video_path.replace(domain, 'ground_truth')
#                     break
#             ground_truth_masks_path = ground_truth_masks_path + "/" + str(frame_id) + ".jpg"
#             ground_truth_mask = cv2.imread(ground_truth_masks_path, cv2.IMREAD_GRAYSCALE)
#             ground_truth_mask = (ground_truth_mask > 0).astype(np.bool_)

#             original_image = cv2.imread(video_path + "/" + str(frame_id) + ".jpg")
#             original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#             # frame_annotations = annotations[video_path.replace("/left", "").replace("/right", "")]
#             frame_annotations = annotations[video_path]

#             # place dots in the original image for each annotation
#             for object_id, object_annotations in frame_annotations.items():
#                 # print(object_annotations)
#                 for annotation in object_annotations:
#                     x = annotation['x']
#                     y = annotation['y']
#                     label = annotation['label']
#                     if object_id == "0":
#                         original_image = cv2.circle(original_image, (x, y), 10, (0, 255, 0), -1)
#                     else:
#                         original_image = cv2.circle(original_image, (x, y), 10, (255, 0, 0), -1)

#             #show the masks and the original image
#             fig, axs = plt.subplots(1, 3, figsize=(30, 15))
#             axs[0].imshow(original_image)
#             axs[0].set_title("Original Image")
#             axs[1].imshow(overall_mask)
#             axs[1].set_title("Overall Mask")
#             axs[2].imshow(ground_truth_mask)
#             axs[2].set_title("Ground Truth Mask")
#             plt.show()
#             break


