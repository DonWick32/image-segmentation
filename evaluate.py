import os
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
import json
import cv2
import gc
from tqdm import tqdm
from SAM.sam2.sam2.build_sam import build_sam2_video_predictor

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


def run_inference(sam2_predictor, frames_path, annotation_path):
	predicted_masks = []
	
	with open(annotation_path, "r") as f:
		annotations = json.load(f)

	inference_state = sam2_predictor.init_state(video_path=frames_path)
	frames_path = 'data/raw/SegSTRONGC'+frames_path.split('SegSTRONGC')[-1]
	current_annotations = annotations[frames_path]

	n_points = 0
	for object in tqdm(current_annotations, desc=f"Processing annotations for objects"):
		n_points += len(current_annotations[object])
		for annotation in current_annotations[object]:
			points = np.array([[int(annotation["x"]), int(annotation["y"])]], np.float32)
			labels = np.array([annotation["label"]], np.int32)
			_, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
				inference_state=inference_state,
				frame_idx=0,
				obj_id=int(object),
				points=points,
				labels=labels,
			)
				
	video_segments = {}

	n_frames = 0
	for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
		n_frames += 1
		video_segments[out_frame_idx] = {
			out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
			for i, out_obj_id in enumerate(out_obj_ids)
		}

	for frame_idx, obj_dict in tqdm(video_segments.items(), desc="Processing video frames"):
		overall_mask = np.zeros((1080, 1920), dtype=bool)

		for obj_id, mask_array in obj_dict.items():
			overall_mask = np.logical_or(overall_mask, mask_array.squeeze())

		predicted_masks.append(overall_mask)

	sam2_predictor.reset_state(inference_state)
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	# 	print("Emptied CUDA cache.")
	# print("SAM state reset.")
	return predicted_masks

def process_video(model, frames_path, sub_dir, annotation_path, is_left):
	stereo_dir = "left" if is_left else "right"
	ground_truth_masks_path = os.path.join(sub_dir, "ground_truth", stereo_dir)

	predicted_masks = run_inference(model, frames_path, annotation_path)
	# print(sub_dir, domain, stereo_dir)
	ground_truth_masks = []
	for i in range(len(predicted_masks)):
		ground_truth_mask_path = os.path.join(ground_truth_masks_path, f"{i}.png")
		ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
		ground_truth_mask = (ground_truth_mask > 0).astype(np.bool_)
		ground_truth_masks.append(ground_truth_mask)
	# print(f"Loaded {len(ground_truth_masks)} ground truth masks.")

	# print("Calculating evaluation metrics...")
	miou = calculate_miou(predicted_masks, ground_truth_masks)
	mdsc = calculate_mdsc(predicted_masks, ground_truth_masks)

	# print(f"Mean IoU for {sub_dir}/{domain}/{stereo_dir}: {miou:.4f}")
	# print(f"Mean DSC for {sub_dir}/{domain}/{stereo_dir}: {mdsc:.4f}")

	return miou, mdsc

def run_eval(model, sub_dir, domain, annotation_path):
	left_video_frames_path = os.path.join(sub_dir, domain, "left")
	right_video_frames_path = os.path.join(sub_dir, domain, "right")

	left_miou, left_msdc = process_video(model, left_video_frames_path, sub_dir, annotation_path, True)
	right_miou, right_msdc = process_video(model, right_video_frames_path, sub_dir, annotation_path, False)

	overall_miou = (left_miou + right_miou) / 2
	overall_msdc = (left_msdc + right_msdc) / 2
	perf = {"iou": overall_miou, "dsc": overall_msdc}
	return perf
	# print(f"\nResults for {sub_dir}:")
	# print(f"  Left: IoU={left_miou:.4f}, DSC={left_msdc:.4f}")
	# print(f"  Right: IoU={right_miou:.4f}, DSC={right_msdc:.4f}")
	# print(f"  Overall: IoU={overall_miou:.4f}, DSC={overall_msdc:.4f}")
			
		
if __name__ == "__main__":
	dir = "../segstrong/SegSTRONGC_test/test/9/0/"
	domain = "blood"
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	CHECKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"
	CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
	model = build_sam2_video_predictor(CONFIG, CHECKPOINT)
	torch.autocast("cuda", dtype=torch.bfloat16)
	annotation_path = "annotations/auto/test.json"

	run_eval(model, dir, domain, annotation_path)
