import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

# domains = ['bg_change', 'blood', 'low_brightness', 'regular', 'smoke']
# annotations = None
# with open('data/annotations/auto/val.json', 'r') as f:
#     annotations = json.load(f)

# path = "data/masks/yolo11x-seg/val/data-raw-SegSTRONGC_val-val-1-2-bg_change-left.pkl"
# frame_idx = 0
# with open(path, 'rb') as f:
#     mass = pickle.load(f)
#     for video_path, video_data in mass.items():
#         # print(video_path, video_data)
#         for frame_id, frame_data in video_data.items():
#             if frame_id == frame_idx:
#                 overall_mask = frame_data
#                 # for data in frame_data:
#                 #     for object_id, mask in data.items():
#                 #         overall_mask = np.logical_or(overall_mask, mask[0])

#                 ground_truth_masks_path = video_path
#                 for domain in domains:
#                     if domain in video_path:
#                         ground_truth_masks_path = video_path.replace(domain, 'ground_truth')
#                         break
#                 ground_truth_masks_path = ground_truth_masks_path + "/" + str(frame_id) + ".jpg"
#                 ground_truth_mask = cv2.imread(ground_truth_masks_path, cv2.IMREAD_GRAYSCALE)
#                 ground_truth_mask = (ground_truth_mask > 0).astype(np.bool_)

#                 original_image = cv2.imread(video_path + "/" + str(frame_id) + ".jpg")
#                 original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#                 # frame_annotations = annotations[video_path.replace("/left", "").replace("/right", "")]
#                 frame_annotations = annotations[video_path]

#                 # place dots in the original image for each annotation
#                 for object_id, object_annotations in frame_annotations.items():
#                     # print(object_annotations)
#                     for annotation in object_annotations:
#                         x = annotation['x']
#                         y = annotation['y']
#                         label = annotation['label']
#                         if object_id == "0":
#                             original_image = cv2.circle(original_image, (x, y), 10, (0, 255, 0), -1)
#                         else:
#                             original_image = cv2.circle(original_image, (x, y), 10, (255, 0, 0), -1)

#                 #show the masks and the original image
#                 fig, axs = plt.subplots(1, 3, figsize=(30, 15))
#                 axs[0].imshow(original_image)
#                 axs[0].set_title("Original Image")
#                 axs[1].imshow(overall_mask)
#                 axs[1].set_title("Overall Mask")
#                 axs[2].imshow(ground_truth_mask)
#                 axs[2].set_title("Ground Truth Mask")
#                 plt.show()
#                 break

        #     "data/raw/SegSTRONGC_val/val/1/0/bg_change": {
        # "0": [
        #     {
        #         "x": 272,
        #         "y": 542,
        #         "label": 1
        #     },
        #     {
        #         "x": 963,
        #         "y": 346,
        #         "label": 1
        #     }
        # ],
            
        # print(f"Loaded {len(ground_truth_masks)} ground truth masks.")
            

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read and binarize the ground truth mask
ground_truth_mask_path = "data/raw/SegSTRONGC_val/val/1/2/ground_truth/right/0.jpg"
ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
ground_truth_mask = (ground_truth_mask > 0).astype(np.bool_)

# Get connected components
num_labels, labels = cv2.connectedComponents(ground_truth_mask.astype(np.uint8))

# Calculate centroids for each component (skip background label 0)
print(num_labels)
print(labels.shape)
# get unique values in labels
unique_labels = np.unique(labels)
plt.imshow(labels, cmap='gray')
plt.show()

# for each unquie label, print the x, y coordinates and number of pixels
for label in unique_labels:
    y_indices, x_indices = np.where(labels == label)
    print(f"Label: {label}, Centroid: ({int(np.mean(x_indices))}, {int(np.mean(y_indices))}), Pixels: {len(x_indices)}")
centroids = []
for label in range(1, num_labels):
    component = (labels == label)
    y_indices, x_indices = np.where(component)
    centroid = (int(np.mean(x_indices)), int(np.mean(y_indices)))
    centroids.append(centroid)

# # Create subplots
# fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# # 1. Original ground truth mask
# axs[0, 0].imshow(ground_truth_mask, cmap='gray')
# axs[0, 0].set_title('Ground Truth Mask')

# # 2. Connected Components image (heatmap)
# if num_labels > 1:
#     # Calculate the count of each label
#     label_counts = np.bincount(labels.flatten())

#     # Normalize the counts to the range [0, 1]
#     normalized_counts = label_counts.astype(float) / np.max(label_counts)

#     # Create a heatmap based on the normalized counts
#     heatmap = np.zeros_like(labels, dtype=float)
    
#     # Only include the top N most frequent labels
#     N = 5  # Change this value to include more/less labels
#     top_labels = np.argsort(label_counts)[-N:]  # Get the indices of the top N labels

#     top_labels = [2, 89]

#     for label in range(num_labels):
#         if label in top_labels:
#             print("Label: ", label, " Count: ", normalized_counts[label])
#             heatmap[labels == label] = normalized_counts[label]
#         else:
#             heatmap[labels == label] = 0  # Set other labels to zero

#     axs[0, 1].imshow(heatmap, cmap='viridis')  # You can choose a different colormap
#     axs[0, 1].set_title(f'Connected Components (Heatmap, Top {N}: {num_labels-1})')
# else:
#     axs[0, 1].imshow(labels, cmap='gray')
#     axs[0, 1].set_title(f'Connected Components (Heatmap, Total: {num_labels-1})')

# # 3. Ground truth with overlaid centroids
# axs[1, 0].imshow(ground_truth_mask, cmap='gray')
# for centroid in centroids:
#     axs[1, 0].plot(centroid[0], centroid[1], 'r+', markersize=10)
# axs[1, 0].set_title('Ground Truth with Centroids')

# # 4. Centroids-only mask
# centroid_mask = np.zeros_like(ground_truth_mask, dtype=np.uint8)
# for centroid in centroids:
#     centroid_mask = cv2.circle(centroid_mask, centroid, 5, 255, -1)
# axs[1, 1].imshow(centroid_mask, cmap='gray')
# axs[1, 1].set_title('Centroids Mask')

# plt.tight_layout()
# plt.show()

# print(f"Number of components: {num_labels-1}")
# print("Centroids:", centroids)
