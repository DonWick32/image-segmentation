domain = "smoke"
video_path = "data/raw/SegSTRONGC_val/val/1/2/" + domain + "/left"

#this has frames, i need you to create a video from these frames
import cv2
import os
import glob
import numpy as np
import random
import shutil

def create_video_from_frames(frame_folder, output_video_path):
    first_image = cv2.imread(frame_folder + '/0.png')
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

    for i in range(0, 300):
        img_path = os.path.join(frame_folder, f'{i}.png')
        img = cv2.imread(img_path)
        video_writer.write(img)

    video_writer.release()
    cv2.destroyAllWindows()
# Create the output video path
output_video_path = os.path.join("data/results", 'output_video.mp4')

# Create the video from frames
create_video_from_frames(video_path, output_video_path)