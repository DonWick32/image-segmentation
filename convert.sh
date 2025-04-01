#!/bin/bash

# Script to convert all PNG images to JPG in a specified folder and its subfolders.
# Displays progress for each folder processed.
# Uses the 'mogrify' command from ImageMagick.
# After successful JPG conversion, it removes the original PNG files.

# **Important Warning:**
# This script will **DELETE** your original PNG files after conversion.
# Please double-check the TARGET_FOLDER variable and make a backup before running!

# --- CONFIGURATION ---
TARGET_FOLDER="/mnt/g/Segstrong"  # **CHANGE THIS TO YOUR DESIRED FOLDER PATH**
# --- END CONFIGURATION ---

# Check if the TARGET_FOLDER exists and is a directory
if [ ! -d "$TARGET_FOLDER" ]; then
  echo "Error: TARGET_FOLDER '$TARGET_FOLDER' does not exist or is not a directory."
  exit 1
fi

echo "Starting PNG to JPG conversion with folder progress in: '$TARGET_FOLDER' and subfolders..."

current_folder=""
processed_count=0
total_folder_images=0

# Find all PNG files recursively starting from the TARGET_FOLDER
find "$TARGET_FOLDER" -name "*.png" -print0 | while IFS= read -r -d $'\0' png_file; do
  folder=$(dirname "$png_file") # Extract the folder path
  jpg_file="${png_file%.png}.jpg"

  # Check if we've entered a new folder
  if [ "$folder" != "$current_folder" ]; then
    if [ -n "$current_folder" ]; then # If not the very first folder, finish previous folder info
      echo "Finished folder: '$current_folder'. Processed $processed_count out of $total_folder_images images."
    fi

    current_folder="$folder"
    processed_count=0

    # Count total PNG images in the current folder
    total_folder_images=$(find "$current_folder" -maxdepth 1 -name "*.png" | wc -l)
    echo "Processing folder: '$current_folder'. Total PNG images: $total_folder_images"
  fi

  # Execute mogrify command to convert PNG to JPG
  mogrify -format jpg "$png_file"

  # Check if the JPG file was created successfully
  if [ -f "$jpg_file" ]; then
    processed_count=$((processed_count + 1))
    echo "Converted: '$png_file' to '$jpg_file' - Progress in '$current_folder': $processed_count/$total_folder_images"

    # Remove the original PNG file
    rm "$png_file"
    # echo "Removed original PNG file: '$png_file'" # Removed to reduce verbosity
  else
    echo "Error: Conversion to JPG failed for: '$png_file'. PNG file NOT removed."
    echo "Please check for ImageMagick errors or permissions issues."
  fi
done

# After the loop finishes, handle the last folder
if [ -n "$current_folder" ]; then
  echo "Finished folder: '$current_folder'. Processed $processed_count out of $total_folder_images images."
fi

echo "Conversion and PNG removal process complete with folder progress."
echo "**IMPORTANT:** Original PNG files have been DELETED if JPG conversion was successful."