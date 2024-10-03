import os

import numpy as np
import torch
from inference.video_processor import segment_video


def save_segmented_frames_as_npy(segmented_frames, output_path):
    """Save the segmented frames as a numpy array."""
    np.save(output_path, segmented_frames)


def save_segmented_frames_as_npz(segmented_frames, output_path):
    """Save the segmented frames as a compressed numpy array. The compressed numpy array is going through a sigmoid function and then normalized to 0-255."""
    normalized_video = (
        (torch.sigmoid(torch.tensor(segmented_frames)) * 255).numpy().astype(np.uint8)
    )
    np.savez_compressed(output_path, normalized_video)


def process_videos_in_folder(video_folder, model, output_folder):
    """Process all the videos in the video folder and save the segmented frames as numpy arrays."""
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".avi")]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        output_path = os.path.join(
            output_folder, os.path.splitext(video_file)[0] + ".npy"
        )
        segmented_frames = segment_video(model, video_path)
        save_segmented_frames_as_npy(segmented_frames, output_path)
        output_path = os.path.splitext(output_path)[0] + ".npz"
        save_segmented_frames_as_npz(segmented_frames, output_path)
