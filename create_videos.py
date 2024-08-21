import argparse
import os

import cv2
import numpy as np
from config import load_config


def overlay_segmentation_on_video(
    original_video_path, segmentation_npy_path, output_video_path
):
    # Load the original video
    cap = cv2.VideoCapture(original_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load the segmentation .npy file
    segmentation = np.load(segmentation_npy_path)

    # Check dimensions match
    if len(segmentation) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        raise ValueError(
            "The number of frames in the segmentation does not match the original video."
        )

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the segmentation mask to match the video frame
        seg_mask = cv2.resize(segmentation[frame_idx], (width, height))

        # Threshold the mask to get a binary mask
        seg_mask = (seg_mask > 0.5).astype(np.uint8) * 255
        # Create an overlay by converting mask to a 3-channel image
        overlay = np.zeros_like(frame)
        overlay[:, :, 2] = seg_mask  # Add the segmentation mask to the red channel

        # Apply the overlay on the original frame
        combined_frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

        # Write the frame to the output video
        out.write(combined_frame)
        frame_idx += 1

    cap.release()
    out.release()


def process_all_videos_in_folder(video_folder, segmentation_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".avi")]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        segmentation_npy_path = os.path.join(
            segmentation_folder, os.path.splitext(video_file)[0] + ".npy"
        )
        output_video_path = os.path.join(output_folder, video_file)

        overlay_segmentation_on_video(
            video_path, segmentation_npy_path, output_video_path
        )


def main():
    parser = argparse.ArgumentParser(description="Train ConvLSTM2D model")
    parser.add_argument(
        "--config", type=str, default="config_file.cf", help="path to config file"
    )
    args = parser.parse_args()
    path_config = args.config

    config = load_config(path_config)
    video_folder = config["paths"]["video_folder"]
    segmentation_folder = config["inference"]["output_folder"]
    output_folder = config["inference"]["overlay_video_output_folder"]

    process_all_videos_in_folder(video_folder, segmentation_folder, output_folder)


if __name__ == "__main__":
    main()
