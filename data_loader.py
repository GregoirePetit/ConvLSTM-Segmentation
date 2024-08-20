import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_file_list(csv_path, video_folder, labels_folder):
    """
    read_file_list: Read the CSV file and return the file paths for training,
    validation, and testing
    """
    df = pd.read_csv(csv_path)
    train_files = df[df["Split"] == "TRAIN"]["FileName"].tolist()
    val_files = df[df["Split"] == "VAL"]["FileName"].tolist()
    test_files = df[df["Split"] == "TEST"]["FileName"].tolist()

    def get_file_paths(file_list):
        video_files = [os.path.join(video_folder, f"{file}.avi") for file in file_list]
        label_files = [os.path.join(labels_folder, f"{file}.npy") for file in file_list]
        return video_files, label_files

    train_videos, train_labels = get_file_paths(train_files)
    val_videos, val_labels = get_file_paths(val_files)
    test_videos, test_labels = get_file_paths(test_files)

    return train_videos, train_labels, val_videos, val_labels, test_videos, test_labels


class VideoFrameDataset(Dataset):
    """
    VideoFrameDataset: Dataset class for loading video frames and labels
    """

    def __init__(self, video_files, label_files, sequence_length=25):
        self.video_files = video_files
        self.label_files = label_files
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label_path = self.label_files[idx]

        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            frames.append(frame)

        cap.release()
        frames = np.array(frames)
        frames = np.expand_dims(frames, -1)  # Add channel dimension

        if frames.shape[0] < self.sequence_length:
            raise ValueError("Video length is shorter than the sequence length.")

        label = np.load(label_path)
        label = (label >= 0).astype(np.float32)  # Convert labels: 0 if < 0, 1 otherwise
        label = np.expand_dims(label, -1)  # Add channel dimension

        start_idx = np.random.randint(0, frames.shape[0] - self.sequence_length + 1)
        end_idx = start_idx + self.sequence_length

        X = frames[start_idx:end_idx]
        y = label[start_idx:end_idx]

        X = torch.from_numpy(X).float()  # Convert to torch tensor
        y = torch.from_numpy(y).float()  # Convert to torch tensor

        X = X.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        return X, y
