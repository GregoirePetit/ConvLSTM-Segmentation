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


def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("can't open file - does it exist ? ")
        print(video_path)
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frames.append(frame)
    
    cap.release()
    frames = np.array(frames)
    return frames  # Retourne bien les frames

class VideoFrameDataset(Dataset):
    """
    VideoFrameDataset: Dataset class for loading video frames and labels
    """

    def __init__(self, video_files, label_files, sequence_length=25, frames_size = None):
        self.video_files = video_files
        self.label_files = label_files
        self.sequence_length = sequence_length
        self.frames_size = frames_size
    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label_path = self.label_files[idx]
        frames = open_video(video_path)
        
        
        
        # frames shape here : (frames_in_clip, 112, 112)
        frames = np.expand_dims(frames, -1)  # Add channel dimension
        # frames shape here : (frames_in_clip, 112, 112, 1)

        if frames.shape[0] < self.sequence_length:
            raise ValueError("Video length is shorter than the sequence length : " + repr(video_path))

        label_extension = label_path[-4:]
        if  label_extension == ".npy":
            label = np.load(label_path)
            label = (label >= 0).astype(np.float32)
        elif label_extension == ".npz":
            content = np.load(label_path)
            if "arr_0" in content:
                label = np.load(label_path)["arr_0"].astype(np.float32)
            elif "output" in content:
                label = np.load(label_path)["output"].astype(np.float32)[:,0,...]
            else:
                raise Exception
            label = label/255
        else:
            raise Exception
        
        label = np.expand_dims(label, -1)  # Add channel dimension

        maximal_available_idx =  min(frames.shape[0] - self.sequence_length, label.shape[0] - self.sequence_length) 
        start_idx = np.random.randint(0,maximal_available_idx + 1)
        end_idx = start_idx + self.sequence_length

        X = frames[start_idx:end_idx]
        y = label[start_idx:end_idx]
        
        if not self.frames_size is None: 
            if self.frames_size == "halfed": # makes the training phase lighter for quick tests / debugging
                X = X[:,::2,::2,:]
                y = y[:,::2,::2,:]

        assert X.shape[0] >= self.sequence_length
        assert y.shape[0] >= self.sequence_length

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        X = X.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        return X, y
