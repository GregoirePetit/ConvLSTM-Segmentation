import cv2
import numpy as np
import torch


def segment_video(model, video_path):
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
    frames = torch.from_numpy(frames).float().unsqueeze(0)  # Add batch dimension
    frames = frames.permute(
        0, 1, 4, 2, 3
    )  # (batch_size, time_steps, channels, height, width)

    with torch.no_grad():
        y_pred = model(frames.to("cuda" if torch.cuda.is_available() else "cpu"))
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.squeeze().cpu().numpy()
        y_pred = (y_pred > 0.5).astype(np.uint8)  # Thresholding

    return y_pred
