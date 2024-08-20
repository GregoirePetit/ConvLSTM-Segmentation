# ConvLSTM-Segmentation

## Overview

**ConvLSTM-Segmentation** is a deep learning-based framework designed for video segmentation using a ConvLSTM architecture. This repository contains the code for training, validating, and testing the ConvLSTM-based model, which is specifically tailored to segment temporal sequences in video data.

## Features

- **ConvLSTM Architecture**: Utilizes ConvLSTM layers to capture both spatial and temporal information from video frames.
- **Configurable Model**: Easily adjust the number of ConvLSTM blocks, layers, and other hyperparameters through a configuration file.
- **PyTorch Lightning**: Leverages the PyTorch Lightning framework for scalable and efficient training.
- **Custom Dataset Support**: Supports custom video datasets with flexible data loading and processing mechanisms.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- PyTorch 1.10+
- OpenCV
- PyTorch Lightning
- NumPy
- Pandas

### Clone the Repository

```bash
git clone https://github.com/GregoirePetit/ConvLSTM-Segmentation.git
cd ConvLSTM-Segmentation
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Directory Structure

The repository is structured as follows:

```
ConvLSTM-Segmentation/
│
├── config_file.cf           # Configuration file for paths and hyperparameters
├── models.py                # Model architecture definitions
├── data_loader.py           # Data loading and preprocessing utilities
├── train.py                 # Script to train and validate the model
├── README.md                # Project documentation
├── requirements.txt         # Python package dependencies
└── data/
    ├── videos/              # Directory containing video files in .avi format
    ├── labels/              # Directory containing label files in .npy format
    └── FileList.csv         # CSV file listing files in the train/val/test splits
```

## Usage
# Prepare Your Dataset

Ensure that your dataset is structured as expected. You need to have a CSV file listing the video files and corresponding label files, and these files should be placed in the correct directories as specified in config_file.cf.
# Configure Paths and Hyperparameters

Edit the config_file.cf file to specify the paths to your dataset and adjust hyperparameters like batch size, learning rate, and model architecture settings.
#Train the Model

Run the following command to start training:
    
    ```bash
    python train.py
    ```

This script will automatically load the data, initialize the model, and begin training. It also supports checkpointing and resuming from the last saved state.
# Evaluate the Model

After training, the model can be evaluated on the test set:
    
        ```bash
        python train.py --evaluate
        ```

# Logging and Checkpoints

Training logs and model checkpoints will be saved in the directory specified in the configuration file. You can monitor the training process using TensorBoard:
    
    ```bash
    tensorboard --logdir logs
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
