# ConvLSTM-Segmentation

## Overview

**ConvLSTM-Segmentation** is a deep learning-based framework designed for video segmentation using a ConvLSTM architecture.
This architecture is based on the ConvLSTM layer used in a U-Net-like architecture to segment temporal sequences in video data. The model is trained end-to-end on video data with corresponding ground-truth labels and can be used to predict segmentation masks for new video sequences. This repository contains the code for training, validating, and testing the ConvLSTM-based model, which is specifically tailored to segment temporal sequences in video data.

![ConvLSTM-Segmentation](./figures/fig1.svg)

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

Alternatively, you can install the required packages using the provided [requirements.txt](https://github.com/GregoirePetit/ConvLSTM-Segmentation/blob/main/requirements.txt) file:

```bash
python3 -m venv ~/convlstm
source ~/convlstm/bin/activate
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone git@github.com:GregoirePetit/ConvLSTM-Segmentation.git
cd ConvLSTM-Segmentation
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
├── run_inference.py         # Script to run inference
├── README.md                # Project documentation
├── requirements.txt         # Python package dependencies
├── data/
|   ├── videos/              # Directory containing video files in .avi format
|   ├── labels/              # Directory containing label files in .npy format
|   └── FileList.csv         # CSV file listing files in the train/val/test splits
└── inference/
    ├── file_saver.py        # Utility to save segmentation masks from a folder of video files and a given model
    ├── model_loader.py      # Utility to load a model from a checkpoint path and config
    └── video_processor.py   # Utility to process a video file and save the segmentation mask from a given model
```

## Usage
# Prepare Your Dataset

Ensure that your dataset is structured as expected. You need to have a CSV file listing the video files and corresponding label files, and these files should be placed in the correct directories as specified in [config_file.cf](https://github.com/GregoirePetit/ConvLSTM-Segmentation/blob/main/config_file.cf).
# Configure Paths and Hyperparameters

Edit the config_file.cf file to specify the paths to your dataset and adjust hyperparameters like batch size, learning rate, and model architecture settings.
#Train the Model

Run the following command to start training:
    
```bash
python train.py --config config_file.cf
```

This script will automatically load the data, initialize the model, and begin training. It also supports checkpointing and resuming from the last saved state.

# Logging and Checkpoints

Training logs and model checkpoints will be saved in the directory specified in the configuration file. You can monitor the training process using TensorBoard:

```bash
tensorboard --logdir logger_path
```

# Run Inference

To run inference on a video file, use the run_inference.py script:
    
```bash
python run_inference.py --video_path path/to/video.avi
```

# Project example that uses this code

[EchoDFKD](https://github.com/GregoirePetit/EchoDFKD) is a project that uses this code to segment the left ventricle in echocardiography videos in a data-free knowledge distillation setup.

# Citation

If you find this code useful in your research, please consider citing using the following BibTeX entry, authors are Grégoire Petit and Nathan Palluau

```bash
@misc{petit2024convlstmsegmentation,
    author = {Petit, Grégoire and Palluau, Nathan},
    title = {ConvLSTM-Segmentation},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/GregoirePetit/ConvLSTM-Segmentation}}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/GregoirePetit/ConvLSTM-Segmentation/blob/main/LICENCE) file for more details.
