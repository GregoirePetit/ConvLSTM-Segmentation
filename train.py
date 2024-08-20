import os

import torch
from config import load_config
from data_loader import VideoFrameDataset, read_file_list
from models import ConvLSTM2Dlightning  # Import the lightning model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import suggested_max_num_workers


def main():
    config = load_config()
    video_folder = config["paths"]["video_folder"]
    labels_folder = config["paths"]["labels_folder"]
    csv_path = config["paths"]["csv_path"]
    logger_path = config["paths"]["logger_path"]

    batch_size = int(config["training"]["batch_size"])
    sequence_length = int(config["training"]["sequence_length"])
    max_epochs = int(config["training"]["max_epochs"])
    num_blocks = int(config["training"]["num_blocks"])
    num_layers_per_block = int(config["training"]["num_layers_per_block"])
    learning_rate = float(config["training"]["learning_rate"])

    input_shape = eval(config["model"]["input_shape"])
    num_filters = int(config["model"]["num_filters"])
    kernel_size = eval(config["model"]["kernel_size"])

    train_videos, train_labels, val_videos, val_labels, test_videos, test_labels = (
        read_file_list(csv_path, video_folder, labels_folder)
    )

    train_gen = VideoFrameDataset(
        train_videos, train_labels, sequence_length=sequence_length
    )
    val_gen = VideoFrameDataset(val_videos, val_labels, sequence_length=sequence_length)
    test_gen = VideoFrameDataset(
        test_videos, test_labels, sequence_length=sequence_length
    )
    local_world_size = torch.cuda.device_count()
    num_workers = suggested_max_num_workers(local_world_size)

    train_loader = torch.utils.data.DataLoader(
        train_gen, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_gen, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_gen, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = ConvLSTM2Dlightning(
        input_shape=input_shape,
        num_filters=num_filters,
        kernel_size=kernel_size,
        num_blocks=num_blocks,
        num_layers_per_block=num_layers_per_block,
    )

    logger = TensorBoardLogger(
        logger_path, name=f"ConvLSTM2D_lightning_B{num_blocks}_l{num_layers_per_block}"
    )

    callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=5,
        filename="model-{epoch:02d}-{val_loss:.2f}",
    )

    exists = False
    i = 0
    last_checkpoint = None
    while os.path.exists(
        f"{logger_path}/torch_B{num_blocks}_l{num_layers_per_block}/ConvLSTM2D_lightning_B{num_blocks}_l{num_layers_per_block}/version_{i}"
    ):
        i += 1
        if os.path.exists(
            f"{logger_path}/torch_B{num_blocks}_l{num_layers_per_block}/ConvLSTM2D_lightning_B{num_blocks}_l{num_layers_per_block}/version_{i-1}/last.ckpt"
        ):
            exists = True
            last_checkpoint = f"{logger_path}/torch_B{num_blocks}_l{num_layers_per_block}/ConvLSTM2D_lightning_B{num_blocks}_l{num_layers_per_block}/version_{i-1}/last.ckpt"

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        devices=-1,
        accelerator="gpu",
        strategy="ddp",
        callbacks=[callback],
    )

    if not exists:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_checkpoint)

    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
