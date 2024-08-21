import argparse

from config import load_config
from inference.file_saver import process_videos_in_folder
from inference.model_loader import load_model


def main():
    parser = argparse.ArgumentParser(description="Run inference on videos")
    parser.add_argument(
        "--config", type=str, default="config_file.cf", help="path to config file"
    )
    args = parser.parse_args()
    path_config = args.config

    config = load_config(path_config)

    video_folder = config["paths"]["video_folder"]
    output_folder = config["inference"]["output_folder"]
    checkpoint_path = config["inference"]["checkpoint_path"]

    model = load_model(checkpoint_path, config)
    process_videos_in_folder(video_folder, model, output_folder)


if __name__ == "__main__":
    main()
