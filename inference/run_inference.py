from config import load_config
from inference.model_loader import load_model
from inference.file_saver import process_videos_in_folder

def main():
    config = load_config()

    video_folder = config['paths']['video_folder']
    output_folder = config['inference']['output_folder']
    checkpoint_path = config['inference']['checkpoint_path']
    
    model = load_model(checkpoint_path, config)
    process_videos_in_folder(video_folder, model, output_folder)

if __name__ == "__main__":
    main()
