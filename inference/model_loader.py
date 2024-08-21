import torch
from models import ConvLSTM2Dlightning

def load_model(checkpoint_path, config):
    input_shape = eval(config['model']['input_shape'])
    num_filters = int(config['model']['num_filters'])
    kernel_size = eval(config['model']['kernel_size'])
    num_blocks = int(config['training']['num_blocks'])
    num_layers_per_block = int(config['training']['num_layers_per_block'])

    model = ConvLSTM2Dlightning(input_shape=input_shape,
                                num_filters=num_filters,
                                kernel_size=kernel_size,
                                num_blocks=num_blocks,
                                num_layers_per_block=num_layers_per_block)

    model = model.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model
