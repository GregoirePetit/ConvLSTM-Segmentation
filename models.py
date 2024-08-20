import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss


class ConvLSTM2DCell(nn.Module):
    """
    ConvLSTM2DCell: Convolutional LSTM 2D cell
    """

    def __init__(self, x_channels, hidden_channels, kernel_size, padding, stride=1):
        super(ConvLSTM2DCell, self).__init__()
        self.x_channels = x_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.conv = nn.Conv2d(
            in_channels=x_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

    def forward(self, x, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((x, h_prev), dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM2D(nn.Module):
    """
    ConvLSTM2D: Convolutional LSTM 2D layer
    """

    def __init__(self, x_channels, hidden_channels, kernel_size, padding, stride=1):
        super(ConvLSTM2D, self).__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTM2DCell(
            x_channels=x_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        h, c = self.init_hidden(batch_size, height, width)
        output_inner = []
        for i in range(time_steps):
            h, c = self.cell(x[:, i, :, :, :], (h, c))
            output_inner.append(h)
        return torch.stack(output_inner, dim=1)

    def init_hidden(self, batch_size, height, width):
        return (
            torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=self.cell.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=self.cell.conv.weight.device,
            ),
        )


class TimeDistributed(nn.Module):
    """
    TimeDistributed: Time distributed layer, to apply a layer to each time step of the
    input
    """

    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        batch_size, time_steps = x.size(0), x.size(1)
        x_reshaped = x.contiguous().view(-1, *x.size()[2:])
        output = self.layer(x_reshaped)
        if isinstance(output, tuple):
            output = tuple(
                [o.view(batch_size, time_steps, *o.size()[1:]) for o in output]
            )
        else:
            output = output.view(batch_size, time_steps, *output.size()[1:])
        return output


class ConvLSTM2DBlock(nn.Module):
    def __init__(
        self,
        input_shape,
        num_filters,
        kernel_size,
        num_layers,
        last_block=False,
        num_block=0,
    ):
        super(ConvLSTM2DBlock, self).__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_block = num_block
        self.conv_lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            x_channels = input_shape[-1] if i == 0 else num_filters
            self.conv_lstm_layers.append(
                ConvLSTM2D(x_channels, num_filters, kernel_size, padding=(1, 1))
            )
            self.conv_lstm_layers.append(TimeDistributed(nn.BatchNorm2d(num_filters)))
        if not last_block:
            self.conv_lstm_layers.append(
                TimeDistributed(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            )

    def forward(self, x):
        for i in range(len(self.conv_lstm_layers)):
            x = self.conv_lstm_layers[i](x)
        return x


class ConvLSTM2DUpSamplingBlock(nn.Module):
    """
    ConvLSTM2DUpSamplingBlock: ConvLSTM2D up-sampling block, to be placed after the
    down-sampling blocks
    """

    def __init__(self, num_filters, kernel_size, num_block=0):
        super(ConvLSTM2DUpSamplingBlock, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_block = num_block
        self.conv_lstm_layers = nn.ModuleList()
        self.conv_lstm_layers.append(TimeDistributed(nn.Upsample(scale_factor=2)))
        self.conv_lstm_layers.append(
            TimeDistributed(
                nn.Conv2d(
                    num_filters * 2, num_filters, kernel_size=kernel_size, padding=1
                )
            )
        )
        self.conv_lstm_layers.append(TimeDistributed(nn.BatchNorm2d(num_filters)))

    def forward(self, x):
        for i in range(len(self.conv_lstm_layers)):
            x = self.conv_lstm_layers[i](x)
        return x


class ConvLSTM2DModel(nn.Module):
    """
    ConvLSTM2DModel: ConvLSTM2D model, composed of sequences of ConvLSTM2D blocks and
    up-sampling blocks
    """

    def __init__(
        self, input_shape, num_filters, kernel_size, num_blocks, num_layers_per_block
    ):
        super(ConvLSTM2DModel, self).__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers_per_block = num_layers_per_block
        self.num_blocks = num_blocks
        self.conv_lstm_blocks = nn.ModuleList()
        for i in range(num_blocks):
            last_block = i == num_blocks - 1
            self.conv_lstm_blocks.append(
                ConvLSTM2DBlock(
                    input_shape,
                    num_filters * (2**i),
                    kernel_size,
                    num_layers_per_block,
                    last_block,
                    i,
                )
            )
            input_shape = (input_shape[0], input_shape[1] // 2, num_filters * (2**i))
        self.conv_lstm_upsampling = nn.ModuleList()
        for i in range(num_blocks - 2, -1, -1):
            self.conv_lstm_upsampling.append(
                ConvLSTM2DUpSamplingBlock(num_filters * (2**i), kernel_size, i)
            )
        self.output_layer = TimeDistributed(
            nn.Conv2d(num_filters, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        for i in range(len(self.conv_lstm_blocks)):
            x = self.conv_lstm_blocks[i](x)
        for i in range(len(self.conv_lstm_upsampling)):
            x = self.conv_lstm_upsampling[i](x)
        x = self.output_layer(x)
        return x


# Create a lightning module out of the model
class ConvLSTM2Dlightning(pl.LightningModule):
    """
    ConvLSTM2Dlightning: PyTorch Lightning module for the ConvLSTM2DModel
    """

    def __init__(
        self, input_shape, num_filters, kernel_size, num_blocks, num_layers_per_block
    ):
        super(ConvLSTM2Dlightning, self).__init__()
        self.model = ConvLSTM2DModel(
            input_shape, num_filters, kernel_size, num_blocks, num_layers_per_block
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = bce_loss(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = bce_loss(y_hat, y)
        self.log("val_loss", val_loss, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
