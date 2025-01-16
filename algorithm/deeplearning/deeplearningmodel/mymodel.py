import torch.nn as nn
import torch
from algorithm.deeplearning.deeplearningmodel.tcn import TemporalConvNet
from torch.nn.utils.parametrizations import weight_norm


# 定义LSTM网络
class LSTM(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小
        # 初始化隐层状态

        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # LSTM运算
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))

        # 全连接层
        output = self.fc(output)  # LSTM出来直接送进全连接层
        # 形状为batch_size, timestep, output_size[32，1，1]

        # 我们只需要返回最后一个时间片的数据即可
        return output[:, -1, :]


class CNNGRU(nn.Module):
    def __init__(self, feature_size, out_channel, hidden_size, num_layers, kernel_size, output_size):
        super(CNNGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        # self.covout = timestep - self.kernel_size // 1 + 1

        self.conv = nn.Conv1d(in_channels=feature_size, out_channels=out_channel, kernel_size=kernel_size, padding=1)
        self.gru = nn.GRU(out_channel, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        batch_size = x.shape[0]
        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden
        x, h_0 = self.gru(x, h_0)
        output = self.fc1(x)  # 32,32,8
        return output[:, -1, :]


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=3, dropout=dropout)
        self.fc1 = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = x.transpose(1, 2)
        x = self.tcn(x)  # input should have dimension (N, C, L)
        x = x.transpose(1, 2)

        x = self.fc1(x)
        return x[:, -1, :]


class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # lstm层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小
        # 初始化隐层状态

        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden

        # LSTM运算
        output, h_0 = self.gru(x, h_0)

        # 全连接层
        output = self.fc(output)  # LSTM出来直接送进全连接层
        # 形状为batch_size, timestep, output_size[32，1，1]

        # 我们只需要返回最后一个时间片的数据即可
        return output[:, -1, :]


class DAR(nn.Module):
    def __init__(self, feature_size, num_channels, dilation, kernel_size, output_size):
        super(DAR, self).__init__()
        self.kernel_size = kernel_size
        self.dcc = MultiLayerDilatedCausalConv1d(feature_size, num_channels, kernel_size, dilation)
        self.fc1 = nn.Linear(num_channels[-1], output_size)
        self.downsample = nn.Conv1d(feature_size, num_channels[-1], 1)
        self.gate = ChannelAttention(num_channels[-1])
        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.transpose(1, 2)
        residual = self.downsample(x) #64,64,60
        residual = self.gate(residual)
        x = self.dcc(x)
        #x = self.relu(x).transpose(1,2)
        x = self.relu(x+residual).transpose(1,2)
        x = self.fc1(x)
        return x[:, -1, :]


class ChannelAttention(nn.Module):  # 通道注意力
    def __init__(self, in_channels, reduction_ratio=32):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Tanh()  #nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc(out).view(b, c, 1)
        out = torch.mul(out, x)
        return out


# 单层DCC
class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedCausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation),
                                  #nn.AdaptiveMaxPool1d(1),
                                  nn.ReLU()
                                  )

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x


# 多层DCC+通道注意力
class MultiLayerDilatedCausalConv1d(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dilation):
        super(MultiLayerDilatedCausalConv1d, self).__init__()
        self.num_levels = len(num_channels)
        layers = []
        skip_connections = []

        for i in range(self.num_levels):
            dilation_size = dilation ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            # Add DilatedCausalConv1d and ChannelAttention layers
            layers.append(DilatedCausalConv1d(in_channels, out_channels, kernel_size, dilation_size))
            layers.append(ChannelAttention(out_channels))

            # Skip connections for layers beyond the first
            if i > 0:
                # Ensure the output channels match the current layer's out_channels
                skip_connections.append(nn.Sequential(
                    nn.Conv1d(num_channels[i - 1], num_channels[-1], kernel_size=1, stride=1),
                    #nn.BatchNorm1d(num_channels[-1])  # Adjust LayerNorm to match the number of channels
                ))

        self.conv_layers = nn.ModuleList(layers)
        self.skip_connections = nn.ModuleList(skip_connections)

    def forward(self, x):
        skip_outputs = []
        for i in range(self.num_levels):
            x = self.conv_layers[2 * i](x)  # Apply DilatedCausalConv1d layer
            x = self.conv_layers[2 * i + 1](x)  # Apply ChannelAttention layer
            skip_outputs.append(x)

        # Adding skip connections
        for i in range(1, self.num_levels):
            skip = self.skip_connections[i - 1](skip_outputs[i - 1])
            x = x + skip  # Ensure x and skip have the same number of channels
        #x / len(skip_outputs)
        return x


class MultiLayerDilatedCausalConv1d2(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dilation):
        super(MultiLayerDilatedCausalConv1d2, self).__init__()
        self.num_levels = len(num_channels)
        layers = []
        skip_connections = []

        
        for i in range(self.num_levels):
            dilation_size = dilation ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            # Add DilatedCausalConv1d and ChannelAttention layers
            layers.append(DilatedCausalConv1d(in_channels, out_channels, kernel_size, dilation_size))
            layers.append(ChannelAttention(out_channels))

            # Skip connections for layers beyond the first
            if i > 0:
                # Ensure the output channels match the current layer's out_channels
                skip_connections.append(nn.Sequential(
                    nn.Conv1d(num_channels[i - 1], num_channels[-1], kernel_size=1, stride=1),
                    #nn.BatchNorm1d(num_channels[-1])  # Adjust LayerNorm to match the number of channels
                ))

        self.conv_layers = nn.ModuleList(layers)
        self.skip_connections = nn.ModuleList(skip_connections)

    def forward(self, x):
        skip_outputs = []
        for i in range(self.num_levels):
            x = self.conv_layers[2 * i](x)  # Apply DilatedCausalConv1d layer
            x = self.conv_layers[2 * i + 1](x)  # Apply ChannelAttention layer
            skip_outputs.append(x)

        # Adding skip connections
        for i in range(1, self.num_levels):
            skip = self.skip_connections[i - 1](skip_outputs[i - 1])
            x = x + skip  # Ensure x and skip have the same number of channels
        #x / len(skip_outputs)
        return x
