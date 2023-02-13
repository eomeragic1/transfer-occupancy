import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
from torch.autograd import Variable
from torchinfo import summary
import math


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, bidirectional=False):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.factor = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear(num_layers*hidden_size*self.factor, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers*self.factor, x.size(0), self.hidden_size, requires_grad=True, device='cuda')  # hidden state
        c_0 = torch.zeros(self.num_layers*self.factor, x.size(0), self.hidden_size, requires_grad=True, device='cuda')  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0.detach(), c_0.detach()))  # lstm with input, hidden, and internal state
        hn = torch.flatten(hn.transpose(0, 1), start_dim=1)
        out = self.relu(hn)
        out = self.fc1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc2(out)  # Final Output
        return torch.sigmoid(out)


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bidirectional=False):
        super(GRU, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.num_classes = output_dim
        self.factor = 2 if bidirectional else 1
        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=bidirectional
        )

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim * self.factor, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device='cuda').requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc2(self.fc1(out))

        return torch.sigmoid(out)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = nn.ReLU()(bn1)
        conv2 = self.conv2(relu1)
        bn2 = nn.ReLU()(self.bn2(conv2))
        summed = bn2+relu1
        out = nn.ReLU()(summed)
        return out

class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_past=None, n_attr=None):
        super(ConvNet, self).__init__()
        self.conv1 = ResNetBlock(in_channels, out_channels=hidden_channels)
        self.conv2 = ResNetBlock(hidden_channels, 2*hidden_channels)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        w_out1 = math.floor((n_past-5+2)/2 + 1)
        h_out1 = math.floor((n_attr-5+2)/2 + 1)

        w_out2 = math.floor((w_out1-5+2)/2+1)
        h_out2 = math.floor((h_out1-5+2)/2+1)

        w_out3 = math.floor((w_out2 - 3 + 2) / 2 + 1)
        h_out3 = math.floor((h_out2 - 3 + 2) / 2 + 1)

        fc1_input = 2*hidden_channels*w_out3*h_out3
        self.fc1 = nn.Linear(fc1_input, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.relu = nn.ReLU()
        self.num_classes = 1

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class LSTMWithAudio(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, audio_in_channels, audio_hidden_channels, bidirectional=False):
        super(LSTMWithAudio, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.factor = 2 if bidirectional else 1
        self.conv1_audio = ResNetBlock(audio_in_channels, out_channels=audio_hidden_channels)
        self.conv2_audio = ResNetBlock(audio_hidden_channels, 2 * audio_hidden_channels)
        self.conv1_env = ResNetBlock(seq_length, seq_length, kernel_size=(3, 5))

        self.pool_audio = nn.MaxPool2d(3, stride=2, padding=1)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)

        w_out1 = math.floor((64 - 5 + 2) / 2 + 1)
        h_out1 = math.floor((10 - 5 + 2) / 2 + 1)

        w_out2 = math.floor((w_out1 - 5 + 2) / 2 + 1)
        h_out2 = math.floor((h_out1 - 5 + 2) / 2 + 1)

        w_out3 = math.floor((w_out2 - 3 + 2) / 2 + 1)
        h_out3 = math.floor((h_out2 - 3 + 2) / 2 + 1)

        fc1_input = 2 * audio_hidden_channels * w_out3 * h_out3

        self.fc1 = nn.Linear(num_layers * hidden_size * self.factor + fc1_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x[0] - environment tensor, x[1] - audio tensor
        h_0 = torch.zeros(self.num_layers * self.factor, x[0].size(0), self.hidden_size, requires_grad=True,
                          device='cuda')  # hidden state
        c_0 = torch.zeros(self.num_layers * self.factor, x[0].size(0), self.hidden_size, requires_grad=True,
                          device='cuda')  # internal state

        x_env = self.conv1_env(torch.transpose(x[0], 1, 2))
        x_env = torch.flatten(x_env, 2)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x_env, (h_0.detach(), c_0.detach()))  # lstm with input, hidden, and internal state
        hn = torch.flatten(hn.transpose(0, 1), start_dim=1)

        x_audio = self.conv1_audio(x[1])
        x_audio = self.conv2_audio(x_audio)
        x_audio = self.pool_audio(x_audio)
        x_audio = torch.flatten(x_audio, start_dim=1)

        out = torch.cat((hn, x_audio), 1)
        out = self.relu(out)
        out = self.fc1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc2(out)  # Final Output
        return torch.sigmoid(out)


class GRUWithAudio(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, audio_in_channels, audio_hidden_channels, seq_length, bidirectional=False):
        super(GRUWithAudio, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.num_classes = output_dim
        self.factor = 2 if bidirectional else 1
        self.conv1_audio = ResNetBlock(audio_in_channels, out_channels=audio_hidden_channels)
        self.conv2_audio = ResNetBlock(audio_hidden_channels, 2 * audio_hidden_channels)
        self.pool_audio = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv1_env = ResNetBlock(seq_length, seq_length, kernel_size=(3, 5))
        w_out1 = math.floor((64 - 5 + 2) / 2 + 1)
        h_out1 = math.floor((10 - 5 + 2) / 2 + 1)

        w_out2 = math.floor((w_out1 - 5 + 2) / 2 + 1)
        h_out2 = math.floor((h_out1 - 5 + 2) / 2 + 1)

        w_out3 = math.floor((w_out2 - 3 + 2) / 2 + 1)
        h_out3 = math.floor((h_out2 - 3 + 2) / 2 + 1)

        fc1_input = 2 * audio_hidden_channels * w_out3 * h_out3
        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=bidirectional
        )

        # Fully connected layer
        self.fc1 = nn.Linear(self.factor * hidden_dim + fc1_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device='cuda').requires_grad_()

        x_env = self.conv1_env(torch.transpose(x[0], 1, 2))
        x_env = torch.flatten(x_env, 2)
        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x_env, h0.detach())
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        x_audio = self.conv1_audio(x[1])
        x_audio = self.conv2_audio(x_audio)
        x_audio = self.pool_audio(x_audio)
        x_audio = torch.flatten(x_audio, start_dim=1)

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = torch.cat((out, x_audio), 1)
        out = self.relu(out)
        out = self.fc1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc2(out)  # Final Output
        return torch.sigmoid(out)

class ConvNetWithAudio(nn.Module):
    def __init__(self, in_channels, hidden_channels, audio_in_channels, audio_hidden_channels, n_past=None, n_attr=None):
        super(ConvNetWithAudio, self).__init__()
        self.conv1 = ResNetBlock(in_channels, out_channels=hidden_channels)
        self.conv2 = ResNetBlock(hidden_channels, 2*hidden_channels)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv1_audio = ResNetBlock(audio_in_channels, out_channels=audio_hidden_channels)
        self.conv2_audio = ResNetBlock(audio_hidden_channels, 2 * audio_hidden_channels)
        self.pool_audio = nn.MaxPool2d(3, stride=2, padding=1)

        w_out1_audio = math.floor((64 - 5 + 2) / 2 + 1)
        h_out1_audio = math.floor((10 - 5 + 2) / 2 + 1)

        w_out2_audio = math.floor((w_out1_audio - 5 + 2) / 2 + 1)
        h_out2_audio = math.floor((h_out1_audio - 5 + 2) / 2 + 1)

        w_out3_audio = math.floor((w_out2_audio - 3 + 2) / 2 + 1)
        h_out3_audio = math.floor((h_out2_audio - 3 + 2) / 2 + 1)

        fc1_input_audio = 2 * audio_hidden_channels * w_out3_audio * h_out3_audio

        w_out1 = math.floor((n_past-5+2)/2 + 1)
        h_out1 = math.floor((n_attr-5+2)/2 + 1)

        w_out2 = math.floor((w_out1-5+2)/2+1)
        h_out2 = math.floor((h_out1-5+2)/2+1)

        w_out3 = math.floor((w_out2 - 3 + 2) / 2 + 1)
        h_out3 = math.floor((h_out2 - 3 + 2) / 2 + 1)

        fc1_input = 2*hidden_channels*w_out3*h_out3 + fc1_input_audio
        self.fc1 = nn.Linear(fc1_input, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.relu = nn.ReLU()
        self.num_classes = 1

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = torch.flatten(x, start_dim=1)

        x_audio = self.conv1_audio(x[1])
        x_audio = self.conv2_audio(x_audio)
        x_audio = self.pool_audio(x_audio)
        x_audio = torch.flatten(x_audio, start_dim=1)

        x = torch.cat((x, x_audio), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class ROBODNaiveClassifier(nn.Module):
    def __init__(self):
        self.num_classes = 1
        super(ROBODNaiveClassifier, self).__init__()

    def forward(self, x):
        sin = x[:, -1, -7].unsqueeze(1)  # Timestamp sin and cos
        cos = x[:, -1, -6].unsqueeze(1)
        acos = torch.acos(cos) * 360 / (2 * 3.14159)
        angle = torch.randn(sin.size(), device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        angle[sin < 0] = -acos[sin < 0] % 360
        angle[sin >= 0] = acos[sin >= 0]
        hours = angle * 24 / 360
        condition = ((hours >= 9) & (hours <= 17))
        angle[condition] = 1
        angle[~condition] = 0
        return angle

class ECONaiveClassifier(nn.Module):
    def __init__(self):
        super(ECONaiveClassifier, self).__init__()
        self.num_classes = 1

    def forward(self, x):
        hours = x[:, -1, -16:-6].sum(axis=1).unsqueeze(1)
        mask = hours > 0
        output = torch.zeros(hours.size(), device='cuda')
        output[mask] = 1
        return output

class HPDMobileNaiveClassifier(nn.Module):
    def __init__(self):
        super(HPDMobileNaiveClassifier, self).__init__()
        self.num_classes = 1

    def forward(self, x):
        env = x[0].to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        hours = env[:, -1, -1, -16:-6].sum(axis=1).unsqueeze(1)
        mask = hours > 0
        output = torch.ones(hours.size(), device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        output[mask] = 0
        return output


if __name__ == '__main__':
    #model = LSTM(hidden_size=64, num_classes=1, input_size=31, num_layers=1, seq_length=11, bidirectional=True)
    #summary(model.cuda(), input_size=(64, 11, 31))

    model = LSTMWithAudio(num_classes=1, input_size=32, hidden_size=32, num_layers=2, seq_length=16, audio_in_channels=5, audio_hidden_channels=16, bidirectional=False)
    summary(model.cuda(), input_size=((1, 11, 32), (5, 10, 64)), batch_dim=0)
