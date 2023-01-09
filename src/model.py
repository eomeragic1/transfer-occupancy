import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
from torch.autograd import Variable
from torchinfo import summary


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1)    # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        self.fc1 = nn.Linear(num_layers*hidden_size*self.factor, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers*self.factor, x.size(0), self.hidden_size, requires_grad=True, device='cuda')  # hidden state
        c_0 = torch.zeros(self.num_layers*self.factor, x.size(0), self.hidden_size, requires_grad=True, device='cuda')  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = torch.flatten(hn.transpose(0, 1), start_dim=1)
        out = self.relu(hn)
        out = self.fc1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc2(out)  # Final Output
        return torch.sigmoid(out)


if __name__ == '__main__':
    model = LSTM(hidden_size=64, num_classes=1, input_size=31, num_layers=1, seq_length=11, bidirectional=True)
    summary(model.cuda(), input_size=(64, 11, 31))
