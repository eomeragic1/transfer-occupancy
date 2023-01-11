import torch
from torch import nn
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

def log_loss_score(predicted, actual, eps=1e-14):
    """
    :param predicted:   The predicted probabilities as floats between 0-1
    :param actual:      The binary labels. Either 0 or 1.
    :param eps:         Log(0) is equal to infinity, so we need to offset our predicted values slightly by eps from 0 or 1
    :return:            The logarithmic loss between the predicted probability assigned to the possible outcomes for item i, and the actual outcome.
    """

    predicted = np.clip(predicted, eps, 1 - eps)
    loss = -1 * np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

    return loss

def train(model, loader, optimizer):
    model.train()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    with tqdm(total=len(loader.dataset), desc='Training',
              unit='chunks') as prog_bar:
        for i, data in enumerate(loader):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            if isinstance(data[0], list):
                x, label = (data[0][0].to(device), data[0][1].to(device)), data[1].to(device)
            else:
                x, label = data[0].to(device), data[1].to(device)

            model.zero_grad()

            # Step 3. Run our forward pass.
            pred = model(x)
            label = label.view(-1, model.num_classes).type(torch.FloatTensor).to(device)
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()
            prog_bar.set_postfix(**{'run:': 'LSTM', 'lr': 0.0001,
                                    'loss': loss.item()
                                    })
            if isinstance(x, tuple):
                prog_bar.update(x[0].size(0))
            else:
                prog_bar.update(x.size(0))

    return model


def test(model, loader):
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    f1 = []
    acc = []
    total = 0
    loss = []
    for data in loader:
        x, label = data[0].to(device), data[1].to(device)
        output = model(x).detach().cpu().numpy()
        label = label.view(-1, model.num_classes).detach().cpu().numpy().astype('float32')
        loss.append(log_loss_score(output.squeeze(), label.squeeze()))
        pred = (output > 0.5).astype('float32')
        acc.append(np.sum(pred == label)/len(pred))
        f1.append(f1_score(label, pred, zero_division=0))
        total += len(label)
    f1 = sum(f1) / len(f1)
    loss = sum(loss) / len(loss)
    acc = sum(acc) / len(acc)
    return f1, loss, acc
