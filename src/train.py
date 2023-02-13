import pandas as pd
import torch
from torch import nn
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

def log_loss_score(predicted, actual, eps=1e-7):
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
    loss_list = []
    with tqdm(total=len(loader.dataset), desc='Training',
              unit='chunks') as prog_bar:
        for i, data in enumerate(loader):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            if isinstance(data[0], list):
                x, label, _ = (data[0][0].to(device), data[0][1].to(device)), data[1].to(device), data[2]
                shape = x[0].size()[0]
            else:
                x, label, _ = data[0].to(device), data[1].to(device), data[2]
                shape = x.size()[0]

            model.zero_grad()

            # Step 3. Run our forward pass.
            pred = model(x)
            label = label.view(-1, model.num_classes).type(torch.FloatTensor).to(device)
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = criterion(pred, label)
            if optimizer:
                loss.backward()
                optimizer.step()
            output = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            loss_list.append(log_loss_score(output.squeeze(), label.squeeze()) / shape)
            prog_bar.set_postfix(**{'run:': 'LSTM', 'lr': 0.0001,
                                    'loss': loss.item()
                                    })
            if isinstance(x, tuple):
                prog_bar.update(x[0].size(0))
            else:
                prog_bar.update(x.size(0))

    loss_list = sum(loss_list) / len(loss_list)
    return model, loss_list


def test(model, loader, visualize=False):
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    f1 = np.zeros(len(loader))
    acc = np.zeros(len(loader))
    total = 0
    loss = np.zeros(len(loader))
    prediction_series = pd.Series(dtype=int)
    with tqdm(total=len(loader.dataset), desc='Validating/Testing',
              unit='chunks') as prog_bar:
        for i, data in enumerate(loader):
            if isinstance(data[0], list):
                x, label, timestamps = (data[0][0].to(device), data[0][1].to(device)), data[1].to(device), data[2]
                shape = x[0].size()[0]
            else:
                x, label, timestamps = data[0].to(device), data[1].to(device), data[2]
                shape = x.size()[0]
            output = model(x).detach().cpu().numpy()
            label = label.view(-1, model.num_classes).detach().cpu().numpy().astype('float32')
            loss[i] = log_loss_score(output.squeeze(), label.squeeze())/shape
            pred = (output > 0.5).astype('float32')
            acc[i] = np.sum(pred == label)/len(pred)
            if visualize:
                prediction_series = pd.concat([prediction_series, pd.Series(np.array(pred == label, dtype=int).squeeze(), index=timestamps.numpy().squeeze())])
            f1[i] = f1_score(label, pred, zero_division=0)
            total += len(label)
            prog_bar.set_postfix(**{'acc': acc[i],
                                    'f1': f1[i],
                                    'loss': loss[i]
                                    })
            if isinstance(x, tuple):
                prog_bar.update(x[0].size(0))
            else:
                prog_bar.update(x.size(0))
    f1 = sum(f1) / len(f1)
    loss = sum(loss) / len(loss)
    acc = sum(acc) / len(acc)
    return f1, loss, acc, prediction_series
