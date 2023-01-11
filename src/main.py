import argparse
import os
from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.optim import SGD
from src.datasets import ROBOD, ECO, HPDMobile
from src.model import LSTM, GRU, ConvNet, LSTMWithAudio
from src.train import train, test
from torch import manual_seed
import random


if __name__ == '__main__':
    # ECO: 190 train days, 15 val days, 31 is ok for n_past
    # ROBOD: 90 train days, 10 val days, 10 for n_past

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../../data/")
    parser.add_argument('--dataset', type=str, default="ROBOD")
    parser.add_argument('--model', type=str, default="ConvNet")
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--n_past', type=int, default=10)
    parser.add_argument('--train_days', type=int, default=90)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val_days', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=64)

    args = parser.parse_args()

    data_path = args.data_path
    model = args.model
    dataset_name = args.dataset
    num_epochs = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    n_past = args.n_past
    train_days = args.train_days
    seed = args.seed
    hidden_size = args.hidden_size
    val_days = args.val_days
    input_size = 0

    if dataset_name == "ROBOD":
        source_dataset = ROBOD(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                               rooms=['Room 3', 'Room 4', 'Room 5'], n_past=n_past, train_days=train_days)
        val_dataset = ROBOD(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                            rooms=['Room 3', 'Room 4', 'Room 5'], n_past=n_past, train_days=train_days,
                            val_days=val_days, is_val=True)
        test_dataset = ROBOD(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                             rooms=['Room 3', 'Room 4', 'Room 5'], n_past=n_past, train_days=train_days,
                             val_days=val_days, is_test=True)
        input_size = 27
    elif dataset_name == "ECO":
        source_dataset = ECO(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                             residencies=['Residency 01', 'Residency 02', 'Residency 03'], n_past=n_past,
                             train_days=train_days)
        val_dataset = ECO(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                          residencies=['Residency 01', 'Residency 02', 'Residency 03'], n_past=n_past,
                          train_days=train_days, val_days=val_days, is_val=True)
        test_dataset = ECO(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                           residencies=['Residency 01', 'Residency 02', 'Residency 03'], n_past=n_past,
                           train_days=train_days, val_days=val_days, is_test=True)
        input_size = 39
    elif dataset_name == 'HPDMobile':
        source_dataset = HPDMobile(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                   households=['Household 01', 'Household 02', 'Household 03'], n_past=n_past,
                                   train_days=train_days)
        val_dataset = HPDMobile(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                households=['Household 01', 'Household 02', 'Household 03'], n_past=n_past,
                                train_days=train_days, val_days=val_days, is_val=True)
        test_dataset = HPDMobile(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                 households=['Household 01', 'Household 02', 'Household 03'], n_past=n_past,
                                 train_days=train_days, val_days=val_days, is_test=True)
        input_size = 16

    else:
        source_dataset = None
        val_dataset = None
        test_dataset = None

    if model == 'LSTM':
        if dataset_name == 'HPDMobile':
            net = LSTMWithAudio(hidden_size=hidden_size, num_classes=1, input_size=input_size, num_layers=1,
                                seq_length=n_past + 1,
                                bidirectional=True, audio_hidden_channels=32, audio_in_channels=5).to('cuda')
        else:
            net = LSTM(hidden_size=hidden_size, num_classes=1, input_size=input_size, num_layers=1,
                       seq_length=n_past + 1,
                       bidirectional=True).to('cuda')
    elif model == 'GRU':
        net = GRU(input_dim=27, hidden_dim=hidden_size, layer_dim=2, output_dim=1).to('cuda')
    elif model == 'ConvNet':
        net = ConvNet(in_channels=1, hidden_channels=hidden_size, n_past=11, n_attr=27).to('cuda')
    else:
        net = None

    manual_seed(seed)
    random.seed(seed)
    train_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    optimizer = SGD(net.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)

    min_val_loss = 1e10
    model_test_f1 = 0
    model_test_loss = 0
    model_val_f1 = 0
    best_epoch = 0
    model_val_acc = 0
    model_test_acc = 0

    path = Path(f'../results/hpo_{dataset_name}.csv')
    if not path.is_file():
        with open(path, 'w') as f:
            f.write('Dataset,Model,BatchSize,StartingLr,N_days,HiddenSize,Seed,BestEpoch,TestAcc,TestLoss,ValAcc,ValLoss,TestF1,ValF1\n')

    for epoch in range(1, num_epochs):
        net = train(net, train_loader, optimizer)
        val_f1, val_loss, val_acc = test(net, val_loader)
        test_f1, test_loss, test_acc = test(net, test_loader)

        scheduler.step(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model_test_loss = test_loss
            model_test_f1 = test_f1
            model_val_f1 = val_f1
            model_test_acc = test_acc
            model_val_acc = val_acc
            best_epoch = epoch
        print(f'Epoch: {epoch}, Dataset: {dataset_name}, Model: {model}, Seed: {seed}, Best epoch: {best_epoch}, '
              f'Test f1: {test_f1}, Test loss: {test_loss}, Val f1: {val_f1}, Val loss: {val_loss}, Test acc: {test_acc}, Val acc: {val_acc}')

    print(f'Training finished! Best epoch: {best_epoch}, Test f1: {model_test_f1}, Test loss: {model_test_loss}, '
          f'Val f1: {model_val_f1}, Val loss: {min_val_loss}, Test acc: {model_test_acc}, Val acc: {model_val_acc}')
    with open(f'../results/hpo_{dataset_name}.csv', 'a') as f:
        f.write(
            f'{dataset_name},{model},{batch_size},{lr},{n_past},{hidden_size},{seed},{best_epoch},{model_test_acc:.6f},{model_test_loss:.4f},{model_val_acc:.6f},{min_val_loss:.4f}, {model_test_f1}, {model_val_f1}\n')
