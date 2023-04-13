import argparse
import datetime
import os
from pathlib import Path
import matplotlib.dates as mdates
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.optim import SGD
from datasets import ROBOD, ECO, HPDMobile
from model import LSTM, GRU, ConvNet, LSTMWithAudio, ROBODNaiveClassifier, ECONaiveClassifier, HPDMobileNaiveClassifier
from train import train, test
from torch import manual_seed, load
import random
import matplotlib.pyplot as plt
import torch
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # ECO: 190 train days, 15 val days, 31 is ok for n_past
    # ROBOD: 90 train days, 10 val days, 10 for n_past
    # HPDMobile: 15 n_past
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../../data/")
    parser.add_argument('--dataset', type=str, default="ECO")
    parser.add_argument('--model', type=str, default="Naive")
    parser.add_argument('--buildings', nargs='+', default=["Residency 04"])
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--n_past', type=int, default=15)
    parser.add_argument('--train_days', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val_days', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--transfer', type=str2bool, default=False)
    parser.add_argument('--transfer_path', type=str, default='../model_checkpoints/HPDMobile_LSTM_2023-01-31_seed_888888_traindays_None_9189.pt')
    parser.add_argument('--save_model', type=str2bool, default=False)
    parser.add_argument('--visualize', type=str2bool, default=False)
    parser.add_argument('--num_frozen_layers', type=int, default=0)

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
    is_transfer = args.transfer
    transfer_path = args.transfer_path
    visualize = args.visualize
    save_model = args.save_model
    buildings = args.buildings
    num_frozen_layers = args.num_frozen_layers

    input_size = 0

    if dataset_name == "ROBOD":
        source_dataset = ROBOD(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                               rooms=buildings, n_past=n_past, train_days=train_days, val_days=val_days)
        val_dataset = ROBOD(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                            rooms=buildings, n_past=n_past, train_days=train_days,
                            val_days=val_days, is_val=True)
        test_dataset = ROBOD(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                             rooms=buildings, n_past=n_past, train_days=train_days,
                             val_days=val_days, is_test=True)
        input_size = 27
    elif dataset_name == "ECO":
        source_dataset = ECO(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                             residencies=buildings, n_past=n_past,
                             train_days=train_days, val_days=val_days)
        val_dataset = ECO(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                          residencies=buildings, n_past=n_past,
                          train_days=train_days, val_days=val_days, is_val=True)
        test_dataset = ECO(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                           residencies=buildings, n_past=n_past,
                           train_days=train_days, val_days=val_days, is_test=True)
        input_size = 38
    elif dataset_name == 'HPDMobile':
        source_dataset = HPDMobile(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                   households=buildings, n_past=n_past,
                                   train_days_all=train_days, val_days_all=val_days, is_train=True)
        val_dataset = HPDMobile(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                households=buildings, n_past=n_past,
                                train_days_all=train_days, val_days_all=val_days, is_val=True)
        test_dataset = HPDMobile(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                 households=buildings, n_past=n_past,
                                 train_days_all=train_days, val_days_all=val_days, is_test=True)
        input_size = 45

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
        net = GRU(input_dim=input_size, hidden_dim=hidden_size, layer_dim=2, output_dim=1).to('cuda')
    elif model == 'ConvNet':
        net = ConvNet(in_channels=1, hidden_channels=hidden_size, n_past=n_past+1, n_attr=input_size).to('cuda')
    elif model == 'Naive':
        if dataset_name == 'ROBOD':
            net = ROBODNaiveClassifier()
        elif dataset_name == 'ECO':
            net = ECONaiveClassifier()
        elif dataset_name == 'HPDMobile':
            net = HPDMobileNaiveClassifier()
    else:
        net = None

    manual_seed(seed)
    random.seed(seed)
    if len(source_dataset):
        train_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = []
    if len(val_dataset):
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        val_loader = []
    if len(test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        test_loader = []

    print('Debug: Length of test loader: ', len(test_loader))
    if model != 'Naive':
        optimizer = SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    else:
        optimizer = None
        scheduler = None

    min_val_loss = 1e10
    model_test_f1 = []
    model_test_loss = []
    model_val_f1 = []
    best_epoch = 0
    model_val_acc = []
    model_test_acc = []
    model_train_loss = []
    model_val_loss = []

    print(is_transfer)
    print(type(is_transfer))
    if is_transfer:
        checkpoint = load(transfer_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.train()
        cntr = 0
        for child in net.children():
            cntr += 1
            if cntr <= num_frozen_layers:
                for param in child.parameters():
                    param.requires_grad = False

    path = Path(f'../results/hpo_{dataset_name}.csv')
    if not path.is_file():
        with open(path, 'w') as f:
            f.write('Dataset,Model,BatchSize,StartingLr,N_days,HiddenSize,Seed,BestEpoch,TestAcc,TestLoss,ValAcc,ValLoss,TestF1,ValF1,TrainLoss\n')

    model_save_paths = []
    for epoch in range(1, num_epochs+1):
        net, train_loss = train(net, train_loader, optimizer)
        val_f1, val_loss, val_acc, pred_series_val = test(net, val_loader, visualize=visualize)

        model_train_loss.append(train_loss)
#        model_test_loss.append(test_loss)
        model_val_loss.append(val_loss)
        model_val_acc.append(val_acc)
#        model_test_acc.append(test_acc)
        model_val_f1.append(val_f1)
#        model_test_f1.append(test_f1)
        if model != 'Naive':
            scheduler.step(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = epoch
            model_save_paths.append(f'../model_checkpoints/{dataset_name}_{model}_{datetime.datetime.now().date()}_seed_{seed}_traindays_{train_days}_{random.randint(0, 10000)}.pt')
            torch.save({
                'model_state_dict': net.state_dict(),
            }, model_save_paths[-1])

        print(f'Epoch: {epoch}, Dataset: {dataset_name}, Model: {model}, Seed: {seed}, Best epoch: {best_epoch}, '
#              f'Test f1: {test_f1}, Test loss: {test_loss}, Val f1: {val_f1}, Val loss: {val_loss}, Test acc: {test_acc}, Val acc: {val_acc}, Train loss: {train_loss}')
              f'Val f1: {val_f1}, Val loss: {val_loss}, Val acc: {val_acc}, Train loss: {train_loss}')
        if visualize:
            fig, ax = plt.subplots()
            ax.plot(range(1, epoch+1), model_train_loss, 'r-', label='Train loss')
            ax.plot(range(1, epoch+1), model_val_loss, 'b-', label='Validation loss')
#            ax.plot(range(1, epoch+1), model_test_loss, 'k-', label='Test loss')
            ax.grid(0.4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training/Validation/Test set losses during training')
            ax.legend()
            plt.savefig(f'../fig/{dataset_name}_losses_{int(is_transfer)}_{datetime.datetime.now().date()}.png', facecolor='white', bbox_inches='tight')
            plt.show()

            fig2, ax2 = plt.subplots()
            ax2.plot(range(1, epoch+1), model_val_f1, 'b--', label='Validation F1')
#            ax2.plot(range(1, epoch+1), model_test_f1, 'k--', label='Test F1')
            ax2.plot(range(1, epoch+1), model_val_acc, 'b-', label='Validation accuracy')
#            ax2.plot(range(1, epoch+1), model_test_acc, 'k-', label='Test accuracy')
            ax2.grid(0.4)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Metrics')
            ax2.set_title('Validation/Test set metrics during training')
            ax2.legend()
            plt.savefig(f'../fig/{dataset_name}_metrics_transfer_{int(is_transfer)}_{datetime.datetime.now().date()}.png', facecolor='white',
                        bbox_inches='tight')
            plt.show()

            pred_series_val.index = pd.to_datetime(pred_series_val.index, unit='ns')
            grouped_val = pred_series_val.groupby(pd.Grouper(freq='D')).agg(np.mean)

#            pred_series_test.index = pd.to_datetime(pred_series_test.index, unit='ns')
#            grouped_test = pred_series_test.groupby(pd.Grouper(freq='D')).agg(np.mean)

            fig3, ax3 = plt.subplots()
            ax3.plot(grouped_val, 'go-', label='Validation set')
 #           ax3.plot(grouped_test, 'bo-', label='Test set')
            ax3.grid(0.4)
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Validation/Test set prediction accuracy through time')
            ax3.legend()
            ax3.set_xticks(ax3.get_xticks())
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
            ax3.set_ylim(0, 1)
            plt.savefig(
                f'../fig/{dataset_name}_prediction_time_{int(is_transfer)}_{datetime.datetime.now().date()}.png',
                facecolor='white',
                bbox_inches='tight')
            plt.show()

#    checkpoint = load(model_save_paths[-1])
#    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    test_f1, test_loss, test_acc, pred_series_test = test(net, test_loader, visualize=visualize)

    if visualize:
        pred_series_test.index = pd.to_datetime(pred_series_test.index, unit='ns')
        grouped_test = pred_series_test.groupby(pd.Grouper(freq='D')).agg(np.mean).sort_index().dropna()
        x = mdates.date2num(grouped_test.index)
        z = np.polyfit(x, grouped_test.values, 1)
        p = np.poly1d(z)
        fig4, ax4 = plt.subplots()
        ax4.plot(grouped_test, 'bo-', label='Test set average prediction accuracy')
        ax4.plot(grouped_test.index, p(x), 'g-', label='Prediction accuracy trend line')
        ax4.grid(0.4)
        ax4.legend()
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Test set prediction accuracy through time')
        ax4.set_xticks(ax4.get_xticks())
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.set_ylim(0, 1)
        plt.savefig(
            f'../fig/{dataset_name}_prediction_time_{int(is_transfer)}_{datetime.datetime.now().date()}_test.png',
            facecolor='white',
            bbox_inches='tight')
        plt.show()

    print(f'Training finished! Best epoch: {best_epoch}, Test f1: {test_f1}, Test loss: {test_loss}, '
          f'Val f1: {model_val_f1[best_epoch-1] if best_epoch else 0}, Val loss: {min_val_loss}, Test acc: {test_acc}, Val acc: {model_val_acc[best_epoch-1] if best_epoch else 0}, Train loss: {model_train_loss[best_epoch-1] if best_epoch else 0}')
    with open(f'../results/hpo_{dataset_name}.csv', 'a') as f:
        f.write(
            f'{dataset_name},{model},{batch_size},{lr},{n_past},{hidden_size},{seed},{best_epoch},{test_acc:.6f},{test_loss:.4f},{model_val_acc[best_epoch-1] if best_epoch else 0:.6f},{min_val_loss:.4f},{test_f1},{model_val_f1[best_epoch-1] if best_epoch else 0},{model_train_loss[best_epoch-1] if best_epoch else 0},{train_days},{num_frozen_layers},{is_transfer}\n')
    for model_save_path in model_save_paths[:-1]:
        os.remove(model_save_path)
    if not save_model:
        os.remove(model_save_paths[-1])
