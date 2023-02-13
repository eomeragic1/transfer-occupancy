import argparse
import datetime
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.optim import SGD
from datasets import ROBOD, ECO, HPDMobile
from model import LSTM, GRU, ConvNet, LSTMWithAudio, ROBODNaiveClassifier
from train import train, test, log_loss_score
from torch import manual_seed, load
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    for seed in [1111, 22, 333, 4444, 5555, 66666, 77777, 888888, 999999999]:
        for train_days in [2, 4, 6, 8, 10]:
            # ECO: 190 train days, 15 val days, 31 is ok for n_past
            # ROBOD: 90 train days, 10 val days, 10 for n_past

            parser = argparse.ArgumentParser()
            parser.add_argument('--data_path', type=str, default="../../data/")
            parser.add_argument('--dataset', type=str, default="HPDMobile")
            parser.add_argument('--buildings', nargs='+', default=["Household 01", "Household 03", "Household 06"])
            parser.add_argument('--train_days', type=int, default=train_days)
            parser.add_argument('--n_past', type=int, default=10)
            parser.add_argument('--seed', type=int, default=seed)
            parser.add_argument('--val_days', type=int, default=3)
            parser.add_argument('--hidden_size', type=int, default=32)
            parser.add_argument('--transfer', type=bool, default=False)
            parser.add_argument('--transfer_path', type=str, default='')

            args = parser.parse_args()

            data_path = args.data_path
            dataset_name = args.dataset
            n_past = args.n_past
            train_days = args.train_days
            seed = args.seed
            hidden_size = args.hidden_size
            val_days = args.val_days
            is_transfer = args.transfer
            transfer_path = args.transfer_path
            buildings = args.buildings
            np.random.seed(seed)
            y_name = None
            ## We can use .source_df object for sklearn random forest
            if dataset_name == "ROBOD":
                source_dataset = ROBOD(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                       rooms=buildings, n_past=n_past, train_days=train_days)
                val_dataset = ROBOD(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                    rooms=buildings, n_past=n_past, train_days=train_days,
                                    val_days=val_days, is_val=True)
                test_dataset = ROBOD(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                     rooms=buildings, n_past=n_past, train_days=train_days,
                                     val_days=val_days, is_test=True)
                input_size = 27
                y_name = 'occupant_presence [binary]'
            elif dataset_name == "ECO":
                source_dataset = ECO(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                     residencies=buildings, n_past=n_past,
                                     train_days=train_days)
                val_dataset = ECO(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                  residencies=buildings, n_past=n_past,
                                  train_days=train_days, val_days=val_days, is_val=True)
                test_dataset = ECO(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                   residencies=buildings, n_past=n_past,
                                   train_days=train_days, val_days=val_days, is_test=True)
                input_size = 38
                y_name = 'value'
            elif dataset_name == 'HPDMobile':
                source_dataset = HPDMobile(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                           households=buildings, n_past=n_past,
                                           train_days_all=train_days, is_train=True)
                val_dataset = HPDMobile(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                        households=buildings, n_past=n_past,
                                        train_days_all=train_days, val_days_all=val_days, is_val=True)
                test_dataset = HPDMobile(data_path=os.path.join(data_path, dataset_name, 'combined_cleaned.csv'),
                                         households=buildings, n_past=n_past,
                                         train_days_all=train_days, val_days_all=val_days, is_test=True)
                input_size = 16
                y_name = 'occupied'
            else:
                source_dataset = None
                val_dataset = None
                test_dataset = None

            X = source_dataset.source_df.loc[:, source_dataset.source_df.columns != y_name].select_dtypes(include='number')
            timestamp_train = source_dataset.source_df.loc[:, source_dataset.source_df.columns == 'Timestamp']
            y = source_dataset.source_df.loc[:, source_dataset.source_df.columns == y_name]
            weights = {0: 1-y.value_counts()[0.0]/len(y), 1: 1-y.value_counts()[1.0]/len(y)}
            X = shuffle(X)
            net = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=seed, class_weight=weights)
            net.fit(X, y.values.squeeze())
            pred_train = net.predict(X)
            pred_series_train = pd.Series(np.array(pred_train == y.values.squeeze(), dtype=int), index=timestamp_train.values.squeeze())

            X_test = test_dataset.source_df.loc[:, test_dataset.source_df.columns != y_name].select_dtypes(include='number')
            timestamp_test = test_dataset.source_df.loc[:, test_dataset.source_df.columns == 'Timestamp']
            y_test = test_dataset.source_df.loc[:, test_dataset.source_df.columns == y_name].values.squeeze()
            pred_proba = net.predict_proba(X_test)[:, 1]
            loss = log_loss_score(pred_proba.squeeze(), y_test.squeeze()) / len(y)
            pred = (pred_proba > 0.5).astype('float32')
            pred_series_test = pd.Series(np.array(y_test == pred, dtype=int), index=timestamp_test.values.squeeze())
            pred_series_train.index = pd.to_datetime(pred_series_train.index)
            grouped_train = pred_series_train.groupby(pd.Grouper(freq='D')).agg(np.mean)
            pred_series_test.index = pd.to_datetime(pred_series_test.index)
            grouped_test = pred_series_test.groupby(pd.Grouper(freq='D')).agg(np.mean)
            acc = np.sum(pred == y_test) / len(pred)
            f1 = f1_score(y_test, pred, zero_division=0)
            # fig3, ax3 = plt.subplots()
            # ax3.plot(grouped_train, 'ro-', label='Training set')
            # ax3.plot(grouped_test, 'bo-', label='Test set')
            # ax3.grid(0.4)
            # ax3.set_xlabel('Time')
            # ax3.set_ylabel('Accuracy')
            # ax3.set_title('Train/Test set prediction accuracy through time')
            # ax3.legend()
            # ax3.set_xticks(ax3.get_xticks())
            # ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
            # ax3.set_ylim(0, 1)
 #           plt.show()
            print(f'Random forest classifier: Test loss: {loss}, Accuracy: {acc}, F1 Score: {f1}')
            with open('../results/results_HPDMobile.csv', 'a') as f:
                f.write(f'ROBOD,RF,,,,,,,{acc},{loss},,,{f1},,,{train_days},2,RF\n')
