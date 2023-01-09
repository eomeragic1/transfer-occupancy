from typing import List
import datetime
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os


class ROBOD(Dataset):
    def __init__(self, data_path: str, rooms: List[str], n_past: int, train_days: int = None, is_test: bool = False,
                 is_val: bool = False, val_days: int = None):
        super().__init__()
        self.data_path = data_path
        self.n_past = n_past
        self.source_rooms = rooms
        full_df = pd.read_csv(self.data_path)
        self.source_df = full_df.loc[full_df['Room'].isin(rooms), :].copy()
        self.source_df['timestamp'] = self.source_df['timestamp'].apply(self.convert_timestamp_to_datetime)

        ## Source training with validation set - train set
        if not train_days and val_days and not is_val:
            timestamp_end = self.source_df.iloc[-1, :]['timestamp']
            timestamp_val_start = timestamp_end - datetime.timedelta(days=val_days)
            self.source_df = self.source_df.loc[self.source_df['timestamp'] < timestamp_val_start, :]

        ## Source training with validation set - validation set
        if not train_days and val_days and is_val:
            timestamp_end = self.source_df.iloc[-1, :]['timestamp']
            timestamp_val_start = timestamp_end - datetime.timedelta(days=val_days)
            source_df = self.source_df.loc[self.source_df['timestamp'] >= timestamp_val_start, :]
            rest = self.source_df.loc[self.source_df['timestamp'] < timestamp_val_start, :]
            self.source_df = pd.concat([rest.tail(n_past), source_df])

        ## Transfer training with validation set - train set
        if train_days and not is_test and not is_val and not val_days:
            timestamp_start = self.source_df.iloc[0, :]['timestamp']
            timestamp_end = timestamp_start + datetime.timedelta(days=train_days)
            self.source_df = self.source_df.loc[self.source_df['timestamp'] < timestamp_end, :]
        ## Transfer training with validation set - validation set
        if train_days and not is_test and is_val and val_days:
            timestamp_start = self.source_df.iloc[0, :]['timestamp']
            timestamp_train_end = timestamp_start + datetime.timedelta(days=train_days)
            timestamp_val_end = timestamp_train_end+datetime.timedelta(days=val_days)
            source_df = self.source_df.loc[(self.source_df['timestamp'] >= timestamp_train_end) & (self.source_df['timestamp'] < timestamp_val_end), :]
            train_tail = self.source_df.loc[self.source_df['timestamp'] >= timestamp_train_end, :].tail(n_past)
            self.source_df = pd.concat([train_tail, source_df])

        #Transfer training with validation set - test set
        if train_days and is_test and val_days:
            timestamp_start = self.source_df.iloc[0, :]['timestamp']
            timestamp_val_end = timestamp_start + datetime.timedelta(days=train_days+val_days)
            source_df = self.source_df.loc[self.source_df['timestamp'] >= timestamp_val_end, :]
            val_tail = self.source_df.loc[self.source_df['timestamp'] < timestamp_val_end, :].tail(n_past)
            self.source_df = pd.concat([val_tail, source_df])


        self.num_samples_per_room = []
        for room in self.source_rooms:
            self.num_samples_per_room.append(len(self.source_df[self.source_df['Room'] == room]) - n_past)
        self.room_indices = np.cumsum(self.num_samples_per_room)
        self.room_df = []
        for room in self.source_rooms:
            self.room_df.append(self.source_df.loc[self.source_df['Room'] == room, :])

    def __len__(self):
        return len(self.source_df) - len(self.source_rooms)*self.n_past

    def __getitem__(self, idx):
        room_index = np.searchsorted(self.room_indices, idx, side='right')
        new_room_indices = np.insert(self.room_indices, 0, 0)
        item_df = self.room_df[room_index].iloc[idx-new_room_indices[room_index]:idx-new_room_indices[room_index]+self.n_past+1]
        y_label = torch.tensor(int(item_df.iloc[-1, :]['occupant_presence [binary]']))
        return torch.from_numpy(item_df.drop(['occupant_presence [binary]'], axis=1).select_dtypes(include='number').values).float(), y_label

    @staticmethod
    def convert_timestamp_to_datetime(timestamp_string):
        return datetime.datetime.strptime(timestamp_string, "%Y-%m-%d %H:%M:%S")


class ECO(Dataset):
    def __init__(self, data_path: str, residencies: List[str], n_past: int, train_days: int = None, is_test: bool = False,
                 is_val: bool = False, val_days: int = None):
        super().__init__()
        self.data_path = data_path
        self.n_past = n_past
        self.source_residencies = residencies
        full_df = pd.read_csv(self.data_path)
        self.source_df = full_df.loc[full_df['Residency'].isin(residencies), :].copy()
        self.source_df['Timestamp'] = self.source_df['Timestamp'].apply(ROBOD.convert_timestamp_to_datetime)


        ## Source training with validation set - train set
        if not train_days and val_days and not is_val:
            timestamp_end = self.source_df.iloc[-1, :]['Timestamp']
            timestamp_val_start = timestamp_end - datetime.timedelta(days=val_days)
            self.source_df = self.source_df.loc[self.source_df['Timestamp'] < timestamp_val_start, :]

        ## Source training with validation set - validation set
        if not train_days and val_days and is_val:
            timestamp_end = self.source_df.iloc[-1, :]['Timestamp']
            timestamp_val_start = timestamp_end - datetime.timedelta(days=val_days)
            source_df = self.source_df.loc[self.source_df['Timestamp'] >= timestamp_val_start, :]
            rest = self.source_df.loc[self.source_df['Timestamp'] < timestamp_val_start, :]
            self.source_df = pd.concat([rest.tail(n_past), source_df])

        ## Transfer training with validation set - train set
        if train_days and not is_test and not is_val and not val_days:
            timestamp_start = self.source_df.iloc[0, :]['Timestamp']
            timestamp_end = timestamp_start + datetime.timedelta(days=train_days)
            self.source_df = self.source_df.loc[self.source_df['Timestamp'] < timestamp_end, :]

        ## Transfer training with validation set - validation set
        if train_days and not is_test and is_val and val_days:
            timestamp_start = self.source_df.iloc[0, :]['Timestamp']
            timestamp_train_end = timestamp_start + datetime.timedelta(days=train_days)
            timestamp_val_end = timestamp_train_end+datetime.timedelta(days=val_days)
            source_df = self.source_df.loc[(self.source_df['Timestamp'] >= timestamp_train_end) & (self.source_df['Timestamp'] < timestamp_val_end), :]
            train_tail = self.source_df.loc[self.source_df['Timestamp'] >= timestamp_train_end, :].tail(n_past)
            self.source_df = pd.concat([train_tail, source_df])

        # Transfer training with validation set - test set
        if train_days and is_test and val_days:
            timestamp_start = self.source_df.iloc[0, :]['Timestamp']
            timestamp_val_end = timestamp_start + datetime.timedelta(days=train_days+val_days)
            source_df = self.source_df.loc[self.source_df['Timestamp'] >= timestamp_val_end, :]
            val_tail = self.source_df.loc[self.source_df['Timestamp'] < timestamp_val_end, :].tail(n_past)
            self.source_df = pd.concat([val_tail, source_df])


        self.num_samples_per_room = []
        for residency in self.source_residencies:
            self.num_samples_per_room.append(len(self.source_df[self.source_df['Residency'] == residency]) - n_past)
        self.residency_indices = np.cumsum(self.num_samples_per_room)
        self.residencies_df = []
        for residency in self.source_residencies:
            self.residencies_df.append(self.source_df.loc[self.source_df['Residency'] == residency, :])

    def __len__(self):
        return len(self.source_df) - len(self.source_residencies) * self.n_past

    def __getitem__(self, idx):
        residency_index = np.searchsorted(self.residency_indices, idx, side='right')
        new_room_indices = np.insert(self.residency_indices, 0, 0)
        item_df = self.residencies_df[residency_index].iloc[
                  idx - new_room_indices[residency_index]:idx - new_room_indices[residency_index] + self.n_past + 1]
        y_label = torch.tensor(int(item_df.iloc[-1, :]['value']))
        return torch.from_numpy(item_df.drop(['value'], axis=1).select_dtypes(
            include='number').values).float(), y_label


class HPDMobile(Dataset):
    def __init__(self, data_path: str, households: List[str], n_past: int, train_days: int = None, is_test: bool = False,
                 is_val: bool = False, val_days: int = None, num_mels: int = 64, spec_second_dim: int = 10):
        super().__init__()
        super().__init__()
        self.data_path = data_path
        self.n_past = n_past
        self.source_households = households
        full_df = pd.read_csv(self.data_path)
        self.source_df = full_df.loc[full_df['Household'].isin(households), :].copy()
        self.source_df['Timestamp'] = self.source_df['Timestamp'].apply(ROBOD.convert_timestamp_to_datetime)
        self.num_mels = num_mels
        self.spec_second_dim = spec_second_dim

        ## Source training with validation set - train set
        if not train_days and val_days and not is_val:
            timestamp_end = self.source_df.iloc[-1, :]['Timestamp']
            timestamp_val_start = timestamp_end - datetime.timedelta(days=val_days)
            self.source_df = self.source_df.loc[self.source_df['Timestamp'] < timestamp_val_start, :]

        ## Source training with validation set - validation set
        if not train_days and val_days and is_val:
            timestamp_end = self.source_df.iloc[-1, :]['Timestamp']
            timestamp_val_start = timestamp_end - datetime.timedelta(days=val_days)
            source_df = self.source_df.loc[self.source_df['Timestamp'] >= timestamp_val_start, :]
            rest = self.source_df.loc[self.source_df['Timestamp'] < timestamp_val_start, :]
            self.source_df = pd.concat([rest.tail(n_past), source_df])

        ## Transfer training with validation set - train set
        if train_days and not is_test and not is_val and not val_days:
            timestamp_start = self.source_df.iloc[0, :]['Timestamp']
            timestamp_end = timestamp_start + datetime.timedelta(days=train_days)
            self.source_df = self.source_df.loc[self.source_df['Timestamp'] < timestamp_end, :]

        ## Transfer training with validation set - validation set
        if train_days and not is_test and is_val and val_days:
            timestamp_start = self.source_df.iloc[0, :]['Timestamp']
            timestamp_train_end = timestamp_start + datetime.timedelta(days=train_days)
            timestamp_val_end = timestamp_train_end + datetime.timedelta(days=val_days)
            source_df = self.source_df.loc[(self.source_df['Timestamp'] >= timestamp_train_end) & (
                        self.source_df['Timestamp'] < timestamp_val_end), :]
            train_tail = self.source_df.loc[self.source_df['Timestamp'] >= timestamp_train_end, :].tail(n_past)
            self.source_df = pd.concat([train_tail, source_df])

        # Transfer training with validation set - test set
        if train_days and is_test and val_days:
            timestamp_start = self.source_df.iloc[0, :]['Timestamp']
            timestamp_val_end = timestamp_start + datetime.timedelta(days=train_days + val_days)
            source_df = self.source_df.loc[self.source_df['Timestamp'] >= timestamp_val_end, :]
            val_tail = self.source_df.loc[self.source_df['Timestamp'] < timestamp_val_end, :].tail(n_past)
            self.source_df = pd.concat([val_tail, source_df])

        self.residencies_df = []
        for household in self.source_households:
            self.residencies_df.append(
                self.transform_household_df(self.source_df.loc[self.source_df['Household'] == household, :]))
        self.num_samples_per_room = []
        for residency in self.residencies_df:
            self.num_samples_per_room.append(int((len(residency)/5)-n_past))
        self.residency_indices = np.cumsum(self.num_samples_per_room)

    def __len__(self):
        return self.residency_indices[-1]

    def get_audio_array(self, date_time, residency, residency_shortened):
        path_to_base_folder, _ = os.path.split(self.data_path)
        path_to_residency_audio = os.path.join(path_to_base_folder, residency, f'{residency_shortened}_AUDIO')
        folder_names = os.listdir(path_to_residency_audio)
        date_folder_name = f'{date_time.year}-{date_time.month}-{date_time.day}'
        hour_folder_name = f'{date_time.hour:02d}{date_time.minute:02d}'
        audio_array = np.zeros(shape=(5, self.num_mels, self.spec_second_dim))
        for folder in folder_names:
            room_number = int(folder.split('_')[1][2])
            path_to_audio_file = os.path.join(path_to_residency_audio, folder, date_folder_name, hour_folder_name)
            audio_file_name = [file for file in os.listdir(path_to_audio_file) if file[-4:] == '.npy'][0]
            with open(os.path.join(path_to_audio_file, audio_file_name), 'rb') as f:
                audio = np.load(file=f)
                audio_array[room_number-1] = audio

        return audio_array

    def __getitem__(self, idx):
        residency_index = np.searchsorted(self.residency_indices, idx, side='right')
        new_room_indices = np.insert(self.residency_indices, 0, 0)
        item_df = self.residencies_df[residency_index].iloc[
            (idx - new_room_indices[residency_index])*5:(idx - new_room_indices[residency_index])*5 + 5*self.n_past + 5]
        y_labels = item_df.tail(5)['occupied'].unique()
        if 1 in y_labels:
            y_label = 1
        else:
            y_label = 0
        env_array = torch.from_numpy(item_df.drop(['occupied'], axis=1).select_dtypes(
            include='number').values).float()
        env_array = torch.transpose(env_array.reshape(int(env_array.shape[0]/5), 5, env_array.shape[1]), 0, 1)
        row_for_prediction = item_df.iloc[-1, :]
        audio_array = self.get_audio_array(row_for_prediction['Timestamp'], f'Household 0{residency_index+1}', f'H{residency_index+1}')
        return (env_array, torch.from_numpy(audio_array)), y_label

    def transform_household_df(self, df):
        timestamps = df['Timestamp'].unique()
        sensors = ['RS1', 'RS2', 'RS3', 'RS4', 'RS5']
        result = df.loc[:, ~df.columns.isin(['Household', 'home'])]
        result = result.pivot_table(index=['Timestamp', 'hub'])
        result = result.reindex(pd.MultiIndex.from_product([timestamps, sensors]), fill_value=0)
        result = result.reset_index()
        result = result.fillna(0)
        result = result.rename(columns={'level_0': 'Timestamp', 'level_1': 'hub'})
        return result


if __name__ == '__main__':
    dataset = HPDMobile(data_path='../../data/HPDMobile/combined.csv', households=['Household 01', 'Household 02', 'Household 03'], n_past=9, val_days=7)
    item = dataset.__getitem__(0)
    print(item[0][0].shape)
