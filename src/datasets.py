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
        self.is_test = is_test
        self.is_val = is_val
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
            train_tail = self.source_df.loc[self.source_df['timestamp'] < timestamp_train_end, :].tail(n_past)
            self.source_df = pd.concat([train_tail, source_df])

        #Transfer training with validation set - test set
        if train_days and is_test and val_days:
            timestamp_start = self.source_df.iloc[0, :]['timestamp']
            timestamp_val_end = timestamp_start + datetime.timedelta(days=train_days+val_days)
            source_df = self.source_df.loc[self.source_df['timestamp'] >= timestamp_val_end, :]
            val_tail = self.source_df.loc[self.source_df['timestamp'] < timestamp_val_end, :].tail(n_past)
            self.source_df = pd.concat([val_tail, source_df])

        self.room_df = []
        self.negative_idx = np.array([])
        self.num_samples_per_room = []
        for i, room in enumerate(self.source_rooms):
            room_df = self.source_df.loc[self.source_df['Room'] == room, :]
            self.num_samples_per_room.append(len(room_df) - n_past)
            self.room_df.append(room_df)
            truncated_df = room_df.iloc[n_past:, :]
            indices = np.where(truncated_df['occupant_presence [binary]'] == 0)[0]
            if i != 0:
                indices = indices + self.num_samples_per_room[i-1]
            self.negative_idx = np.append(self.negative_idx, indices)
        self.room_indices = np.cumsum(self.num_samples_per_room)

        num_positive_indices = len(self.source_df) - len(self.source_rooms)*self.n_past - len(self.negative_idx)
        self.num_upsamples = num_positive_indices - len(self.negative_idx)

    def __len__(self):
        if not self.is_val and not self.is_test and self.num_upsamples > 0:
            return len(self.source_df) - len(self.source_rooms)*self.n_past + self.num_upsamples
        else:
            return len(self.source_df) - len(self.source_rooms)*self.n_past

    def __getitem__(self, idx):
        if idx >= len(self.source_df) - len(self.source_rooms)*self.n_past:
            idx = int(np.random.choice(self.negative_idx, 1)[0])
        room_index = np.searchsorted(self.room_indices, idx, side='right')
        new_room_indices = np.insert(self.room_indices, 0, 0)
        item_df = self.room_df[room_index].iloc[idx-new_room_indices[room_index]:idx-new_room_indices[room_index]+self.n_past+1]
        y_label = torch.tensor(int(item_df.iloc[-1, :]['occupant_presence [binary]']))
        row_for_prediction = item_df.iloc[-1, :]
        return torch.from_numpy(item_df.drop(['occupant_presence [binary]'], axis=1).select_dtypes(include='number').values).float(), y_label, torch.tensor(row_for_prediction['timestamp'].value, dtype=torch.int64)

    @staticmethod
    def convert_timestamp_to_datetime(timestamp_string):
        return datetime.datetime.strptime(timestamp_string, "%Y-%m-%d %H:%M:%S")


class ECO(Dataset):
    def __init__(self, data_path: str, residencies: List[str], n_past: int, train_days: int = None, is_test: bool = False,
                 is_val: bool = False, val_days: int = None):
        super().__init__()
        self.is_val = is_val
        self.is_test = is_test
        self.data_path = data_path
        self.n_past = n_past
        self.source_residencies = residencies
        full_df = pd.read_csv(self.data_path)
        self.source_df = full_df.loc[full_df['Residency'].isin(residencies), :].drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1).copy()
        self.source_df['Timestamp'] = self.source_df['Timestamp'].apply(ROBOD.convert_timestamp_to_datetime)


        ## Source training with validation set - train set
        if not train_days and val_days and not (is_val or is_test):
            timestamp_end = self.source_df.iloc[-1, :]['Timestamp']
            timestamp_val_start = timestamp_end - datetime.timedelta(days=val_days)
            self.source_df = self.source_df.loc[self.source_df['Timestamp'] < timestamp_val_start, :]

        ## Source training with validation set - validation set/test set
        if not train_days and val_days and (is_val or is_test):
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
            train_tail = self.source_df.loc[self.source_df['Timestamp'] < timestamp_train_end, :].tail(n_past)
            self.source_df = pd.concat([train_tail, source_df])

        # Transfer training with validation set - test set
        if train_days and is_test and val_days:
            timestamp_start = self.source_df.iloc[0, :]['Timestamp']
            timestamp_val_end = timestamp_start + datetime.timedelta(days=train_days+val_days)
            source_df = self.source_df.loc[self.source_df['Timestamp'] >= timestamp_val_end, :]
            val_tail = self.source_df.loc[self.source_df['Timestamp'] < timestamp_val_end, :].tail(n_past)
            self.source_df = pd.concat([val_tail, source_df])

        self.residencies_df = []
        self.positive_idx = np.array([], dtype=int)
        self.num_samples_per_room = []
        for i, residency in enumerate(self.source_residencies):
            residency_df = self.source_df.loc[self.source_df['Residency'] == residency, :]
            self.num_samples_per_room.append(len(residency_df) - n_past)
            self.residencies_df.append(residency_df)
            truncated_df = residency_df.iloc[n_past:, :]
            indices = np.where(truncated_df['value'] == 1)[0]
            if i != 0:
                indices = indices + np.sum(self.num_samples_per_room[:i])
            self.positive_idx = np.append(self.positive_idx, indices)
        self.residency_indices = np.cumsum(self.num_samples_per_room)
        num_negative_indices = len(self.source_df) - len(self.source_residencies) * self.n_past - len(self.positive_idx)
        self.num_downsamples = len(self.positive_idx) - num_negative_indices
        self.negative_idx = np.array(list(set(range(len(self.source_df) - len(self.source_residencies) * self.n_past)) - set(self.positive_idx)), dtype=int)
        if self.num_downsamples > 0 and not is_val and not is_test:
            self.positive_idx = np.random.choice(self.positive_idx, len(self.positive_idx)-self.num_downsamples, replace=False).astype(int)

    def __len__(self):
        if not self.is_val and not self.is_test and self.num_downsamples > 0:
            return len(self.source_df) - len(self.source_residencies) * self.n_past - self.num_downsamples
        else:
            return len(self.source_df) - len(self.source_residencies) * self.n_past

    def __getitem__(self, idx):
        if idx >= len(self.negative_idx):
            idx = self.positive_idx[idx-len(self.negative_idx)]
        else:
            idx = self.negative_idx[idx]
        residency_index = np.searchsorted(self.residency_indices, idx, side='right')
        new_room_indices = np.insert(self.residency_indices, 0, 0)
        item_df = self.residencies_df[residency_index].iloc[
                  idx - new_room_indices[residency_index]:idx - new_room_indices[residency_index] + self.n_past + 1]
        row_for_prediction = item_df.iloc[-1, :]
        y_label = torch.tensor(int(item_df.iloc[-1, :]['value']))
        return torch.from_numpy(item_df.drop(['value', 'Occupancy15Min', 'Occupancy30Min'], axis=1).select_dtypes(
            include='number').values).float(), y_label, torch.tensor(row_for_prediction['Timestamp'].value, dtype=torch.int64)


class HPDMobile(Dataset):
    def __init__(self, data_path: str, households: List[str], n_past: int, train_days_all: int = None, is_test: bool = False,
                 is_val: bool = False, val_days_all: int = None, num_mels: int = 64, spec_second_dim: int = 10, is_train = False):
        super().__init__()
        self.is_val = is_val
        self.is_test = is_test
        self.data_path = data_path
        self.n_past = n_past
        self.source_households = households
        full_df = pd.read_csv(self.data_path)
        self.source_df = full_df.loc[full_df['Household'].isin(households), :].copy()
        self.source_df['Timestamp'] = self.source_df['Timestamp'].apply(ROBOD.convert_timestamp_to_datetime)
        self.num_mels = num_mels
        self.spec_second_dim = spec_second_dim

        self.residencies_df = []
        self.negative_idx = np.array([])
        self.num_samples_per_room = []
        for i, household in enumerate(self.source_households):
            train_days = train_days_all
            val_days = val_days_all
            residency_df = self.source_df.loc[self.source_df['Household'] == household, :]
            max_days = residency_df['DaysSinceStart'].max()
            if not train_days and val_days and not (is_val or is_test):
                if isinstance(val_days, float):
                    train_days = int(max_days-val_days*max_days)
                else:
                    train_days = max_days-val_days
                residency_df = residency_df.loc[residency_df['DaysSinceStart'] < train_days]
            if not train_days and val_days and (is_val or is_test):
                if isinstance(val_days, float):
                    train_days = int(max_days-val_days*max_days)
                else:
                    train_days = max_days-val_days
                first_part = residency_df.loc[self.source_df['DaysSinceStart'] >= train_days, :]
                second_part = residency_df.loc[self.source_df['DaysSinceStart'] < train_days, :]
                residency_df = pd.concat([second_part.tail(n_past), first_part])
            if train_days_all and not is_test and not is_val and is_train:
                if isinstance(train_days_all, float):
                    train_days = int(train_days*max_days)
                residency_df = residency_df.loc[residency_df['DaysSinceStart'] < train_days, :]
            if train_days_all and not is_test and is_val and val_days:
                if isinstance(train_days_all, float):
                    train_days = int(train_days*max_days)
                if isinstance(val_days_all, float):
                    val_days = int(val_days*max_days)
                first_part = residency_df.loc[(self.source_df['DaysSinceStart'] >= train_days) & (
                        self.source_df['DaysSinceStart'] < train_days+val_days), :]
                second_part = residency_df.loc[self.source_df['DaysSinceStart'] < train_days, :].tail(n_past)
                residency_df = pd.concat([second_part.tail(n_past), first_part])
            if train_days_all and is_test and val_days:
                if isinstance(train_days_all, float):
                    train_days = int(train_days*max_days)
                if isinstance(val_days, float):
                    val_days = int(val_days*max_days)
                first_part = residency_df.loc[self.source_df['DaysSinceStart'] >= train_days+val_days, :]
                second_part = residency_df.loc[self.source_df['DaysSinceStart'] < train_days+val_days, :].tail(n_past)
                residency_df = pd.concat([second_part.tail(n_past), first_part])

#            self.num_samples_per_room.append(int((len(residency_df)/5)-n_past))
            self.residencies_df.append(self.transform_household_df(residency_df))
            self.num_samples_per_room.append(int(len(self.residencies_df[i])/5)-n_past)
            truncated_df = self.residencies_df[i].iloc[n_past*5:, :]
            indices = np.where(truncated_df['occupied'] == 0)[0][::5] / 5
            if i != 0:
                indices = indices + np.sum(self.num_samples_per_room[:i])
            self.negative_idx = np.append(self.negative_idx, indices)
        self.residency_indices = np.cumsum(self.num_samples_per_room)
        self.source_df = pd.concat(self.residencies_df)
        num_positive_indices = self.residency_indices[-1] - len(self.source_households) * self.n_past - len(self.negative_idx)
        self.num_upsamples = num_positive_indices - len(self.negative_idx)

    def __len__(self):
        if not self.is_val and not self.is_test and self.num_upsamples > 0:
            return self.residency_indices[-1] + self.num_upsamples
        else:
            return self.residency_indices[-1]

    def get_audio_array(self, date_time, residency, residency_shortened):
        path_to_base_folder, _ = os.path.split(self.data_path)
        path_to_residency_audio = os.path.join(path_to_base_folder, residency, f'{residency_shortened}_AUDIO')
        folder_names = os.listdir(path_to_residency_audio)
        date_folder_name = f'{date_time.year}-{date_time.month:02d}-{date_time.day:02d}'
        hour_folder_name = f'{date_time.hour:02d}{date_time.minute:02d}'
        audio_array = np.zeros(shape=(5, self.num_mels, self.spec_second_dim))
        for folder in folder_names:
            room_number = int(folder.split('_')[1][2])
            try:
                path_to_audio_file = os.path.join(path_to_residency_audio, folder, date_folder_name, hour_folder_name)
                audio_file_name = [file for file in os.listdir(path_to_audio_file) if file[-4:] == '.npy'][0]
                with open(os.path.join(path_to_audio_file, audio_file_name), 'rb') as f:
                    audio = np.load(file=f)
                    audio_array[room_number-1] = audio
            except FileNotFoundError:
                audio = np.zeros(shape=(self.num_mels, self.spec_second_dim))
                audio_array[room_number-1] = audio

        return audio_array

    def __getitem__(self, idx):
        if idx >= self.residency_indices[-1]:
            idx = int(np.random.choice(self.negative_idx, 1)[0])
        residency_index = np.searchsorted(self.residency_indices, idx, side='right')
        new_room_indices = np.insert(self.residency_indices, 0, 0)
        item_df = self.residencies_df[residency_index].iloc[
            (idx - new_room_indices[residency_index])*5:(idx - new_room_indices[residency_index])*5 + 5*self.n_past + 5].copy()
        y_labels = item_df.tail(5)['occupied'].unique()
        if 1 in y_labels:
            y_label = 1
        else:
            y_label = 0
        env_array = torch.from_numpy(item_df.drop(['occupied', 'Occupancy15Min', 'Occupancy30Min', 'DaysSinceStart', 'Unnamed: 0'], axis=1).select_dtypes(
            include='number').values).float()
        env_array = torch.transpose(env_array.reshape(int(env_array.shape[0]/5), 5, env_array.shape[1]), 0, 1)
        row_for_prediction = item_df.iloc[-1, :]
        audio_array = self.get_audio_array(row_for_prediction['Timestamp'], f'{self.source_households[residency_index]}', f'H{self.source_households[residency_index][-1]}')
        return (env_array, torch.from_numpy(audio_array).type(torch.FloatTensor)), y_label, torch.tensor(row_for_prediction['Timestamp'].value, dtype=torch.int64)

    def transform_household_df(self, df):
        timestamps = df['Timestamp'].unique()
        sensors = ['RS1', 'RS2', 'RS3', 'RS4', 'RS5']
        result = df.loc[:, ~df.columns.isin(['Household', 'home'])]
        column_order = result.columns
        result = result.pivot_table(index=['Timestamp', 'hub'])
        result = result.reindex(column_order.drop(['Timestamp', 'hub']), axis=1)
        result = result.reindex(pd.MultiIndex.from_product([timestamps, sensors]), fill_value=0)
        result = result.reset_index()
        result = result.fillna(0)
        result = result.rename(columns={'level_0': 'Timestamp', 'level_1': 'hub'})
        for i in range(int(len(result)/5)):
            small_df = result.iloc[5*i:5*i+5, :]
            y_labels = small_df['occupied'].unique()
            if 1 in y_labels:
                result.iloc[5*i:5*i+5, 10] = 1
            else:
                result.iloc[5*i:5*i+5, 10] = 0
        return result


if __name__ == '__main__':
    dataset = ECO(data_path='../../data/ECO/combined_cleaned.csv', residencies=['Residency 01', 'Residency 02', 'Residency 03'], n_past=10, val_days=10, is_val=True)
    item = dataset.__getitem__(1300)
    print(item[0][1].shape)
