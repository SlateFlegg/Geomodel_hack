from torch.utils.data import Dataset
import h5py
import numpy as np
import torch

class FTHCoordsDataset(Dataset):
    def __init__(self, input_batch, output_batch, selected_sensors_names, 
                 number_sources_points, sensors_names, channels, distances_df):

        self.input_batch = h5py.File(input_batch, 'r')
        self.output_batch = h5py.File(output_batch, 'r')
        self.selected_sensors_names = selected_sensors_names
        self.number_sources_points = number_sources_points 
        self.sensors_names = sensors_names
        self.channels = channels
        self.distances_df = distances_df
        self.number_tensor_components = 6  # ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        self.data = self.create_dataset()

    def __len__(self):
        return len(self.data)

    def create_dataset(self):
        self.sens = []
        for ch in self.channels:
            self.sens.extend([s + f'_{ch}' for s in self.selected_sensors_names])

        self.data = []
        for sen in self.sens:
            for pp in self.number_sources_points:
                for comp in range(self.number_tensor_components):
                    self.data.append((sen, pp, comp))
        return self.data

    def __getitem__(self, idx):
        sen, pp, comp = self.data[idx]
        
        input = np.expand_dims(self.input_batch['Channels'][sen]['data'][pp, comp, :], axis=0)
        output = np.expand_dims(self.output_batch['Channels'][sen]['data'][pp, comp, :], axis=0)
        
        sensor_name = sen.split('_')[0]
        
        dist_col = f'dist_to_p{pp+1}'
        sensor_data = self.distances_df.loc[self.distances_df['sensor_name'] == sensor_name]
        
        spatial_features = np.array([
            sensor_data['x'].values[0],
            sensor_data['y'].values[0],
            sensor_data['z'].values[0],
            sensor_data[dist_col].values[0]
        ], dtype=np.float32)
        
        input_tensor = torch.from_numpy(input).float()
        output_tensor = torch.from_numpy(output).float()
        spatial_tensor = torch.from_numpy(spatial_features).float()
        
        return input_tensor, output_tensor, spatial_tensor

    def normalize_spatial_features(self):
        for coord in ['x', 'y', 'z']:
            mean = self.distances_df[coord].mean()
            std = self.distances_df[coord].std()
            self.distances_df[coord] = (self.distances_df[coord] - mean) / std
        
        dist_cols = [f'dist_to_p{pp+1}' for pp in self.number_sources_points]
        overall_mean = self.distances_df[dist_cols].values.mean()
        overall_std = self.distances_df[dist_cols].values.std()
        for col in dist_cols:
            self.distances_df[col] = (self.distances_df[col] - overall_mean) / overall_std