from torch.utils.data import Dataset
import h5py
import numpy as np


class MicroseismDataset(Dataset):

    def __init__(self, input_batch, output_batch, selected_sensors_names, number_sources_points, sensors_names, channels, distances):
        self.input_batch = h5py.File(input_batch,'r')
        self.output_batch = h5py.File(output_batch,'r')
        self.selected_sensors_names = selected_sensors_names
        self.number_sources_points = number_sources_points 
        self.sensors_names = sensors_names
        self.channels = channels
        self.distances = distances
        self.number_tensor_componenets = 6  # ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
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
                for comp in range(self.number_tensor_componenets):
                    self.data.append([sen, pp, comp])
        return self.data

    def __getitem__(self, idx):
        sen, pp, comp = self.data[idx]
        input = np.expand_dims(self.input_batch['Channels'][sen]['data'][pp, comp, :], axis=0)
        output = np.expand_dims(self.output_batch['Channels'][sen]['data'][pp, comp, :], axis=0)

        dist_cl = 'dist_to_p' + str(pp+1)
        sen_df = sen.split("_")[0]
        distance = float(self.distances.loc[self.distances['sensor_name']==sen_df, dist_cl].values[0])

        return input, output, distance
