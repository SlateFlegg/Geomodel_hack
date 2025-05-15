from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
import networkx as nx
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch.nn as nn

class DistanceAwareGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.distance_processor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        edge_weight = torch.sigmoid(self.distance_processor(edge_attr))
    
        x = F.relu(self.conv1(x, edge_index, edge_weight.squeeze()))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight.squeeze())
        
        return self.out(x)
    
    def message(self, x_j, edge_weight):
        return edge_weight * x_j 
    


class MicroseismGraphDataset(Dataset):
    def __init__(self, input_batch, output_batch, selected_sensors_names, 
                 number_sources_points, sensors_names, channels, distances,
                 connection_radius=100.0): 
        super().__init__()
        self.input_batch = h5py.File(input_batch, 'r')
        self.output_batch = h5py.File(output_batch, 'r')
        self.selected_sensors_names = selected_sensors_names
        self.number_sources_points = number_sources_points
        self.sensors_names = sensors_names
        self.channels = channels
        self.distances = distances
        self.number_tensor_components = 6
        self.connection_radius = connection_radius
        
        self.sensor_graph = self._build_sensor_graph()
        self.data = self.create_dataset()

    def _build_sensor_graph(self):
        G = nx.Graph()
        
        for sensor in self.selected_sensors_names:
            coords = self.distances.loc[self.distances['sensor_name'] == sensor,
                                     ['x', 'y', 'z']].values[0]
            G.add_node(sensor, pos=coords)
        
        positions = nx.get_node_attributes(G, 'pos')
        for i, (n1, p1) in enumerate(positions.items()):
            for n2, p2 in list(positions.items())[i+1:]:
                dist = np.linalg.norm(p1.astype(float) - p2.astype(float))
                if dist <= self.connection_radius:
                    G.add_edge(n1, n2, distance=dist)
        
        return G

    def __len__(self):
        return len(self.data)

    def create_dataset(self):
        self.data = []
        for pp in self.number_sources_points:
            for comp in range(self.number_tensor_components):
                self.data.append((pp, comp))
        return self.data

    def __getitem__(self, idx):
        pp, comp = self.data[idx]
        node_features = []
        outputs = []
        sensor_order = []
        
        for ch in self.channels:
            for sen in self.selected_sensors_names:
                sen_key = f"{sen}_{ch}"
                input = self.input_batch['Channels'][sen_key]['data'][pp, comp, :]
                output = self.output_batch['Channels'][sen_key]['data'][pp, comp, :]
                node_features.append(input)
                outputs.append(output)
                sensor_order.append(sen)
        
        node_features = torch.FloatTensor(np.stack(node_features))
        outputs = torch.FloatTensor(np.stack(outputs))
        
        edge_index, edge_attr = self._get_graph_edges(sensor_order, pp)
        
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=outputs,
            source_point=pp,
            component=comp
        )
        
        return graph_data

    def _get_graph_edges(self, sensor_order, pp):
        sensor_to_idx = {name: idx for idx, name in enumerate(sensor_order)}
        
        edges = []
        edge_attrs = []
        for n1, n2, data in self.sensor_graph.edges(data=True):
            if n1 in sensor_to_idx and n2 in sensor_to_idx:
                dist_cl = 'dist_to_p' + str(pp+1)
                d1 = float(self.distances.loc[self.distances['sensor_name']==n1, dist_cl].values[0])
                d2 = float(self.distances.loc[self.distances['sensor_name']==n2, dist_cl].values[0])
                effective_dist = (d1 + d2)/2  
                
                edges.append([sensor_to_idx[n1], sensor_to_idx[n2]])
                edge_attrs.append([effective_dist])
        
        if not edges:
            num_nodes = len(sensor_order)
            edges = [[i,i] for i in range(num_nodes)]
            edge_attrs = [[0.0] for _ in range(num_nodes)]
        
        edge_index = torch.LongTensor(edges).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attrs)
        
        return edge_index, edge_attr