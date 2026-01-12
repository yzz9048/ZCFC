import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

class RCADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        item = dict()
        for key in self.data.keys():
            if key == 'ent_edge_index':
                item[key] = torch.LongTensor(self.data[key][idx])
            else:
                item[key] = torch.FloatTensor(self.data[key][idx].astype(np.float64))
        return item
    
class RCADataLoader():
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.meta_data = dict()
        self.data_loader = defaultdict(dict)  # 修复：使用defaultdict
    
    @staticmethod
    def l2_normalize(x, axis, eps=1e-8):
        # 计算指定维度上的 L2 范数
        norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
        
        # 防止除零
        norm = np.maximum(norm, eps)
        
        # 归一化
        normalized_x = x / norm
        
        return normalized_x
    def load_data(self, data_path_a, data_path_b):
        def process_data_file(data_path, train_key=None):
            with open(data_path, 'rb') as f:
                temp = pickle.load(f)
            
            # Update meta_data with the last loaded metadata
            self.meta_data[train_key] = temp['meta_data']
            data = dict()  # 修复：使用普通dict
            
            for dataset_type in ['train', 'test','z-score']:
                # 修复：先初始化内层字典
                if dataset_type not in data:
                    data[dataset_type] = dict()
                data[dataset_type][train_key] = dict()
                
                # Process modal types
                for modal_type in ['metric', 'trace']:
                    data[dataset_type][train_key][f'x_{modal_type}'] = self.l2_normalize(temp['data'][f'x_{modal_type}_{dataset_type}'].transpose((0, 2, 1)),axis=2)
                
                # Load other data components
                data[dataset_type][train_key]['x_log'] = temp['data'][f'x_log_{dataset_type}']
                data[dataset_type][train_key]['ent_edge_index'] = temp['data'][f'ent_edge_index_{dataset_type}']
                data[dataset_type][train_key]['y'] = temp['data'][f'y_{dataset_type}']
                
                # Create DataLoader
                shuffle = (dataset_type == 'train')
                dataloader = DataLoader(
                    RCADataset(data[dataset_type][train_key]),
                    batch_size=self.param_dict['batch_size'],
                    shuffle=shuffle
                )
                
                # Assign to appropriate loader key
                loader_key = f"{dataset_type}_{train_key}"
                self.data_loader[loader_key] = dataloader
        
        # Process both data files
        process_data_file(data_path_a, 'A')
        process_data_file(data_path_b, 'B')