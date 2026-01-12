import math
import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from model.gnn import *

class BaseBatchGraph:
    def __init__(self, batch_data, meta_data):
        self.meta_data = meta_data
        self.edge_index = ''
        self.batch_size = 0
        self.x = dict()
        self.x_batch = []

    def generate_batch_edge_index(self, edge_index, num_of_nodes):
        edge_index = edge_index.transpose(1, 2).contiguous()
        for i in range(self.batch_size):
            edge_index[i] += i * num_of_nodes
        self.edge_index = edge_index.view(edge_index.shape[0] * edge_index.shape[1], edge_index.shape[2]).t().contiguous()
        self.edge_index = SparseTensor(row=self.edge_index[0], col=self.edge_index[1])

    def generate_x_batch(self, num_of_nodes):
        for i in range(self.batch_size):
            for _ in range(num_of_nodes):
                self.x_batch.append(i)
        self.x_batch = torch.tensor(self.x_batch, dtype=torch.long)

    def to(self, device):
        for key in self.x.keys():
            self.x[key] = self.x[key].to(device)
        self.edge_index = self.edge_index.to(device)
        self.x_batch = self.x_batch.to(device)
        return self


class EntBatchGraph(BaseBatchGraph):
    def __init__(self, batch_data, meta_data):
        super().__init__(batch_data, meta_data)
        self.x['re'] = batch_data['x_ent']
        self.edge_index = batch_data['ent_edge_index']
        self.batch_size = self.edge_index.shape[0]

        num_of_nodes = len(self.meta_data['ent_names'])
        self.generate_x_batch(num_of_nodes)
        self.generate_batch_edge_index(self.edge_index, num_of_nodes)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, in_features, num_of_o11y_features, device):
        super(PositionalEmbedding, self).__init__()

        temp_in_features = in_features
        if temp_in_features % 2 == 1:
            temp_in_features += 1

        temp_num_of_o11y_features = num_of_o11y_features
        if temp_num_of_o11y_features % 2 == 1:
            temp_num_of_o11y_features += 1

        pe = torch.zeros(temp_in_features, temp_num_of_o11y_features).float()
        pe.require_grad = False

        position = torch.arange(0, temp_in_features).float().unsqueeze(1)
        div_term = (torch.arange(0, temp_num_of_o11y_features, 2).float() * -(math.log(10000.0) / temp_num_of_o11y_features)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.transpose(1, 0)[:num_of_o11y_features, :in_features].to(device)

    def forward(self, x):
        return self.pe + x
    
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x: (batch, seq_len, head, hidden)
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)  # (batch, seq_len, head)
        out = (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, head, hidden)
        return out

class Encoder(nn.Module):
    def __init__(self, param_dict, meta_data):
        super(Encoder, self).__init__()
        self.meta_data = meta_data['A'] # for initialization
        self.meta_dataT = meta_data # for forward
        in_dim = param_dict['efi_in_dim']
        self.device = torch.device(f"cuda:{param_dict['gpu']}" if torch.cuda.is_available() else "cpu")
        self.different_modal_mapping_dict = nn.ModuleDict()
        self.positional_embedding_dict = nn.ModuleDict({train_key: nn.ModuleDict() for train_key in meta_data.keys()})
        self.positional_embedding_log = nn.ModuleDict({train_key: nn.ModuleDict() for train_key in meta_data.keys()})
        self.modal_transformer_encoder_layer_dict = nn.ModuleDict()
        self.ent_transformer_encoder_layer_dict = nn.ModuleDict()
        self.ent_feature_align_dict = nn.ModuleDict({ent_type: nn.ModuleDict() for ent_type in self.meta_data['ent_types']})
        
        for modal_type in ['metric','trace']: #self.meta_data['modal_types']
            for train_key in meta_data.keys():
                self.positional_embedding_dict[train_key][modal_type] = PositionalEmbedding(in_features=param_dict['window_size'],
                                                                             num_of_o11y_features=meta_data[train_key]['o11y_length'][modal_type], device=self.device)
                self.positional_embedding_log[train_key] = PositionalEmbedding(in_features=param_dict['window_size'],
                                                                             num_of_o11y_features=param_dict['efi_in_dim']*len(meta_data[train_key]["ent_names"]), device=self.device)
            self.different_modal_mapping_dict[modal_type] = nn.Linear(in_features=param_dict['window_size'],
                                                                      out_features=param_dict['orl_te_in_channels'])
            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=param_dict['orl_te_in_channels'],
                                                                   nhead=param_dict['orl_te_heads'])
            self.modal_transformer_encoder_layer_dict[modal_type] = nn.TransformerEncoder(transformer_encoder_layer, num_layers=param_dict['orl_te_layers'])

        for ent_type in self.meta_data['ent_types']:
            for modal_type in ['metric','trace']:
                self.ent_feature_align_dict[ent_type][modal_type] = nn.Linear(self.meta_data['max_ent_feature_num'][ent_type][modal_type] * in_dim, param_dict['efi_out_dim'])
            

            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=param_dict['efi_te_heads'])
            self.ent_transformer_encoder_layer_dict[ent_type] = nn.TransformerEncoder(transformer_encoder_layer, num_layers=param_dict['efi_te_layers'])
        self.ent_feature_align_dict_log = nn.Sequential(
                                                            nn.Linear(param_dict['llm_embedding'], param_dict['llm_embedding']//2),
                                                            nn.GELU(),
                                                            nn.Linear(param_dict['llm_embedding']//2, param_dict['efi_in_dim'])
                                                        )
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=param_dict['window_size'], nhead=param_dict['efi_te_heads'])
        self.ent_transformer_encoder_layer_log = nn.TransformerEncoder(transformer_encoder_layer, num_layers=param_dict['efi_te_layers'])
        self.linear = nn.Linear(len(self.meta_data['modal_types']) * param_dict['efi_out_dim'], param_dict['efi_out_dim']) 
        self.linear2 = nn.Linear(len(self.meta_data['modal_types']) * param_dict['efi_out_dim'], param_dict['eff_out_dim']) 
        self.linear_log = nn.Linear(param_dict['efi_in_dim'], param_dict['efi_out_dim'])
        self.pool = AttentionPooling(param_dict['efi_in_dim'])

        self.GAT_net = GATNet(in_channels=param_dict['eff_in_dim'],
                              out_channels=param_dict['eff_out_dim'],
                              heads=2,
                              dropout=0.1)
        self.GraphSage = SAGEEncoder(in_channels=param_dict['eff_in_dim'],
                                    hidden_channels=64,
                                    out_channels=param_dict['eff_out_dim'],
                                    num_layers=2,
                                    dropout=0.1)
        self.GIN = GINEncoder(in_channels=param_dict['eff_in_dim'],
                            hidden_channels=64,
                            out_channels=param_dict['eff_out_dim'],
                            num_layers=2,
                            dropout=0.1)
        # self.GGNN = GGNNEncoder(in_channels=param_dict['eff_in_dim'],
        #                      hidden_channels=64,
        #                      out_channels=param_dict['eff_out_dim'],
        #                      num_layers=3,
        #                      dropout=0.1)
        
    def forward(self, batch_data, train_key):
        for modal_type in ['metric','trace']: 
            batch_data[f'x_{modal_type}'] = batch_data[f'x_{modal_type}'].to(self.device)
            batch_data[f'x_{modal_type}'] = self.positional_embedding_dict[train_key[-1]][modal_type](batch_data[f'x_{modal_type}']).contiguous()
            batch_data[f'x_{modal_type}'] = self.different_modal_mapping_dict[modal_type](batch_data[f'x_{modal_type}'])
            batch_data[f'x_{modal_type}'] = self.modal_transformer_encoder_layer_dict[modal_type](batch_data[f'x_{modal_type}'])
        batch_size = batch_data['y'].shape[0]
        batch_data['x_log'] = batch_data['x_log'].to(self.device)
        log_data = self.ent_feature_align_dict_log(batch_data['x_log'])
        log_data = log_data.permute(0, 2, 3, 1)
        BB, FF, DD, WW = log_data.shape
        log_data = log_data.reshape(batch_size, -1, log_data.shape[-1])
        log_data = self.positional_embedding_log[train_key[-1]](log_data)
        log_data = self.ent_transformer_encoder_layer_log(log_data)
        log_data = log_data.reshape(BB, FF, DD, WW)
        log_data = self.pool(log_data.permute(0, 3, 1, 2))
        log_data = self.linear_log(log_data)
        # 需要整合metric trace log特征，映射到一个维度
        
        x_ent = []
        for ent_type in self.meta_dataT[train_key[-1]]['ent_types']: 
            # 单独处理log
            # log_data = self.pool(batch_data['x_log'])
            # log_data = self.ent_feature_align_dict[ent_type]['log'](log_data)
            
            for ent_index in range(self.meta_dataT[train_key[-1]]['ent_type_index'][ent_type][0], self.meta_dataT[train_key[-1]]['ent_type_index'][ent_type][1]): # 遍历每种实体内所有元素
                x = []
                for modal_type in ['metric','trace']:
                    feature_index_pair = self.meta_dataT[train_key[-1]]['ent_features'][modal_type][ent_index][1]
                    modal_data = batch_data[f'x_{modal_type}'][:, feature_index_pair[0]:feature_index_pair[1], :]
                    modal_data = self.ent_transformer_encoder_layer_dict[ent_type](modal_data)
                    modal_data = modal_data.view(batch_size, modal_data.shape[1] * modal_data.shape[2]).contiguous()
                    modal_data = self.ent_feature_align_dict[ent_type][modal_type](modal_data)
                    x.append(modal_data)
                x.append(log_data[:, ent_index,:])
                x = torch.cat(x, dim=1) #将不同模态的数据在特征维度上拼接 
                x_ent.append(x)
        x_ent = torch.stack(x_ent, dim=1)
        
        # # noGNN
        # x_ent = self.linear2(x_ent)
        # x = nn.AdaptiveAvgPool1d(1)(x_ent.transpose(1,2))
        # return x.transpose(1,2).squeeze()

        x_ent = self.linear(x_ent) # 直接将拼接的多模态特征映射到向量空间
        batch_data['x_ent'] = x_ent

        ent_batch_graph = EntBatchGraph(batch_data, self.meta_dataT[train_key[-1]]).to(self.device)
        x = ent_batch_graph.x['re']
        x = self.GraphSage(x, ent_batch_graph.edge_index)

        return torch.mean(x, dim=1) #(B,F, D) --> (B, D)
