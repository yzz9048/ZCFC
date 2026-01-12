from torch.functional import F
import torch_geometric.nn as gnn
import torch.nn as nn

class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super(SAGEEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(gnn.SAGEConv(in_channels, hidden_channels)) # aggr='mean'
        for _ in range(num_layers - 2):
            self.layers.append(gnn.SAGEConv(hidden_channels, hidden_channels))
        self.layers.append(gnn.SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        batch_size = x.shape[0]
        x = x.view(batch_size * x.shape[1], x.shape[2]).contiguous()
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        # e = gnn.global_max_pool(x, edge_index) # 图级池化获得节点特征
        x = F.relu(x)
        x = x.view(batch_size, -1, x.shape[-1]).contiguous()
        return x

class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout):
        super().__init__()
        self.conv1 = gnn.GATv2Conv(in_channels=in_channels,
                                   out_channels=out_channels,
                                   heads=heads,
                                   dropout=dropout,
                                   add_self_loops=False)
        self.conv2 = gnn.GATv2Conv(in_channels=out_channels * heads,
                                   out_channels=int(out_channels / heads),
                                   heads=heads,
                                   dropout=dropout,
                                   add_self_loops=False)
        self.dropout_ratio = dropout

    def forward(self, x, edge_index):
        batch_size = x.shape[0]
        x = x.view(x.shape[0] * x.shape[1], x.shape[2]).contiguous()
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = x.view(batch_size, int(x.shape[0] / batch_size), x.shape[1]).contiguous()
        return x
    
class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super(GINEncoder, self).__init__()
        self.layers = nn.ModuleList()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(gnn.GINConv(nn1))
        for _ in range(num_layers - 2):
            nnk = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(gnn.GINConv(nnk))
        nn_out = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self.layers.append(gnn.GINConv(nn_out))
        self.dropout = dropout

    def forward(self, x, edge_index):
        batch_size = x.shape[0]
        x = x.view(batch_size * x.shape[1], x.shape[2]).contiguous()
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        x = F.relu(x)
        x = x.view(batch_size, -1, x.shape[-1]).contiguous()
        return x
    
class GGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super(GGNNEncoder, self).__init__()
        # 将输入映射到 hidden_channels（确保 in_dim ≤ hidden_dim）
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.ggnn = gnn.GatedGraphConv(out_channels=hidden_channels, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        batch_size = x.shape[0]
        x = x.view(batch_size * x.shape[1], x.shape[2]).contiguous()
        
        # 先升维
        x = self.fc_in(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GGNN传播
        x = self.ggnn(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 输出映射
        x = self.fc_out(x)
        x = F.relu(x)
        x = x.view(batch_size, -1, x.shape[-1]).contiguous()
        return x
    
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super(GCNEncoder, self).__init__()
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.dropout = dropout

        # 构建多层GCN
        for i in range(num_layers):
            self.convs.append(gnn.GCNConv(hidden_channels, hidden_channels))

        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        batch_size = x.shape[0]
        x = x.view(batch_size * x.shape[1], x.shape[2]).contiguous()

        # 输入升维
        x = self.fc_in(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 多层GCN传播
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 输出映射
        x = self.fc_out(x)
        x = F.relu(x)
        x = x.view(batch_size, -1, x.shape[-1]).contiguous()
        return x