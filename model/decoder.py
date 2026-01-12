import torch
import torch.nn as nn
from torch.autograd import Function

class Decoder(nn.Module):
    def __init__(self, param_dict, meta_data):
        super().__init__()
        self.device_marker = nn.Parameter(torch.empty(0))
        self.meta_data = meta_data
        self.linear_dict = nn.ModuleDict()
        for ent_type in self.meta_data['ent_types']:
            index_pair = self.meta_data['ent_fault_type_index'][ent_type]
            self.linear_dict[ent_type] = nn.Linear(param_dict['eff_out_dim'], index_pair[1] - index_pair[0])

    def forward(self, x):
        output = dict()
        for ent_type in self.meta_data['ent_types']:
            temp = x[:, self.meta_data['ent_type_index'][ent_type][0]:self.meta_data['ent_type_index'][ent_type][1], :]
            temp = temp.reshape(temp.shape[0] * temp.shape[1], temp.shape[2]).contiguous()
            output[ent_type] = self.linear_dict[ent_type](temp)
        return output
    
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class ImprovedDecoder(nn.Module):
    def __init__(self, in_dim, num_classes=3):
        super().__init__()
        self.fault = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lambda_grl=1.0):
        
        fault_logits = self.fault(x)
        class_logits = self.classifier(x)
        return fault_logits, class_logits