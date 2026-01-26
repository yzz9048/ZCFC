import torch
import torch.nn as nn
from torch.autograd import Function

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

    def forward(self, x):
        
        fault_logits = self.fault(x)
        class_logits = self.classifier(x)
        return fault_logits, class_logits
