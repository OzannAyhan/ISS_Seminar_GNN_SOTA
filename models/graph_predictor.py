# Ali Jaabous
# Graph Predictor

import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_label_index):
        row, col = edge_label_index
        x = torch.cat([x['user'][row], x['movie'][col]], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
