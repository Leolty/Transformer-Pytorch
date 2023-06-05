'''
@author: Tianyang Liu
@date: 2023-02-04
@homepage: leolty.github.io
'''

from torch import nn

class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, hidden)
        self.w_2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        @param x: [batch_size, seq_len, d_model]
        """
        out = self.w_1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.w_2(out)
        return out