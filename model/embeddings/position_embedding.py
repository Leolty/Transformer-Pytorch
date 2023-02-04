'''
@author: Tianyang Liu
@date: 2023-02-04
@homepage: leolty.github.io
'''

import torch
from torch import nn

class PositionEmbedding(nn.Module):
    '''
    Position Embedding (sinusoidal)
    '''

    def __init__(self, d_model, max_len, device):
        """
        @param d_model: dimension of model
        @param max_len: maximum length of input sequence
        @param device: device to store the embedding
        """
        super(PositionEmbedding, self).__init__()

        # make sure same size with the input matrix
        self.encoding = torch.zeros(max_len, d_model, device=device)

        # no need for gradient
        self.encoding.requires_grad = False

        # 1D => 2D 
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)

        # "i" refers to index of d_model
        # "step=2" refers to 'i' is even (multiple with 2)
        _2i = torch.arange(0, d_model, step = 2, device=device).float()

        # compute positional encoding
        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))


    def forward(self, x):
        """
        @param x: input tensor
        """

        # get sequence length (_ is batch size)
        _, seq_len = x.size()

        return self.encoding[:seq_len, :]
