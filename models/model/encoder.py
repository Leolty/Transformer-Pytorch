'''
@author: Tianyang Liu
@date: 2023-02-04
@homepage: leolty.github.io
'''

from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embeddings.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):

    def __init__(self, vocab_size, d_model, max_len, n_layers, n_head, n_hidden, dropout, device):
        """
        @param vocab_size: vocabulary size
        @param d_model: embedding size
        @param max_len: maximum length of input sequence
        @param n_layers: number of EncoderLayers
        @param n_head: number of heads
        @param n_hidden: number of hidden size
        @param dropout: dropout rate
        """
        super().__init__()
        self.embedding = TransformerEmbedding(d_model=d_model, 
                                                vocab_size=vocab_size,
                                                max_len=max_len,
                                                dropout=dropout,
                                                device=device)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                    n_head=n_head,
                                                    n_hidden=n_hidden,
                                                    dropout=dropout) for _ in range(n_layers)])
        
    def forward(self, x, mask=None):
        """
        @param x: [batch_size, seq_len]
        @param mask: [batch_size, seq_len, seq_len]
        """

        # 1. embedding
        x = self.embedding(x)

        # 2. encoder layers
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        return x
