import torch
import torch.nn as nn

from models.blocks.decoder_layer import DecoderLayer
from models.embeddings.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):

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
        
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                    n_head=n_head,
                                                    n_hidden=n_hidden,
                                                    dropout=dropout) for _ in range(n_layers)])
        
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        @param x: [batch_size, seq_len]
        @param encoder_output: [batch_size, seq_len, d_model]
        @param self_mask: [batch_size, seq_len, seq_len]
        @param cross_mask: [batch_size, seq_len, seq_len]
        """

        # 1. embedding
        x = self.embedding(x)

        # 2. decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask=self_mask, cross_mask=cross_mask)

        # 3. linear to vocab size
        output = self.linear(x)

        return output

