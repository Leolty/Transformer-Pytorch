'''
@author: Tianyang Liu
@date: 2023-02-04
@homepage: leolty.github.io
'''

from torch import nn

from model.embeddings.token_embedding import TokenEmbedding
from model.embeddings.position_embedding import PositionEmbedding

class TransformerEmbedding(nn.Module):
    """
    Transformer Embedding: combine token embedding and position embedding
    """
    def __init__(self, vocab_size, d_model, max_len, dropout, device):
        """
        @param vocab_size: vocabulary size
        @param d_model: embedding size
        @param max_len: maximum length of input sequence
        @param dropout: dropout rate
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(d_model, max_len,device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(x)
        x = self.dropout(token_embedding + position_embedding)
        return x