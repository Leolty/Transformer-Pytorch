from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, n_hidden, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm_1 = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)

        self.feed_forward = PositionWiseFeedForward(d_model, n_hidden)
        self.norm_2 = LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        @param x: [batch_size, seq_len, d_model]
        @param mask: [batch_size, seq_len, seq_len]
        """
        
        # 1. self attention
        _x = x
        context, attention = self.attention(x, mask=mask)

        # 2. add & norm
        x = self.norm_1(_x + self.dropout_1(context))

        # 3. feed forward, add & norm
        _x = x
        x = self.norm_2(_x + self.dropout_2(self.feed_forward(x)))

        return x

