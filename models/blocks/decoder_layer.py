from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_head, n_hidden, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm_1 = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)

        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm_2 = LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout)

        self.feed_forward = PositionWiseFeedForward(d_model, n_hidden)
        self.norm_3 = LayerNorm(d_model)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        @param x: [batch_size, seq_len, d_model]
        @param encoder_output: [batch_size, seq_len, d_model]
        @param self_mask: [batch_size, seq_len, seq_len]
        @param cross_mask: [batch_size, seq_len, seq_len]
        """
        
        # 1. self attention
        _x = x
        context, self_attention = self.self_attention(x, x, x, mask=self_mask)

        # 2. add & norm
        x = self.norm_1(_x + self.dropout_1(context))

        # 3. cross attention
        if cross_mask is not None:
            _x = x
            context, cross_attention = self.cross_attention(x, encoder_output, encoder_output, mask=cross_mask)

            # 4. add & norm
            x = self.norm_2(_x + self.dropout_2(context))
        
        # 5. feed forward, add & norm
        _x = x
        x = self.norm_3(_x + self.dropout_3(self.feed_forward(x)))

        return x