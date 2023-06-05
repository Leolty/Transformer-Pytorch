'''
@author: Tianyang Liu
@date: 2023-02-04
@homepage: leolty.github.io
'''

from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, n_head):
        """
        @param d_model: embedding size
        @param n_head: number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head

        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        @param q: [batch_size, seq_len, d_model]
        @param k: [batch_size, seq_len, d_model]
        @param v: [batch_size, seq_len, d_model]
        @param mask: [batch_size, seq_len, seq_len]
        """

        # linear projection
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # split into n_head
        q, k, v = self.split_heads(q, k, v)

        # scaled dot product attention
        context, attention = self.attention(q, k, v, mask=mask)

        # concat heads
        context = self.concat_heads(q, k, v)

        # final linear layer
        output = self.w_concat(context)

        return output, attention
        
    def split_heads(self, q, k, v):
        """
        Split q, k, v into n_head
        @param q: [batch_size, seq_len, d_model]
        @param k: [batch_size, seq_len, d_model]
        @param v: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = q.size()
        d_head = d_model // self.n_head

        # [batch_size, n_head, seq_len, d_head]
        q = q.view(batch_size, seq_len, self.n_head, d_head).transpose(1, 2)

        return q, k, v

    def concat_heads(self, q, k, v):
        """
        Concatenate heads
        @param q: [batch_size, n_head, seq_len, d_head]
        @param k: [batch_size, n_head, seq_len, d_head]
        @param v: [batch_size, n_head, seq_len, d_head]
        """
        batch_size, _, seq_len, _ = q.size()
        d_model = seq_len * self.n_head

        # [batch_size, seq_len, n_head, d_head]
        q = q.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        k = k.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        v = v.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return q, k, v

