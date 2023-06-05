import math

from torch import nn

class ScaleDotProductAttention(nn.Module):
    """
    compute scaled dot product attention
    
    Query: current word 
    Key: all other words
    Value: all other words
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None):
        """
        @param q: [batch_size, n_head, seq_len, d_q]
        @param k: [batch_size, n_head, seq_len, d_k]
        @param v: [batch_size, n_head, seq_len, d_v]
        @param mask: [batch_size, seq_len, seq_len]
        """

        _, _, _, d_k = k.size()
    

        # 1. dot product of query and key to compute similarity
        score = q @ k.transpose(2, 3) / math.sqrt(d_k) # [batch_size, n_head, seq_len, seq_len]

        # 2. apply mask (optional)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        
        # 3. apply softmax to obtain attention weights
        attention = self.softmax(score) # [batch_size, n_head, seq_len, seq_len]

        # 4. apply attention weights to the value
        context = attention @ v # [batch_size, n_head, seq_len, d_v]

        return context, attention