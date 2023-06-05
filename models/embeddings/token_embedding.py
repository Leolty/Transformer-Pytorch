'''
@author: Tianyang Liu
@date: 2023-02-04
@homepage: leolty.github.io
'''

from torch import nn

class TokenEmbedding(nn.Embedding):
    '''
    dense representation of tokens leveraging weighted matrix
    '''

    def __init__(self, vocab_size, d_model):
        """
        @param vocab_size: size of vocabulary
        @param d_model: dimension of embedding
        """

        # The value of padding_idx should be set to the index of the padding token in the vocabulary
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=0)

