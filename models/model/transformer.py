import torch
import torch.nn as nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, src_vocab_size, tgt_vocab_size, d_model, max_len, n_layers, n_head, n_hidden, dropout, device):
        """
        @param src_pad_idx: source padding index
        @param trg_pad_idx: target padding index
        @param trg_sos_idx: target start of sentence index
        @param src_vocab_size: source vocabulary size
        @param tgt_vocab_size: target vocabulary size
        @param d_model: embedding size
        @param max_len: maximum length of input sequence
        @param n_layers: number of EncoderLayers
        @param n_head: number of heads
        @param n_hidden: number of hidden size
        @param dropout: dropout rate
        """
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, max_len, n_layers, n_head, n_hidden, dropout, device)
        self.decoder = Decoder(tgt_vocab_size, d_model, max_len, n_layers, n_head, n_hidden, dropout, device)
    
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
    
    def make_src_mask(self, src):
        """
        @param src: [batch_size, src_len]
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size, 1, 1, src_len)
        return src_mask
    
    def make_trg_mask(self, trg):
        """
        @param trg: [batch_size, trg_len]
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size, 1, 1, trg_len)
        
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # (trg_len, trg_len)
        
        trg_mask = trg_pad_mask & trg_sub_mask
        # (batch_size, 1, trg_len, trg_len)
        
        return trg_mask

    def forward(self, src, trg):
        """
        @param src: [batch_size, src_len]
        @param trg: [batch_size, trg_len]
        """
        
        # 1. src_mask, trg_mask
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # 2. encoder
        encoder_output = self.encoder(src, mask=src_mask)
        
        # 3. decoder
        output = self.decoder(trg, encoder_output=encoder_output, self_mask=trg_mask, cross_mask=src_mask)
        
        return output
