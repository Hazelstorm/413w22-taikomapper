import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from hyper_param import *

WIN_SIZE = get_window_size() * 2 + 1
N_MELS = get_mel_param()['n_mels']
AUDIO_DIM = WIN_SIZE * N_MELS
MAX_SNAP = get_max_snap()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(MAX_SNAP).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, MAX_SNAP, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
    
class TransformerModel(nn.Module):
    
    def __init__(self, d_model: int = 256, n_heads: int = 4, 
                 d_hid: int = 512, n_layers: int = 3, dropout: float = 0.1):
        """
        Parameters
        ----------
        d_model: dimension of vector embedding
        nhead: number of transformer encoders
        d_hid: hidden size of encoder feedforward component
        n_layers: number of encoders
        
        """
        super().__init__()
        self.embedding = nn.Linear(AUDIO_DIM, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_model, 5)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, src: Tensor, mask: Tensor):
        """
        Args:
            src: Tensor, shape [batch_num, seq_len, win_size, n_mels]
            mask: Tensor, shape [batch_num, seq_len, seq_len]

        Returns:
            Tensor, shape [batch_num, seq_len, 5]
        """
        src = torch.flatten(src, start_dim=2)
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, mask)
        src = self.decoder(src)
        src = self.softmax(src)
        return src
    
model = TransformerModel()

if torch.cuda.is_available():
    model = model.cuda()
    