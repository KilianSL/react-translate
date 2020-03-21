import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        
        self.W = nn.Linear(3*hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    # dec_hidden： （1, B, H)
    # enc_outputs: (B, T, 2H)
    def forward(self, dec_hidden, enc_outputs):
        batch_size = dec_hidden.shape[1]
        src_len = enc_outputs.shape[1]
        
        # dec_hidden: (B, 1, H)
        dec_hidden = dec_hidden.permute(1, 0, 2)
        # dec_hidden: (B, T, H)
        dec_hidden = dec_hidden.repeat(1, src_len, 1)
        
        # energy: (B, T, H)
        energy = torch.tanh(self.W(torch.cat((dec_hidden, enc_outputs), dim=2)))
        
        # attention: (B, T)
        attention = F.softmax(self.v(energy).squeeze(2), dim=1)
        
        return attention