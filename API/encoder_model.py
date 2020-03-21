import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, layers, PAD_IDX=1, bidirectional=False, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.bidirectional = bidirectional
        self.PAD_IDX = PAD_IDX
          
        
        # If we use a bidirectional encoder to encode both forward and backward context,
        # the dimension of the hidden state will double
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            ff_input_dim = 2 * hidden_dim
        else:
            ff_input_dim = hidden_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, self.hidden_dim, layers, dropout=dropout, \
                           bidirectional=bidirectional, bias=False, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(ff_input_dim, hidden_dim),
            nn.Tanh()
        )
        
    # x: (T, B)
    def forward(self, x):
        # x: (B, T)
        x = x.permute(1, 0)
        
        # x: (B, T, E)
        x = self.dropout(self.embedding(x))

        # outputs: (B, T, H*directions)
        # h_n: (layers*directions, B, H)
        outputs, (h_n, c_n) = self.rnn(x)
        

        if self.bidirectional:
            # concatenate the forward and backward hidden states
            h_n = torch.cat((h_n[0::2,:,:], h_n[1::2,:,:]), dim = -1)
            c_n = torch.cat((c_n[0::2,:,:], c_n[1::2,:,:]), dim = -1)
        
        # h_n: (layers, B, H)
        # c_n: (layers, B, H)
        h_n = self.ff(h_n)
        c_n = self.ff(c_n)
        
        # outputs: ()
        return outputs, (h_n, c_n)
        