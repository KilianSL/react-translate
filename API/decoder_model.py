import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, layers, PAD_IDX=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, layers, dropout=dropout, batch_first=True) # we don't set bidirectional here
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, emb_dim),
            nn.Tanh()
        )
        # This linear layer is to ensure the output layer has the same dimensionality with the embedding layer
        self.out = nn.Linear(emb_dim, output_dim)
        
    # x: (B)
    def forward(self, x, hidden):
        # we expand the dim of sequence length
        # x: (B, 1)
        x = x.unsqueeze(1)
        
        # embed: (B, 1, E)
        embed = self.dropout(self.embedding(x))

        
        # output: (B, 1, H)
        # h_n: (layers, B, H)
        output, hidden = self.rnn(embed, hidden)
        
        # output: (B, 1, E)
        output = self.ff(output)
        
        # output: (B, output_dim)
        output = self.out(output).squeeze(1)
        return output, hidden