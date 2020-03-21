import torch
import torch.nn as nn

class AttnDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, attention, PAD_IDX=1, dropout=0.1):
        super(AttnDecoder, self).__init__()
        
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.attention = attention(hidden_dim)
        self.rnn = nn.LSTM(emb_dim+2*hidden_dim, hidden_dim, dropout=dropout, batch_first=True) 
        
        self.ff = nn.Sequential(
            nn.Linear(3*hidden_dim + emb_dim, emb_dim),
            nn.Tanh()
        )
        
        # This linear layer is to ensure the output layer has the same dimensionality with the embedding layer
        self.out = nn.Linear(emb_dim, output_dim)
        
    
    # x: (B)
    # hidden=(h_n, c_n): (1, B, H)
    # enc_outputs=(B, T, 2H)
    def forward(self, x, hidden, enc_outputs):
        # we expand the dim of sequence length
        # x: (B, 1)
        x = x.unsqueeze(1)
        
        # embed: (B, 1, E)
        embed = self.dropout(self.embedding(x))
        
        # attn_score: (B, 1, T)
        attn_score = self.attention(hidden[0], enc_outputs).unsqueeze(1)
        
        # c: (B, 1, 2H)
        c = torch.bmm(attn_score, enc_outputs)
        
        # output: (B, 1, H)
        # h_n: (1, B, H)
        output, hidden = self.rnn(torch.cat((embed, c), dim=2), hidden)
        
        # output: (B, 1, E)
        output = self.ff(torch.cat((output, c, embed), dim=2))
        
        # output: (B, output_dim)
        output = self.out(output).squeeze(1)
        return output, hidden, attn_score.squeeze(1)