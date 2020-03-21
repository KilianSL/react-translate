import torch
import torch.nn as nn
import torch.functional as F
import random

class Seq2seq(nn.Module):
    
    def __init__(self, encoder, decoder, device='cpu', with_attn=False):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.with_attn = with_attn
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        output_dim = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, output_dim).to(self.device)
        
        enc_outputs, hidden = self.encoder(src)
        
        # initialize output sequence with '<sos>'
        dec_output = trg[0,:]
        print("DEC_OUTPUT", dec_output.shape)
        
        # decoder token by token
        for t in range(1, max_len):
            if self.with_attn:
                dec_output, hidden, _ = self.decoder(dec_output, hidden, enc_outputs)
            else:
                dec_output, hidden = self.decoder(dec_output, hidden)
                print("DEC_OUTPUT RNN", dec_output.shape)
                
            outputs[t] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio
            
            pred_next = dec_output.argmax(1)
            
            dec_output = (trg[t] if teacher_force else pred_next)
        return outputs

    # greedy search for actual translation
    def greedy_search(self, src, sos_idx, max_len=50, return_attention=False):
        src = src.to(self.device)
        batch_size = src.shape[1]
        src_len = src.shape[0]
        
        outputs = torch.zeros(max_len, batch_size).to(self.device)
        
        enc_outputs, hidden = self.encoder(src)
        
        
        dec_output = torch.zeros(batch_size, dtype=torch.int64).to(self.device)
        dec_output.fill_(sos_idx)
        
        outputs[0] = dec_output
        
        attentions = torch.zeros(max_len, batch_size, src_len).to(self.device)
        
        for t in range(1, max_len):
            if self.with_attn:
                dec_output, hidden, attention_score = self.decoder(dec_output, hidden, enc_outputs)
                attentions[t] = attention_score
            else:
                dec_output, hidden = self.decoder(dec_output, hidden)
            
            dec_output = dec_output.argmax(1)

            outputs[t] = dec_output
            
        if return_attention:
            return outputs, attentions
        else:
            return outputs