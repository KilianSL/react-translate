from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import torch
import torchtext

class Translator():  #class to hold source and target fields, and handle tokenisation etc. 

    def __init__(self):
        
        self.SRC = Field(
            tokenize="spacy",
            tokenizer_language="de",
            init_token="<sos>",
            eos_token="<eos>",
            lower=True
        )

        self.TRG = Field(
            tokenize="spacy",
            tokenizer_language="en",
            init_token="<sos>",
            eos_token="<eos>",
            lower=True
        )

        data, _, _ = Multi30k.splits(exts = ('.de', '.en'), fields=(self.SRC, self.TRG))

        self.SRC.build_vocab(data, min_freq = 2)
        self.TRG.build_vocab(data, min_freq = 2)

    # def get_eos_position(self, tensor, field):
    #     for position, tok_idx in enumerate(tensor):
    #         tok_idx = int(tok_idx)
    #         token = field.vocab.itos[tok_idx]
        
    #         if token == '<eos>' or token == '<pad>':
    #             break
    #     return position

    def get_text_from_tensor(self, tensor, field, eos='<eos>'):
        batch_output = []
        for i in range(tensor.shape[1]):
            sequence = tensor[:,i]
            words = []
            for tok_idx in sequence:
                tok_idx = int(tok_idx)
                token = field.vocab.itos[tok_idx]

                if token == '<sos>':
                    continue
                elif token == '<eos>' or token == '<pad>':
                    break
                else:
                    words.append(token)
            words = " ".join(words)
            batch_output.append(words)
        return batch_output

    def __call__(self, model, text):
        tokens = self.SRC.preprocess(text)
        input_tensor = self.SRC.process([tokens])
        
        with torch.no_grad():
            outputs, _ = model.greedy_search(input_tensor, self.TRG.vocab.stoi['<sos>'], return_attention=True)
            output_text = self.get_text_from_tensor(outputs, self.TRG)
        return output_text

















