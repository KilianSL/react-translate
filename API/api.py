import torch

from encoder_model import Encoder
from decoder_model import AttnDecoder
from attention_model import Attention as ATTENTION
from seq2seq_model import Seq2seq
from utils import Translator

# Define hyperparams
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 4
OUTPUT_DIM = 4
EMB_DIM = 10
HIDDEN_DIM = 6
LAYERS =  2

# Create models
enc = Encoder(INPUT_DIM, EMB_DIM, HIDDEN_DIM, LAYERS)
dec = AttnDecoder(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, ATTENTION)
model = Seq2seq(enc, dec, device=DEVICE, with_attn=True).to(DEVICE)
model.load_state_dict(torch.load('./model.pt', map_location=torch.device(DEVICE)))

translator = Translator()

print(translator.translate(model, "Ich bin ein bein"))
