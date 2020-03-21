import torch

from encoder_model import Encoder
from decoder_model import AttnDecoder
from attention_model import Attention as ATTENTION
from seq2seq_model import Seq2seq
from utils import Translator

translator = Translator()

# Define hyperparams
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = len(translator.SRC.vocab)
OUTPUT_DIM = len(translator.TRG.vocab)
EMB_DIM=256
HIDDEN_DIM=512
LAYERS=1
DROPOUT=0.5
BIDIRECTIONAL=True

# Create models
enc = Encoder(INPUT_DIM, EMB_DIM, HIDDEN_DIM, LAYERS, bidirectional=BIDIRECTIONAL)
dec = AttnDecoder(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, ATTENTION)
model = Seq2seq(enc, dec, device=DEVICE, with_attn=True).to(DEVICE)
model.load_state_dict(torch.load('API/model.pt', map_location=torch.device(DEVICE)))


# Handle API calls
import flask
from flask import request
from flask import jsonify
from flask_cors import CORS

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.route('/', methods=['GET']) #Return translated string
def predict():
    src_text = request.args['src']

    trg_text = translator(model, src_text)
    return jsonify(trg_text)

app.run()
