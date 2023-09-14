# import pickle
import pickle5 as pickle
import torch

from model import Transformer
from utils2 import infer
from tokenizers import Tokenizer


class Translator:

    def __init__(self, tokenizer_path, pretrained_weights_path, config_path):
        self.tokenizer_path = tokenizer_path
        self.pretrained_weights_path = pretrained_weights_path
        self.config_path = config_path

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        with open(config_path, 'rb') as handle:
            config = pickle.load(handle)

        self.model = Transformer(src_vocabSize=config["src_vocabSize"],
                                 tgt_vocabSize=config["tgt_vocabSize"],
                                 src_max_len=config["src_max_len"],
                                 tgt_max_len=config["tgt_max_len"],
                                 noHeads=config["noHeads"],
                                 d_model=config["d_model"],
                                 d_ff=config["d_ff"],
                                 dropout=config["dropout"],
                                 noEncoder=config["noEncoder"],
                                 noDecoder=config["noDecoder"],
                                 pad_index=config["pad_index"],
                                 )

        self.model.load_state_dict(torch.load(pretrained_weights_path))

    def translate(self, text):

        translation = infer(self.model, text, self.tokenizer)

        return translation

