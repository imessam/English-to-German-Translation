from fastapi import FastAPI
from pydantic import BaseModel

from run import Translator

tokenizer_path = "tokenizers/tokenizer_shared.json"
pretrained_weights_path = "trained/transformer-weights-small"
config_path = "config.pickle"

app = FastAPI()
translator = Translator(tokenizer_path=tokenizer_path, pretrained_weights_path=pretrained_weights_path,
                        config_path=config_path)


class Text(BaseModel):
    text: str


@app.post(path="/translate")
async def run(text: Text):
    return translator.translate(text.text)
