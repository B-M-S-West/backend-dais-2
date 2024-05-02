import os

import torch
from transformers import MarianMTModel, MarianTokenizer, pipeline
from dotenv import load_dotenv
load_dotenv()

MODELS_DIRECTORY = os.getenv("MODELS_DIRECTORY", "/models")
MODEL_NAME = os.getenv("AR_EN_MODEL_NAME", "opus-mt-ar-en")

class Translate_ar_to_en:
    def __init__(self, args):
        self.initialize(args)

    def initialize(self, args):
        load_dir = os.path.join(MODELS_DIRECTORY, MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MarianMTModel.from_pretrained(load_dir).to(self.device)
        self.tokenizer = MarianTokenizer.from_pretrained(load_dir)
        self.generator = pipeline("translation_ar_to_en", model=self.model, tokenizer=self.tokenizer, device=self.device)
        

    async def translate(self, text):
        # Translate text using pipeline
        translated_text = self.generator(text)
        return translated_text