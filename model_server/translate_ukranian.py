import os
import torch
from dotenv import load_dotenv
from transformers import pipeline, MarianMTModel, MarianTokenizer

load_dotenv()

MODELS_DIRECTORY = os.getenv("MODELS_DIRECTORY", "/models")
MODEL_NAME = os.getenv("UK_EN_MODEL_NAME", "opus-mt-uk-en")

class Translate_uk_to_en:
    def __init__(self):
        load_dir = os.path.join(MODELS_DIRECTORY, MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MarianMTModel.from_pretrained(load_dir).to(self.device)
        self.tokenizer = MarianTokenizer.from_pretrained(load_dir)
        self.generator = pipeline("translation_uk_to_en", model=self.model, tokenizer=self.tokenizer, device=self.device)
        

    async def translate(self, text):
        translated_text = self.generator(text)
        return translated_text