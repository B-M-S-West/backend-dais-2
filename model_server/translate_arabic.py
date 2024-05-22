import os
import torch
from dotenv import load_dotenv
from transformers import MarianMTModel, MarianTokenizer, pipeline

load_dotenv()

MODELS_DIRECTORY = os.getenv("MODELS_DIRECTORY", "models")
MODEL_NAME = os.getenv("AR_EN_MODEL_NAME", "opus-mt-ar-en")

class Translate_ar_to_en:
    def _load_model(self):
        load_dir = os.path.join(MODELS_DIRECTORY, MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MarianMTModel.from_pretrained(load_dir).to(self.device)
        self.tokenizer = MarianTokenizer.from_pretrained(load_dir)
        self.generator = pipeline("translation_ar_to_en", model=self.model, tokenizer=self.tokenizer, device=self.device)

    async def translate(self, text):
        self._load_model()
        translated_text = self.generator(text)
        self._unload_model()
        return translated_text
    
    def _unload_model(self):
        self.model = None
        self.tokenizer = None
        self.generator = None