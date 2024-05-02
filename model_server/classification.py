import os
import torch
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, BartForSequenceClassification

load_dotenv()

MODELS_DIRECTORY = os.getenv("MODELS_DIRECTORY", "/models")
MODEL_NAME = os.getenv("CLASSIFICATION_MODEL_NAME", "theme-classification")

class Classification:
    def __init__(self):
        load_dir = os.path.join(MODELS_DIRECTORY, MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BartForSequenceClassification.from_pretrained(load_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.generator = pipeline("zero-shot-classification", model=self.model, tokenizer=self.tokenizer, device=self.device)
        
    async def classify(self, text, themes):
        classification = self.generator(text, themes)
        classification.pop('sequence', None)
        return classification
    
    async def classify_multi(self, text, themes):
        classification = self.generator(text, themes, multi_label=True)
        classification.pop('sequence', None)
        return classification