from dotenv import load_dotenv
import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

load_dotenv()

MODELS_DIRECTORY = os.getenv("MODELS_DIRECTORY", "/models")
MODEL_NAME = os.getenv("NER_MODEL_NAME", "bert-base-NER")

class NER:
    def __init__(self, args):
        self.initialize(args)

    def initialize(self, args):
        load_dir = os.path.join(MODELS_DIRECTORY, MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(load_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.generator = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=self.device, aggregation_strategy="simple")
        
    async def ner(self, text):
        ner = self.generator(text)
        return ner
