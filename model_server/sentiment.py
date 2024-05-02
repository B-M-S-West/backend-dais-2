import os
import torch
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

MODELS_DIRECTORY = os.getenv("MODELS_DIRECTORY", "/models")
MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME", "twitter-roberta-base-sentiment")

class Sentiment:
    def __init__(self):
        load_dir = os.path.join(MODELS_DIRECTORY, MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(load_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.generator = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=self.device)
        
    async def sentiment(self, text):
        sentiment = self.generator(text)
        return sentiment