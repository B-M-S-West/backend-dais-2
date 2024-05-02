import os
import torch
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

MODELS_DIRECTORY = os.getenv("MODELS_DIRECTORY", "/models")
MODEL_NAME = os.getenv("SUMMARIsATION_MODEL_NAME", "long-t5-tglobal-base-16384-book-summary")

class Summarisation:
    def __init__(self):
        load_dir = os.path.join(MODELS_DIRECTORY, MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.generator = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=self.device)
        
    async def summary(self, text: str, maximum_length: int) -> str:
        summary = self.generator(text, max_length = maximum_length)
        return summary