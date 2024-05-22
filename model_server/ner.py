import os
import torch
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

load_dotenv()

MODELS_DIRECTORY = os.getenv("MODELS_DIRECTORY", "models")
MODEL_NAME = os.getenv("NER_MODEL_NAME", "bert-base-NER")

class NER:
    def _load_model(self):
        load_dir = os.path.join(MODELS_DIRECTORY, MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(load_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.generator = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=self.device, aggregation_strategy="simple")

    async def ner(self, text) -> list[dict[str, str]]:
        self._load_model()
        ner_result = self.generator(text)
        extracted_entities = []
        for entity in ner_result:
            extracted_entity = {
                'entity_group': entity['entity_group'],
                'score': str(entity['score']),
                'word': entity['word'],
                'start': str(entity['start']),
                'end': str(entity['end'])
            }
            extracted_entities.append(extracted_entity)
        self._unload_model()
        return extracted_entities
    
    def _unload_model(self):
        self.model = None
        self.tokenizer = None
        self.generator = None
