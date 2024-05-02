import os
import torch
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

load_dotenv()

MODELS_DIRECTORY = os.getenv("MODELS_DIRECTORY", "/models")
MODEL_NAME = os.getenv("AUDIO_MODEL_NAME", "whisper-large-v3")

class Audio:
    def __init__(self):
        load_dir = os.path.join(MODELS_DIRECTORY, MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(load_dir).to(self.device)
        self.processor = AutoProcessor.from_pretrained(load_dir)
        self.generator = pipeline(
                                  "automatic-speech-recognition", 
                                  model=self.model,
                                  tokenizer=self.processor.tokenizer,
                                  feature_extractor=self.processor.feature_extractor,
                                  max_new_tokens=128,
                                  chunk_length_s=30,
                                  batch_size=16,
                                  torch_dtype=self.torch_dtype,
                                  device=self.device
                                  )
        
    async def transcribe(self, audio):
        transcription = self.generator(audio)
        return transcription
    
    async def translate(self, audio):
        transcription = self.generator(audio, generate_kwargs={"task": "translate"})
        return transcription