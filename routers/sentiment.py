from fastapi import APIRouter, Request
from pydantic import BaseModel
from model_server.sentiment import Sentiment

router = APIRouter(
    prefix="/sentiment",
    tags=["ner"],
    responses={404: {"description": "Not found"}},
)

sentiment = Sentiment()

class SentimentInput(BaseModel):
    text: str

@router.post("/sentiment")
async def recognize_entities(input_data: SentimentInput):
    text = input_data.text
    result = await sentiment.sentiment(text)
    return result