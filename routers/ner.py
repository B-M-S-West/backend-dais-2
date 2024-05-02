from fastapi import APIRouter
from pydantic import BaseModel
from model_server.ner import NER

router = APIRouter(
    prefix="/ner",
    tags=["ner"],
    responses={404: {"description": "Not found"}},
)

ner = NER()

class NerInput(BaseModel):
    text: str

@router.post("/entities")
async def recognize_entities(input_data: NerInput):
    text = input_data.text
    result = await ner.ner(text)
    return result