from fastapi import APIRouter
from pydantic import BaseModel
from model_server.summarisation import Summarisation

router = APIRouter(
    prefix="/summarise",
    tags=["Summarisation"],
    responses={404: {"description": "Not found"}},
)

summary = Summarisation()

class SummarisationInput(BaseModel):
    text: str
    maximum_length: int

@router.post("/summarise")
async def summarise_text(input_data: SummarisationInput):
    text = input_data.text
    maximum_length = input_data.maximum_length
    result = await summary.summary(text, maximum_length)
    return result