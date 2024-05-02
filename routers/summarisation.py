from fastapi import APIRouter
from model_server.summarisation import Summarisation

router = APIRouter(
    prefix="/summarise",
    tags=["Summarisation"],
    responses={404: {"description": "Not found"}},
)

summary = Summarisation()

@router.post("/summarise")
async def summarise_text(input_data):
    text = input_data.text
    result = await summary.summary(text)
    return result