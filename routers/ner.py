from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
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
async def named_entity_resolution(input_data: NerInput) -> JSONResponse:
    text = input_data.text
    response = await ner.ner(text)
    print('response', response)
    return JSONResponse(content=response)