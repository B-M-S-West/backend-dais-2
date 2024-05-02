from fastapi import APIRouter
from pydantic import BaseModel
from model_server.classification import Classification

router = APIRouter(
    prefix="/classification",
    tags=["Classification"],
    responses={404: {"description": "Not found"}},
)

classifier = Classification()

class ClassificationInput(BaseModel):
    text: str
    themes: list

@router.post("/classify")
async def classify_endpoint(input_data: ClassificationInput):
    text = input_data.text
    themes = input_data.themes
    # Assuming that your `classify` function takes text and themes as arguments
    result = await classifier.classify(text, themes)
    return result

@router.post("/classify_multi")
async def classify_endpoint_multi(input_data: ClassificationInput):
    text = input_data.text
    themes = input_data.themes
    # Assuming that your `classify` function takes text and themes as arguments
    result = await classifier.classify_multi(text, themes)
    return result