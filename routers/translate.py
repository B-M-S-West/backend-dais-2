from fastapi import APIRouter
from pydantic import BaseModel
from model_server.translate_chinese import Translate_zh_to_en
from model_server.translate_russian import Translate_ru_to_en
from model_server.translate_ukranian import Translate_uk_to_en
from model_server.translate_arabic import Translate_ar_to_en

router = APIRouter(
    prefix="/translate",
    tags=["Text Translation"],
    responses={404: {"description": "Not found"}}
)

chinese = Translate_zh_to_en()
russian = Translate_ru_to_en()
ukranian = Translate_uk_to_en()
arabic = Translate_ar_to_en()

class TranslateInput(BaseModel):
    text: str

@router.post("/chinese")
async def chinese_to_english(input_data: TranslateInput):
    text = input_data.text
    translated_text = await chinese.translate(text)
    return {"translated_text": translated_text}

@router.post("/russian")
async def russian_to_english(input_data: TranslateInput):
    text = input_data.text
    translated_text = await russian.translate(text)
    return {"translated_text": translated_text}

@router.post("/ukranian")
async def ukranian_to_english(input_data: TranslateInput):
    text = input_data.text
    translated_text = await ukranian.translate(text)
    return {"translated_text": translated_text}

@router.post("/arabic")
async def arabic_to_english(input_data: TranslateInput):
    text = input_data.text
    translated_text = await arabic.translate(text)
    return {"translated_text": translated_text}



