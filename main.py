import warnings
from fastapi import FastAPI
import uvicorn
from routers import audio, classification, minio, ner, sentiment, summarisation, translate

warnings.filterwarnings("ignore")


app = FastAPI(
    title="DAIS API",
    description="API for DAIS Models",
    version="1.0.2",
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(audio.router)
app.include_router(classification.router)
app.include_router(minio.router)
app.include_router(ner.router)
app.include_router(sentiment.router)
app.include_router(summarisation.router)
app.include_router(translate.router)

if __name__ == '__main__':
    uvicorn.run("main:app", workers=0, host="0.0.0.0", port=5000)