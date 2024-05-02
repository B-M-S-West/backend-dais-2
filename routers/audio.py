from fastapi import APIRouter, File, UploadFile
from pydub import AudioSegment
import io
from model_server.whisper import Audio

router = APIRouter(
    prefix="/audio",
    tags=["Audio Translation and Transcription"],
    responses={404: {"description": "Not found"}},
)

audio = Audio()


@router.post("/transcribe")
async def transcribe_audio_file(file: UploadFile = File(...)):
    try:
        # Read the audio file
        audio_data = await file.read()
        # Pass the audio data to the Whisper model
        result = transcribe_audio(audio_data)
        return result
    except Exception as e:
        return {"error": str(e)}


@router.post("/translate")
async def translate_audio_file(file: UploadFile = File(...)):
    try:
        # Read the audio file
        audio_data = await file.read()
        # Pass the audio data to the Whisper model
        result = translate_audio(audio_data)
        return result
    except Exception as e:
        return {"error": str(e)}
    

# Function to convert the sample rate of the audio file
def convert_sample_rate(audio_bytes):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    # Check if the sample rate is 16000 Hz, if not convert it
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    return audio


def transcribe_audio(audio_bytes):
    # Convert the audio bytes to the correct sample rate
    audio = convert_sample_rate(audio_bytes)
    # Save the converted audio file to a temporary file
    temp_file = "temp_audio.mp3"
    audio.export(temp_file, format="mp3")

    # Run the transcription pipeline
    result = audio.transcribe(temp_file)
    return result["text"]


def translate_audio(audio_bytes):
    # Convert the audio bytes to the correct sample rate
    audio = convert_sample_rate(audio_bytes)
    # Save the converted audio file to a temporary file
    temp_file = "temp_audio.mp3"
    audio.export(temp_file, format="mp3")

    # Run the translation pipeline
    result = audio.translate(temp_file)
    return result["translation_text"]