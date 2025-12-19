import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcrever")
async def transcrever(audio: UploadFile = File(...)):
    # salva áudio temporariamente
    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await audio.read()
        tmp.write(data)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f,
                response_format="json"
            )
        text = getattr(result, "text", "")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return {"text": text}
