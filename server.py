import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

# Cliente oficial OpenAI
client = OpenAI()

# Libera CORS geral (depois restringimos)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "iuris-audiencias-backend"}

@app.post("/transcrever-audio")
async def transcrever_audio(audio: UploadFile = File(...)):
    """
    Recebe um arquivo de áudio (webm/mp3/wav), manda para OpenAI
    e devolve o texto transcrito.
    """
    tmp_path = None

    try:
        suffix = os.path.splitext(audio.filename or "")[1] or ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await audio.read()
            tmp.write(data)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f,
                response_format="json"
            )

        texto = getattr(result, "text", "")
        return {"texto": texto}

    except Exception as e:
        return {"erro": str(e)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
