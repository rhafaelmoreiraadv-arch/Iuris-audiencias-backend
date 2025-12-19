import os
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI

# Carrega variáveis de ambiente (.env / Render)
load_dotenv()

app = FastAPI()

# Cliente OpenAI usando a chave que você colocou na Render
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Libera o frontend (tanto no iPad quanto em outros devices) a chamar o backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # se quiser, depois restringimos pro domínio do app
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
    Recebe um arquivo de áudio (webm/mp3/wav), manda para a OpenAI
    e devolve o texto transcrito.
    """
    tmp_path = None

    try:
        # Garante uma extensão pro arquivo temporário
        suffix = os.path.splitext(audio.filename or "")[1] or ".webm"

        # Salva o áudio em arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await audio.read()
            tmp.write(data)
            tmp_path = tmp.name

        # Abre o arquivo e envia para a OpenAI
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",  # modelo de transcrição
                file=f,
                response_format="json",
            )

        texto = getattr(result, "text", "")

        return {"texto": texto}

    except Exception as e:
        # Se der erro, devolve mensagem clara pro frontend
        return JSONResponse(
            status_code=500,
            content={
                "erro": "Falha ao transcrever o áudio.",
                "detalhe": str(e),
            },
        )
    finally:
        # Tenta apagar o arquivo temporário
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
