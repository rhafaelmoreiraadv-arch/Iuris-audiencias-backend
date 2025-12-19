import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI


# Carrega variáveis de ambiente (.env / Render)
load_dotenv()

app = FastAPI()

# Cliente OpenAI usando a chave que você colocou nas variáveis do Render
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Libera o frontend (iPad, navegador, etc.) para chamar o backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # depois podemos restringir pro domínio do app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    Rota de saúde. Usada só pra testar se o backend está no ar.
    """
    return {"status": "ok", "service": "iuris-audiencias-backend"}


# Aceita /transcrever-audio, /transcrever e /transcribe
@app.post("/transcrever-audio")
@app.post("/transcrever")
@app.post("/transcribe")
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
                model="gpt-4o-transcribe",   # modelo de transcrição
                file=f,
                response_format="json",
            )

        texto = getattr(result, "text", "")

        if not texto:
            raise HTTPException(
                status_code=500,
                detail="Falha ao obter texto da transcrição."
            )

        return {"text": texto}

    except HTTPException:
        # Deixa passar HTTPException “como está”
        raise
    except Exception as e:
        # Qualquer outro erro vira 500 pro frontend
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao transcrever áudio: {str(e)}",
        )
    finally:
        # Tenta apagar o arquivo temporário
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
