import os
import tempfile
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# STT
from faster_whisper import WhisperModel

# LLM
from llama_cpp import Llama

# TTS (Piper)
from piper import PiperVoice

# audio library
import numpy as np
import soundfile as sf

# Twilio
from xml.etree import ElementTree as ET

# Config (ENV)
BASE_DIR = Path(os.getenv("BASE_DIR", "/home/yashdhanani30/ai-call-server"))
PIPER_MODEL = os.getenv("PIPER_MODEL", str(BASE_DIR / "piper_models" / "en_US-amy-medium.onnx"))
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", str(BASE_DIR / "models" / "llama.gguf"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")  # tiny, base, small, medium, large

# where to write generated wav files
ASSETS_DIR = BASE_DIR + "/generated"
os.makedirs(ASSETS_DIR, exist_ok=True)

app = FastAPI(title="AI Calling Agent - Local Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons + locks (to prevent concurrent access collisions)
_whisper_lock = asyncio.Lock()
_tts_lock = asyncio.Lock()
_llm_lock = asyncio.Lock()

print("Loading models (this may take a while)...")

# Load Whisper model (STT)
whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8_float16")  # adjust compute_type if needed

# Load Llama model (LLM)
llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048)

# Load Piper voice
piper_voice = PiperVoice.load(PIPER_MODEL)

print("Models loaded. Server ready.")


# ---------- Helpers ----------
def save_temp_file(data: bytes, suffix: str = "") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    return path


def synthesize_to_wav(text: str, out_filename: Optional[str] = None, sample_rate: int = 24000) -> str:
    """
    Use Piper to synthesize text -> WAV file. Returns path to wav.
    """
    if out_filename is None:
        out_filename = f"{uuid.uuid4().hex}.wav"
    out_path = os.path.join(ASSETS_DIR, out_filename)

    # synth generator
    gen = piper_voice.synthesize(text)

    arrays = []
    sr = None
    for chunk in gen:
        # chunk contains attributes like audio_int16_array, sample_rate
        if hasattr(chunk, "audio_int16_array") and chunk.audio_int16_array is not None:
            arr = chunk.audio_int16_array
            arrays.append(arr)
            sr = chunk.sample_rate
        elif hasattr(chunk, "audio_float_array") and chunk.audio_float_array is not None:
            # convert float to int16
            arr = (chunk.audio_float_array * 32767.0).astype(np.int16)
            arrays.append(arr)
            sr = chunk.sample_rate
        elif hasattr(chunk, "audio_int16_bytes") and chunk.audio_int16_bytes:
            # fallback: read bytes into numpy
            arr = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            arrays.append(arr)
            sr = chunk.sample_rate if hasattr(chunk, "sample_rate") else sample_rate
        else:
            # unknown chunk format - skip
            continue

    if len(arrays) == 0:
        raise RuntimeError("No audio chunks received from Piper")

    # concatenate
    try:
        audio = np.concatenate(arrays)
    except ValueError:
        # if arrays are 2D, attempt vstack then reshape
        audio = np.vstack(arrays)
    # write wav (PCM_16)
    sf.write(out_path, audio, sr, subtype="PCM_16")
    return out_path


# ---------- API endpoints ----------

@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """Upload an audio file (wav) -> returns transcription text."""
    data = await file.read()
    tmp = save_temp_file(data, suffix=".wav")
    async with _whisper_lock:
        segments, info = whisper.transcribe(tmp, language=language)
        text = " ".join([s.text for s in segments]).strip()
    os.remove(tmp)
    return {"text": text, "language": info.language if hasattr(info, "language") else language}


@app.post("/ai")
async def ai_endpoint(prompt: str = Form(...), max_tokens: int = Form(256)):
    """Send prompt -> get generated reply from local LLM."""
    async with _llm_lock:
        res = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=0.7)
    # llama-cpp returns a dict with choices
    text = ""
    if isinstance(res, dict) and "choices" in res and len(res["choices"]) > 0:
        text = res["choices"][0].get("text", "").strip()
    else:
        text = str(res)
    return {"reply": text}


@app.post("/tts")
async def tts_endpoint(text: str = Form(...), voice: Optional[str] = Form(None)):
    """Synthesize text and return audio file (wav)."""
    async with _tts_lock:
        wav_path = synthesize_to_wav(text)
    # return wav file ready to stream
    return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve a generated audio file by filename (for Twilio to fetch)."""
    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(path):
        return PlainTextResponse("Not found", status_code=404)
    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.post("/twilio/twiml")
async def twilio_twiml(request: Request):
    """
    Twilio voice webhook that returns TwiML. This endpoint will:
    - synth a quick greeting and return <Play> URL to Twilio
    - in production you'd stream, but demo uses pre-generated wav URL.
    """
    # generate text greeting
    greeting = "Hello. This is your AI test call. Please say something after the beep."
    async with _tts_lock:
        wav = synthesize_to_wav(greeting, out_filename=f"twilio_greeting_{uuid.uuid4().hex}.wav")
    # Build TwiML: <Response><Play>https://yourserver/audio/{filename}</Play></Response>
    response = ET.Element("Response")
    play = ET.SubElement(response, "Play")
    host = request.headers.get("host", None)
    scheme = "https" if request.url.scheme == "https" else "http"
    url = f"{scheme}://{host}/audio/{os.path.basename(wav)}"
    play.text = url
    xml_str = ET.tostring(response, encoding="utf-8")
    return Response(content=xml_str, media_type="application/xml")


@app.post("/twilio/status")
async def twilio_status(request: Request):
    # Twilio status callbacks: just log body
    body = await request.form()
    print("Twilio status callback:", dict(body))
    return PlainTextResponse("OK")


@app.get("/")
async def root():
    return {"status": "AI calling agent backend - ok"}
