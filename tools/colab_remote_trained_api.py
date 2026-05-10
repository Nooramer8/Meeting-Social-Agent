"""
Google Colab remote trained-model API for Meeting Social Agent.

Run this file's cells in Colab, expose port 8000 with ngrok or cloudflared,
then set TRAINED_REMOTE_API_URL in Railway to the public tunnel URL.
"""

# Colab cell 1:
# !pip install -q fastapi uvicorn python-multipart transformers torch sentencepiece soundfile scipy numpy huggingface_hub pyngrok

# Colab cell 2:
# from huggingface_hub import login
# login("hf_your_token_here")

# Colab cell 3:
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from scipy.signal import resample_poly
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


ASR_REPO_ID = "Nooriso/whisper-ar-meetings"
SUMMARY_REPO_ID = "Nooriso/meeting-summary-ar"
ASR_SUBFOLDER = "whisper-ar-meetings"
SUMMARY_SUBFOLDER = "meeting-summary-ar"
HF_TOKEN = "hf_your_token_here"

app = FastAPI(title="Meeting Social Agent Colab Trained API")

device_arg = 0 if torch.cuda.is_available() else -1
asr = pipeline(
    "automatic-speech-recognition",
    model=ASR_REPO_ID,
    token=HF_TOKEN,
    subfolder=ASR_SUBFOLDER,
    device=device_arg,
    chunk_length_s=30,
)
summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_REPO_ID, token=HF_TOKEN, subfolder=SUMMARY_SUBFOLDER)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_REPO_ID, token=HF_TOKEN, subfolder=SUMMARY_SUBFOLDER)
if torch.cuda.is_available():
    summary_model = summary_model.to("cuda")
summary_model.eval()


def load_audio(path: Path) -> dict:
    audio, sampling_rate = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    target_rate = 16000
    if sampling_rate != target_rate:
        gcd = np.gcd(sampling_rate, target_rate)
        audio = resample_poly(audio, target_rate // gcd, sampling_rate // gcd).astype("float32")
        sampling_rate = target_rate
    return {"array": audio, "sampling_rate": sampling_rate}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cuda": torch.cuda.is_available()}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("ar")) -> dict:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        audio_path = Path(tmp.name)

    generate_kwargs = {"task": "transcribe"}
    if language == "ar":
        generate_kwargs["language"] = "Arabic"
    elif language == "en":
        generate_kwargs["language"] = "English"

    result = asr(load_audio(audio_path), generate_kwargs=generate_kwargs)
    audio_path.unlink(missing_ok=True)
    return {"text": result.get("text", ""), "raw": result}


@app.post("/summarize")
def summarize(payload: dict) -> dict:
    transcript = (payload.get("transcript") or "").strip()
    target_language = (payload.get("target_language") or "ar").strip().lower()
    if not transcript:
        return {"text": ""}

    if target_language in {"ar", "auto"}:
        prefix = (
            "لخص الاجتماع بصيغة منظمة. أخرج الأقسام التالية فقط: "
            "العنوان، الملخص، النقاط الرئيسية، القرارات، المهام، تنبيهات للمراجعة، "
            "منشور فيسبوك، تعليق إنستغرام، الوسوم.\nالنص:\n"
        )
    else:
        prefix = (
            "Summarize the meeting in a structured format. Output only these sections: "
            "title, summary, key points, decisions, tasks, review warnings, Facebook post, "
            "Instagram caption, hashtags.\nTranscript:\n"
        )

    inputs = summary_tokenizer(prefix + transcript, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {key: value.to(summary_model.device) for key, value in inputs.items()}
    outputs = summary_model.generate(
        **inputs,
        max_new_tokens=512,
        min_new_tokens=40,
        num_beams=5,
        no_repeat_ngram_size=3,
        repetition_penalty=1.18,
        early_stopping=True,
    )
    text = summary_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": text}


# Colab cell 4, ngrok option:
# from pyngrok import ngrok
# ngrok.set_auth_token("your_ngrok_token_here")
# public_url = ngrok.connect(8000).public_url
# print("TRAINED_REMOTE_API_URL=", public_url)
# uvicorn.run(app, host="0.0.0.0", port=8000)
