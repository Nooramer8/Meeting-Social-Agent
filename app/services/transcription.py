from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from groq import Groq
from openai import OpenAI

from app.config import get_settings
from app.services.model_downloader import ensure_trained_asr_model


SUPPORTED_EXTENSIONS = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.ogg'}
SUPPORTED_LANGUAGE_HINTS = {'auto', 'ar', 'en'}


def normalize_language(language: str | None) -> str:
    value = (language or 'auto').strip().lower()
    aliases = {
        'arabic': 'ar',
        'عربي': 'ar',
        'العربية': 'ar',
        'english': 'en',
        'انجليزي': 'en',
        'الإنجليزية': 'en',
    }
    value = aliases.get(value, value)
    if not value:
        return 'auto'
    return value


def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    text = getattr(response, 'text', None)
    if text:
        return text
    if isinstance(response, dict):
        return response.get('text', '')
    if hasattr(response, 'model_dump'):
        data = response.model_dump()
        return data.get('text', '') or str(data)
    return str(response)


def _to_jsonable(response: Any) -> dict[str, Any] | str:
    if isinstance(response, (dict, str)):
        return response
    if hasattr(response, 'model_dump'):
        return response.model_dump()
    return {'raw': str(response)}


def _validate_audio_path(file_path: str | Path) -> Path:
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f'Unsupported file type: {path.suffix}. Use one of: {sorted(SUPPORTED_EXTENSIONS)}')
    if not path.exists():
        raise FileNotFoundError(f'Audio file not found: {path}')
    return path


def _is_retryable_transcription_error(exc: Exception) -> bool:
    text = str(exc).lower()
    retryable_markers = (
        'timed out',
        'timeout',
        'temporarily unavailable',
        'connection',
        'rate limit',
        '429',
        '500',
        '502',
        '503',
        '504',
    )
    return any(marker in text for marker in retryable_markers)


def _load_audio_for_transformers(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError('soundfile, scipy, and numpy are required for local Transformers ASR. Run: pip install librosa') from exc

    audio, sampling_rate = sf.read(path, dtype='float32', always_2d=False)
    if getattr(audio, 'ndim', 1) > 1:
        audio = audio.mean(axis=1)

    target_rate = 16000
    if sampling_rate != target_rate:
        gcd = np.gcd(sampling_rate, target_rate)
        audio = resample_poly(audio, target_rate // gcd, sampling_rate // gcd).astype('float32')
        sampling_rate = target_rate

    return {'array': audio, 'sampling_rate': sampling_rate}


def transcribe_file(
    file_path: str | Path,
    language: str | None = None,
    backend: str | None = None,
) -> tuple[str, dict[str, Any] | str]:
    """Cloud transcription using Groq or OpenAI."""
    settings = get_settings()
    selected_backend = (backend or settings.transcription_backend or 'groq').strip().lower()

    path = _validate_audio_path(file_path)
    lang = normalize_language(language or settings.default_language)
    model = settings.transcription_model

    if selected_backend == 'groq':
        if not settings.groq_api_key:
            raise RuntimeError('GROQ_API_KEY is missing. Add it to .env or set TRANSCRIPTION_BACKEND=openai.')
        client: Groq | OpenAI = Groq(api_key=settings.groq_api_key, timeout=settings.cloud_request_timeout_seconds)
    elif selected_backend == 'openai':
        if not settings.openai_api_key:
            raise RuntimeError('OPENAI_API_KEY is missing. Add it to .env or set TRANSCRIPTION_BACKEND=groq.')
        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.cloud_request_timeout_seconds)
    else:
        raise ValueError('TRANSCRIPTION_BACKEND must be groq or openai.')

    attempts = max(1, settings.cloud_transcription_retries + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with path.open('rb') as audio_file:
                kwargs: dict[str, Any] = {
                    'model': model,
                    'file': audio_file,
                }
                if lang != 'auto':
                    kwargs['language'] = lang

                # OpenAI-only diarized transcription. Groq Whisper does not use diarized_json.
                if selected_backend == 'openai' and 'diarize' in model:
                    kwargs['response_format'] = 'diarized_json'
                    kwargs['chunking_strategy'] = 'auto'
                else:
                    kwargs['response_format'] = 'json'

                response = client.audio.transcriptions.create(**kwargs)
            return _extract_text(response), _to_jsonable(response)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts or not _is_retryable_transcription_error(exc):
                raise
            time.sleep(max(0.0, settings.cloud_transcription_retry_delay_seconds))

    raise RuntimeError('Cloud transcription failed without returning a response.') from last_error


@lru_cache(maxsize=4)
def _get_faster_whisper_model(model_size: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError('faster-whisper is not installed. Run: pip install faster-whisper') from exc
    return WhisperModel(model_size, device=device, compute_type=compute_type)


@lru_cache(maxsize=2)
def _get_transformers_asr_pipeline(model_path: str, device: str):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError('transformers is not installed. Run: pip install transformers torch') from exc

    device_arg = -1
    if device.lower().startswith('cuda'):
        device_arg = 0
    return pipeline('automatic-speech-recognition', model=model_path, device=device_arg, chunk_length_s=30)


def transcribe_file_trained(file_path: str | Path, language: str | None = None) -> tuple[str, dict[str, Any] | str]:
    """Local/trained transcription.

    - faster_whisper: free local Whisper inference. Good before you fine-tune.
    - transformers_whisper: uses your fine-tuned Whisper model in models/asr/.
    """
    settings = get_settings()
    backend = (settings.trained_transcription_backend or 'disabled').strip().lower()
    path = _validate_audio_path(file_path)
    lang = normalize_language(language or settings.default_language)

    if backend == 'disabled':
        raise RuntimeError('TRAINED_TRANSCRIPTION_BACKEND=disabled. Enable faster_whisper or transformers_whisper.')

    if backend == 'faster_whisper':
        model = _get_faster_whisper_model(
            settings.trained_transcription_model_size,
            settings.trained_asr_device,
            settings.trained_asr_compute_type,
        )
        segments, info = model.transcribe(
            str(path),
            language=None if lang == 'auto' else lang,
            vad_filter=True,
        )
        text_parts: list[str] = []
        segment_data: list[dict[str, Any]] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                text_parts.append(text)
            segment_data.append({'start': segment.start, 'end': segment.end, 'text': segment.text})
        text = '\n'.join(text_parts)
        return text, {
            'backend': 'faster_whisper',
            'model_size': settings.trained_transcription_model_size,
            'language': getattr(info, 'language', lang),
            'duration': getattr(info, 'duration', None),
            'segments': segment_data,
        }

    if backend == 'transformers_whisper':
        ensure_trained_asr_model(settings)
        model_path = settings.trained_asr_model_path
        if not Path(model_path).exists():
            raise RuntimeError(f'Trained ASR model not found: {model_path}. Train it first or use TRAINED_TRANSCRIPTION_BACKEND=faster_whisper.')
        asr = _get_transformers_asr_pipeline(model_path, settings.trained_asr_device)
        generate_kwargs: dict[str, Any] = {'task': 'transcribe'}
        if lang == 'ar':
            generate_kwargs['language'] = 'Arabic'
        elif lang == 'en':
            generate_kwargs['language'] = 'English'
        audio = _load_audio_for_transformers(path)
        result = asr(audio, generate_kwargs=generate_kwargs)
        text = result.get('text', '') if isinstance(result, dict) else str(result)
        return text, {'backend': 'transformers_whisper', 'model_path': model_path, 'raw': result}

    if backend == 'remote':
        if not settings.trained_remote_api_url:
            raise RuntimeError('TRAINED_REMOTE_API_URL is missing. Set it to your Colab tunnel URL.')
        url = settings.trained_remote_api_url.rstrip('/') + '/transcribe'
        with path.open('rb') as audio_file:
            response = requests.post(
                url,
                files={'file': (path.name, audio_file, 'application/octet-stream')},
                data={'language': lang},
                timeout=settings.cloud_request_timeout_seconds,
            )
        response.raise_for_status()
        data = response.json()
        text = data.get('text') or data.get('transcript') or ''
        if not text:
            raise RuntimeError('Remote trained ASR returned no transcript text.')
        return text, {'backend': 'remote', 'url': url, 'raw': data}

    raise ValueError('TRAINED_TRANSCRIPTION_BACKEND must be faster_whisper, transformers_whisper, remote, or disabled.')
