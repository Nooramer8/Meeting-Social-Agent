from __future__ import annotations

import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests

from app.config import Settings


def _has_model_files(path: Path) -> bool:
    return path.exists() and any(path.glob('config.json')) and any(path.iterdir())


def _download_file(url: str, target: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target.open('wb') as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = ''.join(archive.suffixes).lower()
    if suffixes.endswith('.zip'):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
        return
    if suffixes.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(archive) as tf:
            tf.extractall(destination)
        return
    raise RuntimeError(f'Unsupported model archive format: {archive.name}. Use .zip, .tar, .tar.gz, or .tgz.')


def _download_huggingface_repo(repo_id: str, destination: Path, token: str | None, revision: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError('huggingface-hub is required to download trained models from Hugging Face.') from exc

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision or None,
        token=token or None,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )


def _ensure_model(destination: Path, archive_url: str | None, hf_repo_id: str | None, token: str | None, revision: str | None) -> None:
    if _has_model_files(destination):
        return
    if hf_repo_id:
        _download_huggingface_repo(hf_repo_id, destination, token, revision)
        return
    if archive_url:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / archive_url.split('/')[-1].split('?')[0]
            _download_file(archive_url, archive)
            if destination.exists():
                shutil.rmtree(destination)
            _extract_archive(archive, destination)
        return


def ensure_trained_asr_model(settings: Settings) -> None:
    """Download the trained ASR model only when the trained ASR pipeline is used."""
    if (settings.trained_transcription_backend or 'disabled').strip().lower() == 'transformers_whisper':
        _ensure_model(
            Path(settings.trained_asr_model_path),
            settings.trained_asr_archive_url,
            settings.trained_asr_hf_repo_id,
            settings.huggingface_token,
            settings.trained_asr_hf_revision,
        )


def ensure_trained_summary_model(settings: Settings) -> None:
    """Download the trained summarizer only when the trained summary pipeline is used."""
    if (settings.trained_summary_backend or 'disabled').strip().lower() == 'transformers_seq2seq':
        _ensure_model(
            Path(settings.trained_summary_model_path),
            settings.trained_summary_archive_url,
            settings.trained_summary_hf_repo_id,
            settings.huggingface_token,
            settings.trained_summary_hf_revision,
        )


def ensure_trained_models(settings: Settings) -> None:
    """Download all trained model artifacts. Prefer lazy per-pipeline calls in web deployments."""
    ensure_trained_asr_model(settings)
    ensure_trained_summary_model(settings)
