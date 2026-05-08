"""Generate synthetic TTS audio files from an ASR JSONL manifest.

Each input line should contain at least:
  {"id": "meeting_001", "audio": "data/asr/audio/meeting_001.mp3", "text": "..."}
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import edge_tts


DEFAULT_VOICE = "ar-SA-HamedNeural"
MOJIBAKE_MARKERS = ("Ø", "Ù", "Û")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MP3 files from a JSONL ASR dataset using edge-tts."
    )
    parser.add_argument("--input", required=True, help="Input JSONL manifest.")
    parser.add_argument("--out_dir", required=True, help="Directory for generated MP3s.")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="edge-tts voice name.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files that already exist.",
    )
    parser.add_argument(
        "--no-fix-mojibake",
        action="store_true",
        help="Do not try to repair Arabic text that was decoded as Windows-1252.",
    )
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    if path.exists():
        return path

    fallback = path.parent / "audio" / path.name
    if fallback.exists():
        print(f"Input not found at {path}; using {fallback} instead.")
        return fallback

    raise FileNotFoundError(f"Input JSONL not found: {path}")


def fix_mojibake(text: str) -> str:
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text

    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text

    return repaired if repaired else text


def output_path(record: dict[str, Any], out_dir: Path) -> Path:
    audio = record.get("audio")
    if isinstance(audio, str) and audio.strip():
        name = Path(audio).with_suffix(".mp3").name
    else:
        record_id = str(record.get("id") or "sample").strip() or "sample"
        name = f"{record_id}.mp3"

    return out_dir / name


def load_records(path: Path, repair_text: bool) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            text = str(record.get("text") or "").strip()
            if not text:
                print(f"Skipping line {line_number}: missing text.")
                continue

            if repair_text:
                record["text"] = fix_mojibake(text)

            records.append((line_number, record))

    return records


async def generate_record(
    line_number: int,
    record: dict[str, Any],
    out_dir: Path,
    voice: str,
    overwrite: bool,
) -> bool:
    destination = output_path(record, out_dir)
    if destination.exists() and not overwrite:
        print(f"Skipping {destination}: already exists.")
        return False

    text = str(record["text"]).strip()
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(destination))
    except Exception as exc:
        raise RuntimeError(f"Failed on line {line_number} ({destination}): {exc}") from exc

    print(f"Wrote {destination}")
    return True


async def main() -> None:
    args = parse_args()
    input_path = resolve_input(Path(args.input))
    out_dir = Path(args.out_dir)
    records = load_records(input_path, repair_text=not args.no_fix_mojibake)

    written = 0
    for line_number, record in records:
        if await generate_record(line_number, record, out_dir, args.voice, args.overwrite):
            written += 1

    print(f"Done. Generated {written} file(s); {len(records) - written} skipped.")


if __name__ == "__main__":
    asyncio.run(main())
