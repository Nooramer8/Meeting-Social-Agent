# Build the two “our training” models

This project has two local/custom model slots:

1. **Our training speech-to-text model**
   - Purpose: turn Arabic meeting audio into text.
   - Training script: `training/train_asr_whisper.py`
   - Output folder: `models/asr/whisper-ar-meetings`

2. **Our training summarization model**
   - Purpose: turn meeting transcript into summary + social draft source.
   - Training script: `training/train_summarizer.py`
   - Output folder: `models/summarizer/meeting-summary-ar`

The FastAPI dashboard compares these with Groq:

```text
Speech to text from Groq        vs Speech to text from our training
Summary from Groq               vs Summary from our training
```

## Important

I can give you the complete training pipeline, but a real trained model requires your data. Without your meeting recordings, transcripts, and summaries, any model would be fake or useless.

## Dataset 1: speech-to-text / ASR

Create this file:

```text
data/asr/train.jsonl
data/asr/eval.jsonl
```

Each line:

```json
{"audio": "data/asr/audio/meeting_001.wav", "text": "مرحبا بكم في اجتماع اليوم..."}
```

Use corrected human transcripts for best quality.

## Dataset 2: summarization

Create this file:

```text
data/summarization/train.jsonl
data/summarization/eval.jsonl
```

Each line:

```json
{"transcript": "نص الاجتماع الكامل...", "summary": "ملخص الاجتماع..."}
```

## Fast path: use Groq as a teacher, then correct the data

Put recordings here:

```text
data/raw_recordings/
```

Then run:

```powershell
$env:GROQ_API_KEY="your_groq_key"
python training/create_teacher_dataset_from_recordings.py --recordings_dir data/raw_recordings --language ar
python training/split_jsonl.py --input data/asr/all.jsonl --output_dir data/asr
python training/split_jsonl.py --input data/summarization/all.jsonl --output_dir data/summarization
```

Now review and correct the generated `train.jsonl` and `eval.jsonl` files. This is important because teacher data can contain mistakes.

## Train both models in one command

```powershell
python -m pip install -r requirements-training.txt
python training/validate_training_data.py
python training/train_our_two_models.py --asr_epochs 3 --sum_epochs 3 --batch_size 2
```

This creates:

```text
models/asr/whisper-ar-meetings
models/summarizer/meeting-summary-ar
models/MODEL_STATUS.json
```

## Use your trained models in the app

Edit `.env`:

```env
TRAINED_TRANSCRIPTION_BACKEND=transformers_whisper
TRAINED_ASR_MODEL_PATH=models/asr/whisper-ar-meetings

TRAINED_SUMMARY_BACKEND=transformers_seq2seq
TRAINED_SUMMARY_MODEL_PATH=models/summarizer/meeting-summary-ar

DRAFT_SOURCE=trained
```

Restart FastAPI:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

## Hardware recommendation

For a first test, use `openai/whisper-small` and `google/mt5-small`. CPU training will be slow. A GPU is strongly recommended for real training.

## Minimum data recommendation

For a proof of concept:

```text
ASR: 20–50 short meeting clips with corrected transcripts
Summary: 50–200 transcript-summary examples
```

For better production quality:

```text
ASR: 100+ hours of Arabic meeting audio
Summary: 1,000+ transcript-summary examples
```
