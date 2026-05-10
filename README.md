# Meeting Social Agent

A working MVP that:

1. Uploads a recorded meeting audio/video file.
2. Transcribes it with OpenAI.
3. Extracts summary, key points, decisions, action items, risks, and social post drafts.
4. Generates a simple Instagram recap image.
5. Requires human approval.
6. Publishes to a Facebook Page and Instagram professional account through Meta APIs.

## Arabic support

Arabic meetings are supported.

Before upload, choose **Arabic / العربية** in the language dropdown. The app will:

- Send `language=ar` as a transcription hint.
- Write the summary, key points, decisions, action items, Facebook post, Instagram caption, and hashtags in Arabic.
- Display Arabic text right-to-left in the dashboard.
- Generate an Arabic right-to-left Instagram recap image.

You can also set the default language in `.env`:

```bash
DEFAULT_LANGUAGE=ar
```

Use `DEFAULT_LANGUAGE=auto` if you want the app to follow the primary language of the transcript.

## Groq timeout tuning

For larger recordings or slow network responses, increase the cloud request timeout and retry count in `.env`:

```bash
CLOUD_REQUEST_TIMEOUT_SECONDS=180
CLOUD_TRANSCRIPTION_RETRIES=2
CLOUD_TRANSCRIPTION_RETRY_DELAY_SECONDS=2
```

## Important safety choice

This app intentionally requires approval before publishing. Meeting recordings often contain private or confidential information.

## Requirements

- Python 3.10+
- An OpenAI API key
- For publishing:
  - A Meta developer app
  - A Facebook Page ID and Page access token
  - An Instagram professional account connected through Meta
  - A public HTTPS deployment URL for Instagram image publishing

## Setup

```bash
cd meeting_social_agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your keys.

## Run locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

## Free hosting (Render)

This repo now includes `render.yaml` for one-click free deployment on Render.

Important for free hosting:

- This app should run in cloud-only mode on free tier.
- Keep large local training assets out of git (`models/`, `data/`, checkpoints).
- Add your real API keys in Render environment variables.

### 1) Push code to GitHub

```bash
cd meeting_social_agent
git init
git add .
git commit -m "Deploy-ready setup for Render free hosting"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### 2) Deploy from Render

1. Open Render dashboard and click **New +** -> **Blueprint**.
2. Connect your GitHub repository.
3. Render detects `render.yaml` and creates the web service.
4. In Render service settings, add env vars:
   - `GROQ_API_KEY`
   - `PUBLIC_BASE_URL` = your Render URL (for example `https://meeting-social-agent.onrender.com`)
   - Optional Meta vars for publishing:
     - `FACEBOOK_PAGE_ID`
     - `FACEBOOK_PAGE_ACCESS_TOKEN`
     - `INSTAGRAM_USER_ID`
     - `INSTAGRAM_ACCESS_TOKEN`
5. Redeploy after saving env vars.

### 3) Open your live app

Use the Render URL shown in your service, for example:

```text
https://meeting-social-agent.onrender.com
```

## Using the app

1. Choose the meeting language: **Auto-detect**, **Arabic**, or **English**.
2. Upload an `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, or `.webm` meeting file.
3. Click **Process meeting**.
4. Review the transcript-derived summary and social drafts.
5. Edit the drafts if needed.
6. Click **Approve drafts**.
7. Click **Publish to Facebook** and/or **Publish to Instagram**.

## Instagram note

The Instagram Content Publishing API needs an image or video URL that Meta can fetch from the public internet. Localhost URLs will not work for Instagram publishing. Deploy this app somewhere public, set `PUBLIC_BASE_URL`, and make sure `/storage/generated/...` files are reachable.

## Deploy trained models without committing them

Keep `models/` out of Git. For production, upload each final model folder to Hugging Face or as an archive on S3/R2, then let the app download it at startup.

Hugging Face example:

```text
TRAINED_TRANSCRIPTION_BACKEND=transformers_whisper
TRAINED_ASR_HF_REPO_ID=your-user/whisper-ar-meetings
TRAINED_ASR_MODEL_PATH=models/asr/whisper-ar-meetings

TRAINED_SUMMARY_BACKEND=transformers_seq2seq
TRAINED_SUMMARY_HF_REPO_ID=your-user/meeting-summary-ar
TRAINED_SUMMARY_MODEL_PATH=models/summarizer/meeting-summary-ar

HUGGINGFACE_TOKEN=hf_your_token_if_the_repos_are_private
```

S3/R2 archive example:

```text
TRAINED_TRANSCRIPTION_BACKEND=transformers_whisper
TRAINED_ASR_ARCHIVE_URL=https://your-bucket.example.com/whisper-ar-meetings.zip
TRAINED_ASR_MODEL_PATH=models/asr/whisper-ar-meetings

TRAINED_SUMMARY_BACKEND=transformers_seq2seq
TRAINED_SUMMARY_ARCHIVE_URL=https://your-bucket.example.com/meeting-summary-ar.zip
TRAINED_SUMMARY_MODEL_PATH=models/summarizer/meeting-summary-ar
```

Each archive should contain the model files directly, such as `config.json`, tokenizer files, and `model.safetensors`.

## API endpoints

```text
GET    /health
POST   /api/meetings
GET    /api/meetings
GET    /api/meetings/{meeting_id}
POST   /api/meetings/{meeting_id}/process
PATCH  /api/meetings/{meeting_id}/draft
POST   /api/meetings/{meeting_id}/approve
POST   /api/meetings/{meeting_id}/publish/facebook
POST   /api/meetings/{meeting_id}/publish/instagram
```

### Upload with Arabic language through API

```bash
curl -X POST http://localhost:8000/api/meetings \\
  -F "language=ar" \\
  -F "file=@meeting.m4a"
```

## Project structure

```text
app/
  main.py
  config.py
  db.py
  models.py
  services/
    image_generator.py
    meta_publisher.py
    summarizer.py
    transcription.py
  static/
    index.html
storage/
  uploads/
  generated/
```

## Production checklist

Before using this with real meetings:

- Add user authentication.
- Add role-based approval permissions.
- Store recordings in private object storage, not local disk.
- Add file size validation and audio chunking for long meetings.
- Add PII/confidentiality checks before creating public posts.
- Add queue workers for long transcription jobs.
- Add webhook-based upload from Zoom, Google Meet, or Teams.
- Add retries and audit logs for publishing.
- Put Meta tokens in a secret manager.


# Synthetic Arabic Meeting Dataset Examples

This package contains **30 short Arabic meeting examples** for your Groq-vs-our-training social media agent.

It includes:

- ASR / speech-to-text JSONL files
- Summarization JSONL files
- Train / eval / test splits
- Optional script to generate synthetic Arabic MP3 clips using free TTS

## Important

These are synthetic examples. They are good for testing your pipeline and understanding the data format. They are **not enough** to train a high-quality production model.

For real training quality, replace the synthetic audio with real meeting clips and human-corrected transcripts/summaries.

## Files

```text
data/asr/train.jsonl
data/asr/eval.jsonl
data/asr/test.jsonl
data/asr/all.jsonl

data/summarization/train.jsonl
data/summarization/eval.jsonl
data/summarization/test.jsonl
data/summarization/all.jsonl
```

## ASR format

```json
{"id":"meeting_001","language":"ar","audio":"data/asr/audio/meeting_001.mp3","text":"النص الصحيح للمقطع الصوتي"}
```

## Summarization format

```json
{
  "id":"meeting_001",
  "language":"ar",
  "title":"عنوان الاجتماع",
  "transcript":"النص الكامل",
  "summary":"الملخص الصحيح",
  "key_points":["نقطة 1", "نقطة 2"],
  "facebook_post":"منشور فيسبوك",
  "instagram_caption":"تعليق إنستغرام"
}
```

## Generate synthetic Arabic audio clips

From this folder, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-synthetic-audio.txt
python tools/generate_tts_audio.py --input data/asr/all.jsonl --out_dir data/asr/audio
```

This will create MP3 files such as:

```text
data/asr/audio/meeting_001.mp3
```

## Use with your project

Copy the `data/asr` and `data/summarization` folders into your training-ready agent project. Then run your validation and training scripts.
