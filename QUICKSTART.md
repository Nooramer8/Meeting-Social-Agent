# Quickstart

```bash
cd meeting_social_agent
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000`.

For Arabic meetings, choose **Arabic / العربية** before upload, or set this in `.env`:

```bash
DEFAULT_LANGUAGE=ar
```
