from pathlib import Path
from uuid import uuid4
import shutil
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db import create_meeting, get_meeting, init_db, list_meetings, update_meeting
from app.models import DraftUpdate, MeetingSummary
from app.services.image_generator import create_instagram_recap_image, public_url_for_generated_file
from app.services.meta_publisher import publish_facebook_post, publish_instagram_image
from app.services.model_downloader import ensure_trained_models
from app.services.summarizer import summarize_transcript, summarize_transcript_trained
from app.services.transcription import SUPPORTED_EXTENSIONS, normalize_language, transcribe_file, transcribe_file_trained


settings = get_settings()
app = FastAPI(title='Meeting Social Agent', version='0.2.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory=Path(__file__).parent / 'static'), name='static')
app.mount('/storage/generated', StaticFiles(directory=settings.generated_path), name='generated')


@app.on_event('startup')
def startup() -> None:
    ensure_trained_models(settings)
    init_db()


@app.get('/')
def index() -> FileResponse:
    return FileResponse(
        Path(__file__).parent / 'static' / 'index.html',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
        },
    )


@app.get('/favicon.ico', include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get('/health')
def health() -> dict[str, str]:
    return {
        'status': 'ok',
        'transcription_backend': settings.transcription_backend,
        'trained_transcription_backend': settings.trained_transcription_backend,
        'summary_backend': settings.summary_backend,
        'trained_summary_backend': settings.trained_summary_backend,
        'draft_source': settings.draft_source,
    }


@app.post('/api/meetings')
def upload_meeting(file: UploadFile = File(...), language: str = Form('auto')) -> dict:
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f'Unsupported file type. Use one of: {sorted(SUPPORTED_EXTENSIONS)}')

    lang = normalize_language(language)
    meeting_id = uuid4().hex
    safe_name = Path(file.filename or f'meeting{suffix}').name
    target = settings.uploads_path / f'{meeting_id}-{safe_name}'
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open('wb') as out:
        shutil.copyfileobj(file.file, out)

    return create_meeting(meeting_id=meeting_id, filename=safe_name, upload_path=str(target), language=lang)


@app.get('/api/meetings')
def meetings() -> list[dict]:
    return list_meetings()


@app.get('/api/meetings/{meeting_id}')
def meeting_detail(meeting_id: str) -> dict:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail='Meeting not found')
    return meeting


def _summary_or_none(value: MeetingSummary | None) -> dict | None:
    return value.model_dump() if value else None


def _choose_source(groq_value, trained_value, preferred: str):
    preferred = (preferred or 'groq').strip().lower()
    if preferred == 'trained':
        return ('trained', trained_value) if trained_value else ('groq', groq_value)
    return ('groq', groq_value) if groq_value else ('trained', trained_value)


def _ensure_social_drafts(summary: MeetingSummary, language: str) -> tuple[str, str]:
    facebook = (summary.suggested_facebook_post or '').strip()
    instagram = (summary.suggested_instagram_caption or '').strip()
    if facebook and instagram:
        return facebook, instagram

    lang = normalize_language(language)
    is_ar = lang == 'ar' or any('\u0600' <= ch <= '\u06FF' for ch in (summary.short_summary or '') + (summary.title or ''))
    points = [p.strip() for p in (summary.key_points or []) if p and p.strip()]
    top_points = points[:4]

    if not facebook:
        if is_ar:
            facebook = 'ملخص اجتماع اليوم:\n\n' + '\n'.join(f'• {p}' for p in (top_points or [summary.short_summary or summary.title]))
        else:
            facebook = 'Meeting recap:\n\n' + '\n'.join(f'• {p}' for p in (top_points or [summary.short_summary or summary.title]))

    if not instagram:
        if is_ar:
            instagram = (
                'ملخص سريع من الاجتماع:\n\n'
                + '\n'.join(f'• {p}' for p in (top_points[:3] or [summary.short_summary or summary.title]))
                + '\n\n#ملخص_اجتماع #فريق_العمل'
            )
        else:
            instagram = (
                'Quick meeting recap:\n\n'
                + '\n'.join(f'• {p}' for p in (top_points[:3] or [summary.short_summary or summary.title]))
                + '\n\n#MeetingRecap #TeamUpdate'
            )

    return facebook.strip(), instagram.strip()


@app.post('/api/meetings/{meeting_id}/process')
def process_meeting(meeting_id: str, mode: str = 'both') -> dict:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail='Meeting not found')
    if meeting.get('status') in {'transcribing_groq', 'transcribing_trained', 'summarizing_groq', 'summarizing_trained'}:
        raise HTTPException(status_code=409, detail='This meeting is already processing. Please wait for it to finish.')

    language = meeting.get('language') or settings.default_language
    mode_value = (mode or 'both').strip().lower()
    if mode_value not in {'both', 'groq', 'trained'}:
        raise HTTPException(status_code=400, detail='mode must be one of: both, groq, trained')

    errors: dict[str, str] = {}

    groq_transcript_text = None
    groq_transcript_json = None
    trained_transcript_text = None
    trained_transcript_json = None
    groq_summary: MeetingSummary | None = None
    trained_summary: MeetingSummary | None = None

    try:
        # 1) Speech-to-text from selected pipeline(s).
        if mode_value in {'both', 'groq'}:
            try:
                update_meeting(meeting_id, status='transcribing_groq', error=None, comparison_errors=None)
                groq_transcript_text, groq_transcript_json = transcribe_file(
                    meeting['upload_path'],
                    language=language,
                    backend=settings.transcription_backend,
                )
                update_meeting(
                    meeting_id,
                    groq_transcript_text=groq_transcript_text,
                    groq_transcript_json=groq_transcript_json,
                )
            except Exception as exc:
                errors['groq_transcription'] = str(exc)
                print('========== GROQ/CLOUD TRANSCRIPTION ERROR ==========')
                print(traceback.format_exc())

        if mode_value in {'both', 'trained'}:
            try:
                update_meeting(meeting_id, status='transcribing_trained', comparison_errors=errors or None)
                trained_transcript_text, trained_transcript_json = transcribe_file_trained(meeting['upload_path'], language=language)
                update_meeting(
                    meeting_id,
                    trained_transcript_text=trained_transcript_text,
                    trained_transcript_json=trained_transcript_json,
                )
            except Exception as exc:
                errors['trained_transcription'] = str(exc)
                print('========== TRAINED TRANSCRIPTION ERROR ==========')
                print(traceback.format_exc())

        if mode_value == 'both' and not groq_transcript_text and not trained_transcript_text:
            raise RuntimeError('Both transcription pipelines failed. Check comparison_errors for details.')
        if mode_value == 'groq' and not groq_transcript_text:
            raise RuntimeError('Groq transcription pipeline failed. Check comparison_errors for details.')
        if mode_value == 'trained' and not trained_transcript_text:
            raise RuntimeError('Trained transcription pipeline failed. Check comparison_errors for details.')

        # 2) Summary from selected pipeline(s).
        if mode_value in {'both', 'groq'} and groq_transcript_text:
            try:
                update_meeting(meeting_id, status='summarizing_groq', comparison_errors=errors or None)
                groq_summary = summarize_transcript(groq_transcript_text, target_language=language, backend=settings.summary_backend)
                update_meeting(meeting_id, groq_summary_json=groq_summary.model_dump())
            except Exception as exc:
                errors['groq_summary'] = str(exc)
                print('========== GROQ/CLOUD SUMMARY ERROR ==========')
                print(traceback.format_exc())

        if mode_value in {'both', 'trained'} and trained_transcript_text:
            try:
                update_meeting(meeting_id, status='summarizing_trained', comparison_errors=errors or None)
                trained_summary = summarize_transcript_trained(trained_transcript_text, target_language=language)
                update_meeting(meeting_id, trained_summary_json=trained_summary.model_dump())
            except Exception as exc:
                errors['trained_summary'] = str(exc)
                print('========== TRAINED SUMMARY ERROR ==========')
                print(traceback.format_exc())

        if mode_value == 'both' and not groq_summary and not trained_summary:
            raise RuntimeError('Both summary pipelines failed. Check comparison_errors for details.')
        if mode_value == 'groq' and not groq_summary:
            raise RuntimeError('Groq summary pipeline failed. Check comparison_errors for details.')
        if mode_value == 'trained' and not trained_summary:
            raise RuntimeError('Trained summary pipeline failed. Check comparison_errors for details.')

        selected_source, selected_summary = _choose_source(groq_summary, trained_summary, settings.draft_source)
        if mode_value == 'groq':
            selected_source, selected_summary = 'groq', groq_summary
        elif mode_value == 'trained':
            selected_source, selected_summary = 'trained', trained_summary

        if not selected_summary:
            raise RuntimeError('No summary was generated for the selected mode.')

        selected_transcript = groq_transcript_text if selected_source == 'groq' else trained_transcript_text
        selected_transcript_json = groq_transcript_json if selected_source == 'groq' else trained_transcript_json

        image_path = create_instagram_recap_image(selected_summary)
        facebook_post, instagram_caption = _ensure_social_drafts(selected_summary, language)

        updated = update_meeting(
            meeting_id,
            status='draft_ready',
            selected_source=selected_source,
            transcript_text=selected_transcript,
            transcript_json=selected_transcript_json,
            summary_json=selected_summary.model_dump(),
            facebook_post=facebook_post,
            instagram_caption=instagram_caption,
            instagram_image_path=image_path,
            approved=False,
            comparison_errors=errors or None,
            error='; '.join(f'{k}: {v}' for k, v in errors.items()) if errors else None,
        )
        return updated or {}
    except Exception as exc:
        errors['process'] = str(exc)
        update_meeting(meeting_id, status='error', error=str(exc), comparison_errors=errors)
        raise HTTPException(
            status_code=500,
            detail={
                'message': str(exc),
                'comparison_errors': errors,
            },
        ) from exc


@app.patch('/api/meetings/{meeting_id}/draft')
def update_draft(meeting_id: str, payload: DraftUpdate) -> dict:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail='Meeting not found')

    fields: dict = {'approved': False}
    if payload.facebook_post is not None:
        fields['facebook_post'] = payload.facebook_post
    if payload.instagram_caption is not None:
        fields['instagram_caption'] = payload.instagram_caption

    updated = update_meeting(meeting_id, **fields)
    return updated or {}


@app.post('/api/meetings/{meeting_id}/approve')
def approve_meeting(meeting_id: str) -> dict:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail='Meeting not found')
    if not meeting.get('facebook_post') and not meeting.get('instagram_caption'):
        raise HTTPException(status_code=400, detail='Process the meeting and create drafts before approval')
    updated = update_meeting(meeting_id, approved=True, status='approved')
    return updated or {}


@app.post('/api/meetings/{meeting_id}/publish/facebook')
def publish_facebook(meeting_id: str) -> dict:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail='Meeting not found')
    if not meeting.get('approved'):
        raise HTTPException(status_code=400, detail='Draft must be approved before publishing')
    if not meeting.get('facebook_post'):
        raise HTTPException(status_code=400, detail='Facebook post draft is empty')

    try:
        result = publish_facebook_post(meeting['facebook_post'])
        post_id = result.get('id')
        update_meeting(meeting_id, facebook_post_id=post_id, status='published_facebook')
        return {'platform': 'facebook', 'remote_id': post_id, 'raw_response': result}
    except Exception as exc:
        update_meeting(meeting_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/api/meetings/{meeting_id}/publish/instagram')
def publish_instagram(meeting_id: str) -> dict:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail='Meeting not found')
    if not meeting.get('approved'):
        raise HTTPException(status_code=400, detail='Draft must be approved before publishing')
    if not meeting.get('instagram_caption'):
        raise HTTPException(status_code=400, detail='Instagram caption draft is empty')
    if not meeting.get('instagram_image_path'):
        raise HTTPException(status_code=400, detail='Instagram image is missing')

    try:
        image_url = public_url_for_generated_file(meeting['instagram_image_path'])
        result = publish_instagram_image(image_url=image_url, caption=meeting['instagram_caption'])
        media_id = result.get('published', {}).get('id')
        update_meeting(meeting_id, instagram_media_id=media_id, status='published_instagram')
        return {'platform': 'instagram', 'remote_id': media_id, 'raw_response': result}
    except Exception as exc:
        update_meeting(meeting_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
