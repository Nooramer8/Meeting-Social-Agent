from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from groq import BadRequestError, Groq
from openai import OpenAI
from pydantic import TypeAdapter

from app.config import get_settings
from app.models import ActionItem, MeetingSummary
from app.services.model_downloader import ensure_trained_summary_model
from app.services.transcription import normalize_language


SUMMARY_SCHEMA: dict[str, Any] = MeetingSummary.model_json_schema()


SYSTEM_PROMPT = '''
You are a careful multilingual meeting assistant. Convert a meeting transcript into a concise internal recap and public-safe social media drafts.
Rules:
- Do not invent facts, names, deadlines, numbers, or outcomes.
- If something is unclear, say "Not specified" in English outputs or "غير محدد" in Arabic outputs, or omit it from lists when appropriate.
- Identify sensitive/confidential/private content that should be reviewed before public posting.
- Facebook can be a little more detailed.
- Instagram should be shorter, warmer, and include hashtags.
- Do not include private personal data, client secrets, financial details, legal issues, passwords, internal conflicts, or confidential strategy in public posts.
- Follow the requested output language for all user-facing fields: title, summary, key points, decisions, action items, warnings, posts, captions, and hashtags.
- If Arabic is requested, write natural Modern Standard Arabic or the Arabic style used in the transcript, preserve names accurately, and use Arabic-friendly hashtags when appropriate.
'''


LANGUAGE_NAMES = {
    'auto': 'the primary language of the transcript',
    'ar': 'Arabic',
    'en': 'English',
}

_JSON_OBJECT_HINT = (
    '\n\nReturn one JSON object only with keys: '
    'language, title, short_summary, key_points, decisions, '
    'action_items (array of objects with owner, task, deadline strings), '
    'risks_or_sensitive_items, suggested_facebook_post, '
    'suggested_instagram_caption, suggested_hashtags.'
)

AR_TRAINED_PROMPT = (
    'لخص الاجتماع بصيغة منظمة. أخرج الأقسام التالية فقط: '
    'العنوان، الملخص، النقاط الرئيسية، القرارات، المهام، تنبيهات للمراجعة، '
    'منشور فيسبوك، تعليق إنستغرام، الوسوم.\n'
    'النص:\n'
)
EN_TRAINED_PROMPT = (
    'Summarize the meeting in a structured format. Output only these sections: '
    'title, summary, key points, decisions, tasks, review warnings, Facebook post, Instagram caption, hashtags.\n'
    'Transcript:\n'
)


def _is_retryable_summary_error(exc: Exception) -> bool:
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


def summarize_transcript(
    transcript_text: str,
    target_language: str | None = None,
    backend: str | None = None,
) -> MeetingSummary:
    """Cloud summarization using Groq or OpenAI."""
    settings = get_settings()
    selected_backend = (backend or settings.summary_backend or 'groq').strip().lower()

    if selected_backend == 'groq':
        if not settings.groq_api_key:
            raise RuntimeError('GROQ_API_KEY is missing. Add it to .env, or set SUMMARY_BACKEND=openai.')
        client: Groq | OpenAI = Groq(api_key=settings.groq_api_key, timeout=settings.cloud_request_timeout_seconds)
    elif selected_backend == 'openai':
        if not settings.openai_api_key:
            raise RuntimeError('OPENAI_API_KEY is missing. Add it to .env, or set SUMMARY_BACKEND=groq.')
        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.cloud_request_timeout_seconds)
    else:
        raise ValueError('SUMMARY_BACKEND must be groq or openai.')

    transcript_for_model = transcript_text.strip()
    if not transcript_for_model:
        raise ValueError('Transcript is empty; cannot summarize.')

    lang_code = normalize_language(target_language or settings.default_language)
    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)

    user_content = (
        'Create a meeting summary and social drafts from this transcript.\n'
        f'Output language: {lang_name}.\n'
        'If the output language is Arabic, make the Facebook post and Instagram caption fully Arabic and suitable for right-to-left reading.\n\n'
        f'TRANSCRIPT:\n{transcript_for_model}'
    )
    messages: list[dict[str, str]] = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_content},
    ]

    attempts = max(1, settings.cloud_transcription_retries + 1)
    last_error: Exception | None = None
    response = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=settings.summary_model,
                messages=messages,
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'meeting_summary',
                        'schema': SUMMARY_SCHEMA,
                        'strict': True,
                    },
                },
                temperature=0.2,
            )
            break
        except BadRequestError as exc:
            err_text = str(exc).lower()
            if selected_backend != 'groq' or 'json_schema' not in err_text:
                raise
            response = client.chat.completions.create(
                model=settings.summary_model,
                messages=[messages[0], {'role': 'user', 'content': user_content + _JSON_OBJECT_HINT}],
                response_format={'type': 'json_object'},
                temperature=0.2,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt >= attempts or not _is_retryable_summary_error(exc):
                raise
            time.sleep(max(0.0, settings.cloud_transcription_retry_delay_seconds))

    if response is None:
        raise RuntimeError('Cloud summary failed without returning a response.') from last_error

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError('The summary model returned an empty response.')

    data = json.loads(content)
    return TypeAdapter(MeetingSummary).validate_python(data)


# --------------------------
# Local/trained summarization
# --------------------------


def _looks_arabic(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text or ''))


def _clean_preserve_lines(text: str) -> str:
    text = text or ''
    text = re.sub(r'<extra_id_\d+>', ' ', text)
    text = re.sub(r'</?s>|<pad>|<unk>', ' ', text, flags=re.IGNORECASE)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\t ]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip(' \n\t')


def _clean_inline(text: str) -> str:
    text = _clean_preserve_lines(text)
    text = re.sub(r'[`*_]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip(' .،؛:|\n\t')
    return text.strip()


def _sentences(text: str) -> list[str]:
    text = _clean_inline(text)
    parts = re.split(r'(?<=[.!؟?])\s+|[\n\r]+', text.strip())
    clean = [p.strip(' -•\t،.') for p in parts if len(p.strip()) > 8]
    if not clean and text.strip():
        clean = [text.strip()[:600]]
    return clean


def _shorten(text: str, max_chars: int) -> str:
    text = _clean_inline(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + '…'


def _dedupe(items: list[str], limit: int = 6) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = _clean_inline(item).strip(' -•')
        if not item or item in {'غير محدد', 'Not specified'}:
            continue
        key = re.sub(r'\W+', '', item.lower())[:90]
        if key and key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _topic_title_ar(transcript: str, generated_summary: str) -> str:
    text = transcript + ' ' + generated_summary
    if 'فريق التصميم' in text or 'التصميم' in text:
        return 'ملخص اجتماع فريق التصميم'
    if 'التسويق' in text or 'حملة' in text or 'منشور' in text:
        return 'ملخص اجتماع التسويق'
    if 'المبيعات' in text:
        return 'ملخص اجتماع المبيعات'
    if 'المشروع' in text:
        return 'ملخص اجتماع المشروع'
    if 'العميل' in text or 'العملاء' in text:
        return 'ملخص اجتماع العملاء'
    return 'ملخص الاجتماع'


def _topic_title_en(transcript: str, generated_summary: str) -> str:
    text = (transcript + ' ' + generated_summary).lower()
    if 'design' in text:
        return 'Design team meeting summary'
    if 'marketing' in text or 'campaign' in text:
        return 'Marketing meeting summary'
    if 'sales' in text:
        return 'Sales meeting summary'
    if 'project' in text:
        return 'Project meeting summary'
    return 'Meeting summary'


def _extract_key_points_ar(summary_sentences: list[str], transcript_sentences: list[str]) -> list[str]:
    priority_words = (
        'ناقش', 'مناقشة', 'اتفق', 'تم الاتفاق', 'قرر', 'تحديد', 'إعداد', 'إنشاء', 'نشر',
        'التصميم', 'التسويق', 'إنستغرام', 'فيسبوك', 'تعليق', 'إعجاب', 'mentions', 'الفرق',
        'العميل', 'المشروع', 'المتابعين', 'التفاعل', 'خطة'
    )
    candidates = summary_sentences + [s for s in transcript_sentences if any(w in s for w in priority_words)] + transcript_sentences
    return _dedupe([_shorten(s, 180) for s in candidates], limit=5)


def _extract_decisions_ar(transcript_sentences: list[str]) -> list[str]:
    decision_keywords = ('قرر', 'قررنا', 'اتفق', 'تم الاتفاق', 'اعتمد', 'وافق', 'تم تحديد', 'تم اختيار')
    return _dedupe([_shorten(s, 180) for s in transcript_sentences if any(k in s for k in decision_keywords)], limit=5)


def _extract_actions_ar(transcript_sentences: list[str], unknown: str) -> list[ActionItem]:
    action_keywords = ('سوف', 'سيتم', 'سيقوم', 'ستقوم', 'يجب', 'مطلوب', 'المهمة', 'تكليف', 'تجهيز', 'إعداد', 'إنشاء', 'توثيق', 'مراجعة', 'رفع')
    raw = _dedupe([_shorten(s, 180) for s in transcript_sentences if any(k in s for k in action_keywords)], limit=5)
    return [ActionItem(owner=unknown, task=s, deadline=unknown) for s in raw]


def _extract_key_points_en(summary_sentences: list[str], transcript_sentences: list[str]) -> list[str]:
    priority_words = ('discuss', 'agreed', 'decided', 'prepare', 'publish', 'facebook', 'instagram', 'design', 'marketing', 'customer', 'project')
    candidates = summary_sentences + [s for s in transcript_sentences if any(w in s.lower() for w in priority_words)] + transcript_sentences
    return _dedupe([_shorten(s, 180) for s in candidates], limit=5)


def _extract_decisions_en(transcript_sentences: list[str]) -> list[str]:
    decision_keywords = ('decided', 'agreed', 'approved', 'confirmed', 'selected')
    return _dedupe([_shorten(s, 180) for s in transcript_sentences if any(k in s.lower() for k in decision_keywords)], limit=5)


def _extract_actions_en(transcript_sentences: list[str], unknown: str) -> list[ActionItem]:
    action_keywords = ('will', 'should', 'must', 'task', 'owner', 'deadline', 'prepare', 'review', 'create', 'publish')
    raw = _dedupe([_shorten(s, 180) for s in transcript_sentences if any(k in s.lower() for k in action_keywords)], limit=5)
    return [ActionItem(owner=unknown, task=s, deadline=unknown) for s in raw]


def _is_bad_generated_summary(generated: str, transcript: str) -> bool:
    raw = generated or ''
    clean = _clean_inline(raw)
    if '<extra_id_' in raw:
        return True
    if len(clean) < 20:
        return True
    if _looks_arabic(transcript):
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', clean))
        if arabic_chars < 10:
            return True
    words = clean.split()
    if len(words) >= 8 and len(set(words)) <= max(3, len(words) // 4):
        return True
    return False


def _parse_bullets(block: str) -> list[str]:
    out: list[str] = []
    for line in block.splitlines():
        line = _clean_inline(re.sub(r'^[-•✅]\s*', '', line.strip()))
        if line and line not in {'غير محدد', 'Not specified'}:
            out.append(line)
    if not out and _clean_inline(block):
        out = [_clean_inline(block)]
    return _dedupe(out, limit=6)


def _parse_action_items(block: str, unknown: str) -> list[ActionItem]:
    items: list[ActionItem] = []
    for line in _parse_bullets(block):
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 3:
            owner, task, deadline = parts[0], parts[1], parts[2]
        else:
            owner, task, deadline = unknown, line, unknown
        if task and task not in {'غير محدد', 'Not specified'}:
            items.append(ActionItem(owner=owner or unknown, task=task, deadline=deadline or unknown))
    return items[:5]


def _parse_sectioned_summary(generated: str, transcript: str, lang_code: str) -> MeetingSummary | None:
    """Parse the better fine-tuned output with Arabic headings.

    The improved training script teaches the model this section format, so this
    parser uses the model's real output instead of ignoring it.
    """
    text = _clean_preserve_lines(generated)
    if not text or _is_bad_generated_summary(text, transcript):
        return None

    heading_map = {
        'العنوان': 'title',
        'الملخص': 'summary',
        'النقاط الرئيسية': 'key_points',
        'القرارات': 'decisions',
        'المهام': 'tasks',
        'تنبيهات للمراجعة': 'risks',
        'منشور فيسبوك': 'facebook',
        'تعليق إنستغرام': 'instagram',
        'الوسوم': 'hashtags',
        'title': 'title',
        'summary': 'summary',
        'key points': 'key_points',
        'decisions': 'decisions',
        'tasks': 'tasks',
        'review warnings': 'risks',
        'facebook post': 'facebook',
        'instagram caption': 'instagram',
        'hashtags': 'hashtags',
    }
    sections: dict[str, list[str]] = {v: [] for v in set(heading_map.values())}
    current: str | None = None
    heading_pattern = re.compile(r'^\s*([\u0600-\u06FFA-Za-z ]{3,30})\s*[:：]\s*(.*)$')

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = heading_pattern.match(line)
        if match:
            heading = match.group(1).strip().lower()
            rest = match.group(2).strip()
            key = heading_map.get(heading)
            if key:
                current = key
                if rest:
                    sections[current].append(rest)
                continue
        if current:
            sections[current].append(line)

    is_ar = lang_code == 'ar' or (lang_code == 'auto' and _looks_arabic(transcript + generated))
    unknown = 'غير محدد' if is_ar else 'Not specified'
    language_label = 'Arabic' if is_ar else 'English'

    title = _clean_inline(' '.join(sections.get('title', [])))
    summary = _clean_inline(' '.join(sections.get('summary', [])))
    key_points = _parse_bullets('\n'.join(sections.get('key_points', [])))
    decisions = _parse_bullets('\n'.join(sections.get('decisions', [])))
    action_items = _parse_action_items('\n'.join(sections.get('tasks', [])), unknown)
    risks = _parse_bullets('\n'.join(sections.get('risks', [])))
    facebook = _clean_preserve_lines('\n'.join(sections.get('facebook', [])))
    instagram = _clean_preserve_lines('\n'.join(sections.get('instagram', [])))
    hashtags = _parse_bullets('\n'.join(sections.get('hashtags', [])))

    if not summary and not key_points:
        return None

    if not title:
        title = _topic_title_ar(transcript, generated) if is_ar else _topic_title_en(transcript, generated)
    if not summary:
        summary = ' '.join(key_points[:2])
    if not risks:
        risks = ['يرجى مراجعة المحتوى قبل النشر للتأكد من عدم وجود معلومات خاصة أو سرية.'] if is_ar else ['Review before publishing to ensure no private or confidential information is included.']
    if not facebook:
        facebook = ('ملخص اجتماع اليوم:\n\n' if is_ar else 'Meeting recap:\n\n') + '\n'.join(f'• {p}' for p in (key_points or [summary]))
    if not instagram:
        instagram = ('ملخص سريع من اجتماع اليوم 👇\n\n' if is_ar else 'Quick recap from today’s meeting 👇\n\n') + '\n'.join(f'✅ {p}' for p in (key_points[:4] or [summary]))
    if not hashtags:
        hashtags = ['#ملخص_اجتماع', '#فريق_العمل'] if is_ar else ['#MeetingRecap', '#TeamUpdate']

    return MeetingSummary(
        language=language_label,
        title=_shorten(title, 100),
        short_summary=_shorten(summary, 650),
        key_points=key_points[:5] or [_shorten(summary, 180)],
        decisions=decisions[:5],
        action_items=action_items[:5],
        risks_or_sensitive_items=risks[:5],
        suggested_facebook_post=facebook,
        suggested_instagram_caption=instagram,
        suggested_hashtags=hashtags[:8],
    )


def _build_structured_summary_from_text(transcript: str, generated_summary: str, lang_code: str) -> MeetingSummary:
    parsed = _parse_sectioned_summary(generated_summary, transcript, lang_code)
    if parsed:
        return parsed

    generated_summary = _clean_inline(generated_summary)
    use_fallback = _is_bad_generated_summary(generated_summary, transcript)
    is_ar = lang_code == 'ar' or (lang_code == 'auto' and _looks_arabic(transcript + generated_summary))
    language_label = 'Arabic' if is_ar else 'English'
    unknown = 'غير محدد' if is_ar else 'Not specified'

    transcript_sentences = _sentences(transcript)
    summary_sentences = transcript_sentences[:3] if use_fallback else (_sentences(generated_summary) or transcript_sentences[:3])

    if is_ar:
        title = _topic_title_ar(transcript, generated_summary)
        short_summary = _shorten(' '.join(summary_sentences[:2]) if summary_sentences else 'تم تلخيص الاجتماع من النص المتاح.', 500)
        key_points = _extract_key_points_ar(summary_sentences, transcript_sentences)
        decisions = _extract_decisions_ar(transcript_sentences)
        action_items = _extract_actions_ar(transcript_sentences, unknown)
        risks = ['يرجى مراجعة المحتوى قبل النشر للتأكد من عدم وجود معلومات خاصة أو سرية.']
        facebook = 'ملخص اجتماع اليوم:\n\n' + '\n'.join(f'• {p}' for p in (key_points or [short_summary]))
        instagram = 'ملخص سريع من اجتماع اليوم 👇\n\n' + '\n'.join(f'✅ {p}' for p in (key_points[:4] or [short_summary])) + '\n\n#ملخص_اجتماع #فريق_العمل'
        hashtags = ['#ملخص_اجتماع', '#فريق_العمل']
    else:
        title = _topic_title_en(transcript, generated_summary)
        short_summary = _shorten(' '.join(summary_sentences[:2]) if summary_sentences else 'The meeting was summarized from the available transcript.', 500)
        key_points = _extract_key_points_en(summary_sentences, transcript_sentences)
        decisions = _extract_decisions_en(transcript_sentences)
        action_items = _extract_actions_en(transcript_sentences, unknown)
        risks = ['Review before publishing to ensure no private or confidential information is included.']
        facebook = 'Meeting recap:\n\n' + '\n'.join(f'• {p}' for p in (key_points or [short_summary]))
        instagram = 'Quick recap from today’s meeting 👇\n\n' + '\n'.join(f'✅ {p}' for p in (key_points[:4] or [short_summary])) + '\n\n#MeetingRecap #TeamUpdate'
        hashtags = ['#MeetingRecap', '#TeamUpdate']

    return MeetingSummary(
        language=language_label,
        title=title,
        short_summary=short_summary or unknown,
        key_points=key_points or [short_summary or unknown],
        decisions=decisions,
        action_items=action_items,
        risks_or_sensitive_items=risks,
        suggested_facebook_post=facebook,
        suggested_instagram_caption=instagram,
        suggested_hashtags=hashtags,
    )


def _rule_based_summary(transcript_text: str, lang_code: str) -> MeetingSummary:
    sentences = _sentences(transcript_text)
    generated = ' '.join(sentences[:3]) if sentences else transcript_text[:1000]
    return _build_structured_summary_from_text(transcript_text, generated, lang_code)


@lru_cache(maxsize=2)
def _get_seq2seq_model(model_path: str, device: str):
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('transformers and torch are required. Run: pip install transformers torch sentencepiece') from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    if device.lower().startswith('cuda') and torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    return tokenizer, model


def summarize_transcript_trained(transcript_text: str, target_language: str | None = None) -> MeetingSummary:
    """Local/trained summarization.

    The better training script teaches the model to generate structured Arabic
    sections. This function parses those sections and returns the same schema as
    the Groq panel. If the model output is weak, it falls back to deterministic
    extraction from the transcript.
    """
    settings = get_settings()
    backend = (settings.trained_summary_backend or 'disabled').strip().lower()
    lang_code = normalize_language(target_language or settings.default_language)
    transcript = transcript_text.strip()
    if not transcript:
        raise ValueError('Transcript is empty; cannot summarize.')

    if backend == 'disabled':
        raise RuntimeError('TRAINED_SUMMARY_BACKEND=disabled. Enable rule_based or transformers_seq2seq.')

    if backend == 'rule_based':
        return _rule_based_summary(transcript, lang_code)

    if backend == 'transformers_seq2seq':
        ensure_trained_summary_model(settings)
        model_path = settings.trained_summary_model_path
        if not Path(model_path).exists():
            raise RuntimeError(f'Trained summarizer model not found: {model_path}. Train it first or use TRAINED_SUMMARY_BACKEND=rule_based.')
        tokenizer, model = _get_seq2seq_model(model_path, settings.trained_summary_device)
        prefix = AR_TRAINED_PROMPT if lang_code in {'ar', 'auto'} else EN_TRAINED_PROMPT
        inputs = tokenizer(prefix + transcript, return_tensors='pt', truncation=True, max_length=1024)
        if settings.trained_summary_device.lower().startswith('cuda'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            min_new_tokens=40,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.18,
            length_penalty=1.0,
            early_stopping=True,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        return _build_structured_summary_from_text(transcript, generated, lang_code)

    raise ValueError('TRAINED_SUMMARY_BACKEND must be rule_based, transformers_seq2seq, or disabled.')
