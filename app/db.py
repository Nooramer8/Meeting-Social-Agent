import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import get_settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = '''
CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    upload_path TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'auto',
    status TEXT NOT NULL,
    transcript_text TEXT,
    transcript_json TEXT,
    selected_source TEXT,
    groq_transcript_text TEXT,
    groq_transcript_json TEXT,
    trained_transcript_text TEXT,
    trained_transcript_json TEXT,
    summary_json TEXT,
    groq_summary_json TEXT,
    trained_summary_json TEXT,
    facebook_post TEXT,
    instagram_caption TEXT,
    instagram_image_path TEXT,
    approved INTEGER NOT NULL DEFAULT 0,
    facebook_post_id TEXT,
    instagram_media_id TEXT,
    error TEXT,
    comparison_errors TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
'''


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    db_path = Path(get_settings().database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row['name'] for row in conn.execute(f'PRAGMA table_info({table})').fetchall()}
    if column not in existing:
        conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Lightweight migration for users upgrading from the first MVP.
        _ensure_column(conn, 'meetings', 'language', "TEXT NOT NULL DEFAULT 'auto'")
        _ensure_column(conn, 'meetings', 'selected_source', 'TEXT')
        _ensure_column(conn, 'meetings', 'groq_transcript_text', 'TEXT')
        _ensure_column(conn, 'meetings', 'groq_transcript_json', 'TEXT')
        _ensure_column(conn, 'meetings', 'trained_transcript_text', 'TEXT')
        _ensure_column(conn, 'meetings', 'trained_transcript_json', 'TEXT')
        _ensure_column(conn, 'meetings', 'groq_summary_json', 'TEXT')
        _ensure_column(conn, 'meetings', 'trained_summary_json', 'TEXT')
        _ensure_column(conn, 'meetings', 'comparison_errors', 'TEXT')


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in (
        'transcript_json',
        'groq_transcript_json',
        'trained_transcript_json',
        'summary_json',
        'groq_summary_json',
        'trained_summary_json',
        'comparison_errors',
    ):
        if data.get(key):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    data['approved'] = bool(data.get('approved'))
    return data


def create_meeting(meeting_id: str, filename: str, upload_path: str, language: str = 'auto') -> dict[str, Any]:
    now = utc_now()
    with get_conn() as conn:
        conn.execute(
            '''
            INSERT INTO meetings (id, filename, upload_path, language, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (meeting_id, filename, upload_path, language, 'uploaded', now, now),
        )
        row = conn.execute('SELECT * FROM meetings WHERE id = ?', (meeting_id,)).fetchone()
    return row_to_dict(row) or {}


def list_meetings() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute('SELECT * FROM meetings ORDER BY created_at DESC').fetchall()
    return [row_to_dict(row) or {} for row in rows]


def get_meeting(meeting_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute('SELECT * FROM meetings WHERE id = ?', (meeting_id,)).fetchone()
    return row_to_dict(row)


def update_meeting(meeting_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return get_meeting(meeting_id)

    prepared: dict[str, Any] = {}
    for key, value in fields.items():
        if key in {
            'transcript_json',
            'groq_transcript_json',
            'trained_transcript_json',
            'summary_json',
            'groq_summary_json',
            'trained_summary_json',
            'comparison_errors',
        } and value is not None:
            prepared[key] = json.dumps(value, ensure_ascii=False)
        elif key == 'approved' and value is not None:
            prepared[key] = 1 if value else 0
        else:
            prepared[key] = value
    prepared['updated_at'] = utc_now()

    assignments = ', '.join(f'{key} = ?' for key in prepared)
    values = list(prepared.values()) + [meeting_id]
    with get_conn() as conn:
        conn.execute(f'UPDATE meetings SET {assignments} WHERE id = ?', values)
        row = conn.execute('SELECT * FROM meetings WHERE id = ?', (meeting_id,)).fetchone()
    return row_to_dict(row)
