"""
Transcript parsers for common conversation export formats.

All parsers return a ConversationTranscript ready for analysis.
Supported formats:
  - plain      "Speaker: message" line-by-line
  - whatsapp   WhatsApp chat export  ([DD/MM/YYYY, HH:MM:SS] Speaker: message)
  - telegram   Telegram text export  (similar timestamp prefix)
  - slack      Slack JSON export (list of message objects)
  - json       Generic JSON array with configurable field names
  - csv        CSV with configurable column names
  - list       Python list of dicts (programmatic use)
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import ConversationDomain, ConversationTranscript, Message


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_transcript(
    raw_messages: List[Dict[str, Any]],
    domain: ConversationDomain = ConversationDomain.GENERAL,
    context_note: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationTranscript:
    """Convert a list of raw dicts to a ConversationTranscript."""
    messages = [
        Message(
            turn_index=i,
            speaker=row["speaker"].strip(),
            text=row["text"].strip(),
            timestamp=row.get("timestamp"),
            metadata=row.get("metadata", {}),
        )
        for i, row in enumerate(raw_messages)
        if row.get("text", "").strip()
    ]
    return ConversationTranscript(
        messages=messages,
        domain=domain,
        context_note=context_note,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------

def parse_plain(
    text: str,
    *,
    separator: str = ":",
    domain: ConversationDomain = ConversationDomain.GENERAL,
    context_note: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationTranscript:
    """
    Parse a plain-text transcript where each line is ``Speaker: message``.

    Blank lines and lines without the separator are skipped.
    Multi-line messages are not supported by this format (use JSON/CSV instead).

    Example input::

        Alice: Hey, did you get my email?
        Bob: Yes, just read it. Looks good overall.
        Alice: Glad to hear it. Any concerns?
    """
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if separator not in line:
            # Continuation line — append to previous message if any
            if rows:
                rows[-1]["text"] += " " + line
            continue
        speaker, _, content = line.partition(separator)
        rows.append({"speaker": speaker.strip(), "text": content.strip()})
    return _build_transcript(rows, domain, context_note, metadata)


# WhatsApp date patterns:
#   [12/07/2024, 14:35:22] Alice: Hello!
#   [07/12/2024, 14:35:22] Bob: Hi there
_WA_PATTERN = re.compile(
    r"^\[(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}),?\s+(\d{2}:\d{2}(?::\d{2})?)\]\s+([^:]+):\s+(.*)"
)
_WA_DATE_FMTS = ["%d/%m/%Y", "%m/%d/%Y", "%d/%m/%y", "%m/%d/%y", "%d-%m-%Y", "%Y-%m-%d"]


def _parse_wa_timestamp(date_str: str, time_str: str) -> Optional[datetime]:
    for fmt in _WA_DATE_FMTS:
        try:
            return datetime.strptime(f"{date_str} {time_str}", f"{fmt} %H:%M:%S")
        except ValueError:
            pass
        try:
            return datetime.strptime(f"{date_str} {time_str}", f"{fmt} %H:%M")
        except ValueError:
            pass
    return None


def parse_whatsapp(
    text: str,
    *,
    domain: ConversationDomain = ConversationDomain.GENERAL,
    context_note: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationTranscript:
    """
    Parse a WhatsApp chat export file.

    WhatsApp exports look like::

        [12/07/2024, 14:35:22] Alice: Hey!
        [12/07/2024, 14:36:01] Bob: Hey back

    System messages (e.g. "Messages and calls are end-to-end encrypted") are
    dropped automatically.
    """
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _WA_PATTERN.match(line)
        if not m:
            # Possible continuation of a multi-line message
            if rows:
                rows[-1]["text"] += "\n" + line
            continue
        date_s, time_s, speaker, content = m.groups()
        ts = _parse_wa_timestamp(date_s, time_s)
        rows.append({"speaker": speaker.strip(), "text": content.strip(), "timestamp": ts})
    return _build_transcript(rows, domain, context_note, metadata)


def parse_telegram(
    text: str,
    *,
    domain: ConversationDomain = ConversationDomain.GENERAL,
    context_note: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationTranscript:
    """
    Parse Telegram text export format.

    Telegram exports (saved as .txt) typically look like::

        Alice, [12 Jul 2024 at 14:35:22]:
        Hello there!

        Bob, [12 Jul 2024 at 14:36:01]:
        Hi!

    The speaker+timestamp appears on one line; the message on the next.
    """
    _TG_HEADER = re.compile(r"^(.+),\s+\[(\d{1,2}\s+\w+\s+\d{4}\s+at\s+\d{2}:\d{2}:\d{2})\]:?$")
    rows: List[Dict[str, Any]] = []
    pending_speaker: Optional[str] = None
    pending_ts: Optional[datetime] = None

    for line in text.splitlines():
        m = _TG_HEADER.match(line.strip())
        if m:
            pending_speaker = m.group(1).strip()
            try:
                pending_ts = datetime.strptime(m.group(2), "%d %b %Y at %H:%M:%S")
            except ValueError:
                pending_ts = None
        elif pending_speaker and line.strip():
            rows.append({
                "speaker": pending_speaker,
                "text": line.strip(),
                "timestamp": pending_ts,
            })
            pending_speaker = None
            pending_ts = None
        elif rows and line.strip():
            rows[-1]["text"] += "\n" + line.strip()

    return _build_transcript(rows, domain, context_note, metadata)


def parse_slack_json(
    data: Union[str, List[Dict[str, Any]]],
    *,
    user_map: Optional[Dict[str, str]] = None,
    domain: ConversationDomain = ConversationDomain.GENERAL,
    context_note: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationTranscript:
    """
    Parse a Slack channel JSON export (a list of message objects).

    ``user_map`` maps Slack user IDs (e.g. "U012AB3CD") to display names.
    If not provided, the raw ``user`` field is used as the speaker name.

    Only messages with ``"type": "message"`` and a non-empty ``"text"`` are
    included; bot messages and system messages are dropped.
    """
    if isinstance(data, str):
        data = json.loads(data)

    user_map = user_map or {}
    rows: List[Dict[str, Any]] = []

    for item in data:
        if item.get("type") != "message":
            continue
        if item.get("subtype"):  # bot_message, channel_join, etc.
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        user_id = item.get("user", "unknown")
        speaker = user_map.get(user_id, user_id)
        ts_raw = item.get("ts")
        ts = datetime.fromtimestamp(float(ts_raw)) if ts_raw else None
        rows.append({"speaker": speaker, "text": text, "timestamp": ts,
                     "metadata": {"slack_ts": ts_raw, "user_id": user_id}})

    return _build_transcript(rows, domain, context_note, metadata)


def parse_json(
    data: Union[str, List[Dict[str, Any]]],
    *,
    speaker_field: str = "speaker",
    text_field: str = "text",
    timestamp_field: Optional[str] = "timestamp",
    domain: ConversationDomain = ConversationDomain.GENERAL,
    context_note: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationTranscript:
    """
    Parse a generic JSON array of message objects.

    Field names are configurable so this can adapt to any JSON shape.

    Example::

        [
          {"from": "Alice", "body": "Hello!", "sent_at": "2024-07-12T14:35:00"},
          {"from": "Bob",   "body": "Hi!",    "sent_at": "2024-07-12T14:36:00"}
        ]

    Called with ``speaker_field="from", text_field="body", timestamp_field="sent_at"``.
    """
    if isinstance(data, str):
        data = json.loads(data)

    rows: List[Dict[str, Any]] = []
    for item in data:
        speaker = str(item.get(speaker_field, "unknown"))
        text = str(item.get(text_field, "")).strip()
        if not text:
            continue
        ts = None
        if timestamp_field and timestamp_field in item:
            raw = item[timestamp_field]
            if isinstance(raw, (int, float)):
                ts = datetime.fromtimestamp(raw)
            elif isinstance(raw, str):
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f"):
                    try:
                        ts = datetime.strptime(raw, fmt)
                        break
                    except ValueError:
                        pass
        rows.append({"speaker": speaker, "text": text, "timestamp": ts})

    return _build_transcript(rows, domain, context_note, metadata)


def parse_csv(
    data: Union[str, Path],
    *,
    speaker_col: str = "speaker",
    text_col: str = "text",
    timestamp_col: Optional[str] = "timestamp",
    delimiter: str = ",",
    domain: ConversationDomain = ConversationDomain.GENERAL,
    context_note: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationTranscript:
    """
    Parse a CSV file or CSV string.

    ``data`` may be a file path (``Path`` or string ending in ``.csv``) or a
    raw CSV string.  Column names are configurable.
    """
    if isinstance(data, Path) or (isinstance(data, str) and data.strip().endswith(".csv")):
        text = Path(data).read_text(encoding="utf-8")
    else:
        text = data  # type: ignore[assignment]

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows: List[Dict[str, Any]] = []
    for row in reader:
        speaker = row.get(speaker_col, "").strip()
        text_val = row.get(text_col, "").strip()
        if not speaker or not text_val:
            continue
        ts = None
        if timestamp_col and timestamp_col in row:
            raw = row[timestamp_col].strip()
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    ts = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    pass
        rows.append({"speaker": speaker, "text": text_val, "timestamp": ts})

    return _build_transcript(rows, domain, context_note, metadata)


def parse_list(
    messages: List[Dict[str, Any]],
    *,
    speaker_field: str = "speaker",
    text_field: str = "text",
    timestamp_field: Optional[str] = None,
    domain: ConversationDomain = ConversationDomain.GENERAL,
    context_note: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationTranscript:
    """
    Parse a Python list of dicts — for programmatic use.

    Example::

        messages = [
            {"speaker": "Therapist", "text": "How are you feeling today?"},
            {"speaker": "Patient",   "text": "Honestly, pretty anxious."},
        ]
        transcript = parse_list(messages, domain=ConversationDomain.THERAPY)
    """
    rows: List[Dict[str, Any]] = []
    for item in messages:
        speaker = str(item.get(speaker_field, "unknown"))
        text = str(item.get(text_field, "")).strip()
        if not text:
            continue
        ts = item.get(timestamp_field) if timestamp_field else None
        rows.append({"speaker": speaker, "text": text, "timestamp": ts})

    return _build_transcript(rows, domain, context_note, metadata)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

_FORMAT_MAP = {
    "plain": parse_plain,
    "txt": parse_plain,
    "whatsapp": parse_whatsapp,
    "wa": parse_whatsapp,
    "telegram": parse_telegram,
    "tg": parse_telegram,
    "slack": parse_slack_json,
    "json": parse_json,
    "csv": parse_csv,
    "list": parse_list,
}


def parse_transcript(
    source: Union[str, Path, List[Dict[str, Any]]],
    format: str = "plain",  # noqa: A002
    *,
    domain: ConversationDomain = ConversationDomain.GENERAL,
    context_note: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ConversationTranscript:
    """
    Unified parser — choose a format and pass the source data.

    Parameters
    ----------
    source:
        Raw text string, a file path (``Path``), or a list of dicts.
    format:
        One of: ``plain``, ``whatsapp``, ``telegram``, ``slack``,
        ``json``, ``csv``, ``list``.
    domain:
        Conversational domain for context-aware analysis.
    context_note:
        Optional free-text description of the conversation's background.
    metadata:
        Arbitrary key-value pairs attached to the transcript.
    **kwargs:
        Forwarded to the specific parser (e.g. ``separator``, ``speaker_col``).

    Returns
    -------
    ConversationTranscript
    """
    fmt = format.lower()
    if fmt not in _FORMAT_MAP:
        raise ValueError(
            f"Unknown format {format!r}. Choose from: {', '.join(_FORMAT_MAP)}"
        )

    # Auto-read file paths
    if isinstance(source, Path):
        if fmt in ("json", "slack"):
            source = json.loads(source.read_text(encoding="utf-8"))
        elif fmt == "csv":
            source = source.read_text(encoding="utf-8")
        else:
            source = source.read_text(encoding="utf-8")

    parser = _FORMAT_MAP[fmt]
    return parser(  # type: ignore[call-arg]
        source,
        domain=domain,
        context_note=context_note,
        metadata=metadata,
        **kwargs,
    )
