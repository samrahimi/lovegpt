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
  - llm / auto LLM-based parser for any unsupported or ambiguous format
               (requires OPENROUTER_API_KEY; uses google/gemini-3-flash-preview)
"""

from __future__ import annotations

import csv
import io
import json
import os
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
# LLM-based fallback parser
# ---------------------------------------------------------------------------

LLM_PARSE_MODEL = "google/gemini-3-flash-preview"
LLM_PARSE_BASE_URL = "https://openrouter.ai/api/v1"

_LLM_SYSTEM = (
    "You are a transcript parser. Extract conversational turns from raw text "
    "and return them as structured JSON. "
    "Output ONLY valid JSON — no commentary, no markdown fences."
)

_LLM_USER_TEMPLATE = """\
Extract all conversational turns from the text below.
{hint_block}
Return a JSON object with this exact schema:
{{
  "detected_format": "<brief description of the source format, e.g. 'iMessage export', 'Discord log', 'email thread'>",
  "messages": [
    {{
      "speaker": "<normalized speaker name>",
      "text": "<message content, cleaned of formatting artifacts>",
      "timestamp": "<ISO 8601 string, e.g. 2024-07-12T14:35:00, or null>"
    }}
  ]
}}

Rules:
1. Include ONLY actual conversational messages from participants.
   Skip system notifications (e.g. "X joined the group", "missed call",
   "end-to-end encrypted", delivery/read receipts) and standalone metadata lines.
2. Normalize speaker names — use the same display name consistently even if the
   source uses usernames, phone numbers, abbreviations, or role labels.
3. Strip formatting artifacts from the text field (embedded timestamps, emoji
   reactions, edit/delete markers, etc.).
4. Preserve the original message order exactly.
5. Timestamps: if clearly present and parseable, format as ISO 8601; otherwise null.
6. If a speaker cannot be determined for a message, use "Unknown".

Source text:
---
{text}
---"""


def parse_llm(
    text: str,
    *,
    api_key: Optional[str] = None,
    model: str = LLM_PARSE_MODEL,
    base_url: str = LLM_PARSE_BASE_URL,
    format_hint: Optional[str] = None,
    domain: ConversationDomain = ConversationDomain.GENERAL,
    context_note: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationTranscript:
    """
    Parse a transcript in *any* format by delegating structure-extraction to an LLM.

    Use this when the source text is in a format not covered by the deterministic
    parsers — e.g. iMessage screenshots described as text, custom forum exports,
    email threads, proprietary chat logs, or any other layout where speaker turns
    are discernible but the format is non-standard.

    Requires ``OPENROUTER_API_KEY`` to be set (or passed via ``api_key``).
    Uses ``google/gemini-3-flash-preview`` via OpenRouter by default — a fast,
    cheap model well-suited to structured extraction tasks.

    Parameters
    ----------
    text:
        Raw transcript text in any format.
    api_key:
        OpenRouter API key.  Falls back to the ``OPENROUTER_API_KEY`` env var.
    model:
        Model to use for parsing.  Default: ``google/gemini-3-flash-preview``.
    base_url:
        API base URL.  Default: ``https://openrouter.ai/api/v1`` (OpenRouter).
    format_hint:
        Optional free-text hint about the source format, e.g.
        ``"Discord server export"`` or ``"iMessage thread between two people"``.
        Helps the model avoid common mis-classifications.
    domain:
        Conversation domain attached to the returned transcript.
    context_note:
        Optional free-text context about the conversation.
    metadata:
        Arbitrary key-value pairs attached to the transcript.

    Returns
    -------
    ConversationTranscript

    Raises
    ------
    ValueError
        If no API key is available.
    RuntimeError
        If the model response cannot be parsed as valid JSON.

    Examples
    --------
    ::

        # Any weird format — let the model figure it out
        raw = open("mystery_chat.txt").read()
        transcript = parse_llm(raw, format_hint="exported from a private forum")

        # Via the unified entry point
        transcript = parse_transcript(raw, format="llm",
                                      format_hint="iMessage export")
    """
    try:
        import openai  # deferred — only required for this parser
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for parse_llm(). "
            "Install it with: pip install openai"
        ) from exc

    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_key:
        raise ValueError(
            "parse_llm() requires an OpenRouter API key. "
            "Set OPENROUTER_API_KEY or pass api_key= explicitly."
        )

    client = openai.OpenAI(api_key=resolved_key, base_url=base_url)

    hint_block = (
        f'Format hint from caller: "{format_hint}"\n\n' if format_hint else ""
    )
    user_prompt = _LLM_USER_TEMPLATE.format(
        hint_block=hint_block, text=text.strip()
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_output = (response.choices[0].message.content or "").strip()

    # Strip markdown code fences if the model wraps the output
    if raw_output.startswith("```"):
        lines = raw_output.splitlines()
        raw_output = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM parser returned output that is not valid JSON.\n"
            f"Model: {model}\n"
            f"Raw output (first 500 chars):\n{raw_output[:500]}"
        ) from exc

    detected = parsed.get("detected_format", "unknown")
    messages_raw: List[Dict[str, Any]] = parsed.get("messages", [])
    if not isinstance(messages_raw, list):
        raise RuntimeError(
            f"LLM parser returned unexpected structure — "
            f"'messages' field is {type(messages_raw).__name__}, expected list."
        )

    rows: List[Dict[str, Any]] = []
    for item in messages_raw:
        speaker = str(item.get("speaker") or "Unknown").strip()
        msg_text = str(item.get("text") or "").strip()
        if not msg_text:
            continue
        ts: Optional[datetime] = None
        ts_raw = item.get("timestamp")
        if ts_raw and ts_raw != "null":
            for fmt in (
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    ts = datetime.strptime(str(ts_raw), fmt)
                    break
                except ValueError:
                    pass
        rows.append({"speaker": speaker, "text": msg_text, "timestamp": ts})

    meta = dict(metadata or {})
    meta["llm_detected_format"] = detected
    meta["llm_parse_model"] = model

    return _build_transcript(rows, domain, context_note, meta)


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
    "llm": parse_llm,
    "auto": parse_llm,   # alias: "I don't know the format, figure it out"
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
        One of:

        ============  ====================================================
        ``plain``     ``"Speaker: message"`` line-by-line (default)
        ``whatsapp``  WhatsApp chat export
        ``telegram``  Telegram text export
        ``slack``     Slack JSON export
        ``json``      Generic JSON array (configurable field names)
        ``csv``       CSV (configurable column names)
        ``list``      Python list of dicts
        ``llm``       LLM-based parser for unsupported/ambiguous formats
        ``auto``      Alias for ``llm``
        ============  ====================================================

        The ``llm`` / ``auto`` formats require ``OPENROUTER_API_KEY`` and
        use ``google/gemini-3-flash-preview`` by default.  Pass
        ``api_key=``, ``model=``, or ``format_hint=`` via ``**kwargs``
        to customise.

    domain:
        Conversational domain for context-aware analysis.
    context_note:
        Optional free-text description of the conversation's background.
    metadata:
        Arbitrary key-value pairs attached to the transcript.
    **kwargs:
        Forwarded to the specific parser.  Common options:

        - ``separator`` (plain) — field separator, default ``":"``
        - ``speaker_col`` / ``text_col`` (csv) — column names
        - ``speaker_field`` / ``text_field`` (json/list) — field names
        - ``user_map`` (slack) — ``{user_id: display_name}`` mapping
        - ``api_key`` / ``model`` / ``format_hint`` (llm/auto)

    Returns
    -------
    ConversationTranscript
    """
    fmt = format.lower()
    if fmt not in _FORMAT_MAP:
        raise ValueError(
            f"Unknown format {format!r}. "
            f"Choose from: {', '.join(sorted(set(_FORMAT_MAP)))}"
        )

    # Auto-read file paths — all text-based formats (including llm/auto) get str
    if isinstance(source, Path):
        if fmt in ("json", "slack"):
            source = json.loads(source.read_text(encoding="utf-8"))
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
