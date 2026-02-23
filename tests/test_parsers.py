"""Tests for transcript parsers — no API calls required."""

import json
from datetime import datetime

import pytest

from emotion_analysis.models import ConversationDomain
from emotion_analysis.parsers import (
    parse_csv,
    parse_json,
    parse_list,
    parse_plain,
    parse_slack_json,
    parse_telegram,
    parse_transcript,
    parse_whatsapp,
)


# ---------------------------------------------------------------------------
# plain
# ---------------------------------------------------------------------------

PLAIN_TEXT = """\
Alice: Hello there.
Bob: Hi! How are you?
Alice: I'm doing well, thanks.
"""


def test_parse_plain_basic():
    t = parse_plain(PLAIN_TEXT)
    assert len(t.messages) == 3
    assert t.messages[0].speaker == "Alice"
    assert t.messages[0].text == "Hello there."
    assert t.messages[1].speaker == "Bob"
    assert t.messages[0].turn_index == 0
    assert t.messages[2].turn_index == 2


def test_parse_plain_participants_inferred():
    t = parse_plain(PLAIN_TEXT)
    assert t.participants == ["Alice", "Bob"]


def test_parse_plain_blank_lines_skipped():
    text = "\nAlice: Line one.\n\nBob: Line two.\n"
    t = parse_plain(text)
    assert len(t.messages) == 2


def test_parse_plain_custom_separator():
    text = "Alice | Hello\nBob | World"
    t = parse_plain(text, separator="|")
    assert t.messages[0].text == "Hello"
    assert t.messages[1].speaker == "Bob"


def test_parse_plain_domain_passed():
    t = parse_plain(PLAIN_TEXT, domain=ConversationDomain.THERAPY)
    assert t.domain == ConversationDomain.THERAPY


def test_parse_plain_context_note():
    t = parse_plain(PLAIN_TEXT, context_note="intake session")
    assert t.context_note == "intake session"


# ---------------------------------------------------------------------------
# whatsapp
# ---------------------------------------------------------------------------

WA_TEXT = """\
[12/07/2024, 14:35:22] Alice: Hey, did you see that?
[12/07/2024, 14:36:01] Bob: Yes I did!
[12/07/2024, 14:37:45] Alice: Amazing.
"""


def test_parse_whatsapp_basic():
    t = parse_whatsapp(WA_TEXT)
    assert len(t.messages) == 3
    assert t.messages[0].speaker == "Alice"
    assert t.messages[0].text == "Hey, did you see that?"


def test_parse_whatsapp_timestamps():
    t = parse_whatsapp(WA_TEXT)
    ts = t.messages[0].timestamp
    assert isinstance(ts, datetime)
    assert ts.day == 12
    assert ts.month == 7


# ---------------------------------------------------------------------------
# json
# ---------------------------------------------------------------------------

JSON_DATA = [
    {"from": "Alice", "body": "Hello!", "sent_at": "2024-07-12T14:35:00"},
    {"from": "Bob", "body": "Hi there.", "sent_at": "2024-07-12T14:36:00"},
]


def test_parse_json_basic():
    t = parse_json(
        JSON_DATA, speaker_field="from", text_field="body", timestamp_field="sent_at"
    )
    assert len(t.messages) == 2
    assert t.messages[0].speaker == "Alice"
    assert t.messages[0].text == "Hello!"


def test_parse_json_from_string():
    t = parse_json(
        json.dumps(JSON_DATA),
        speaker_field="from",
        text_field="body",
        timestamp_field="sent_at",
    )
    assert len(t.messages) == 2


def test_parse_json_timestamp_parsed():
    t = parse_json(JSON_DATA, speaker_field="from", text_field="body",
                   timestamp_field="sent_at")
    assert isinstance(t.messages[0].timestamp, datetime)


# ---------------------------------------------------------------------------
# csv
# ---------------------------------------------------------------------------

CSV_DATA = "speaker,text,timestamp\nAlice,Hello!,2024-07-12T14:35:00\nBob,Hi!,2024-07-12T14:36:00"


def test_parse_csv_basic():
    t = parse_csv(CSV_DATA)
    assert len(t.messages) == 2
    assert t.messages[0].speaker == "Alice"
    assert t.messages[1].text == "Hi!"


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def test_parse_list_basic():
    messages = [
        {"speaker": "Therapist", "text": "How are you feeling today?"},
        {"speaker": "Patient", "text": "Honestly, pretty anxious."},
    ]
    t = parse_list(messages, domain=ConversationDomain.THERAPY)
    assert len(t.messages) == 2
    assert t.domain == ConversationDomain.THERAPY
    assert t.messages[1].text == "Honestly, pretty anxious."


# ---------------------------------------------------------------------------
# slack
# ---------------------------------------------------------------------------

SLACK_DATA = [
    {"type": "message", "user": "U001", "text": "Hello team!", "ts": "1720789200.0"},
    {"type": "message", "user": "U002", "text": "Hi everyone!", "ts": "1720789260.0"},
    {"type": "message", "subtype": "bot_message", "text": "Bot says hi", "ts": "1720789300.0"},
]


def test_parse_slack_basic():
    t = parse_slack_json(SLACK_DATA, user_map={"U001": "Alice", "U002": "Bob"})
    # Bot message should be filtered
    assert len(t.messages) == 2
    assert t.messages[0].speaker == "Alice"
    assert t.messages[1].speaker == "Bob"


# ---------------------------------------------------------------------------
# unified parse_transcript
# ---------------------------------------------------------------------------

def test_parse_transcript_plain():
    t = parse_transcript(PLAIN_TEXT, format="plain")
    assert len(t.messages) == 3


def test_parse_transcript_unknown_format():
    with pytest.raises(ValueError, match="Unknown format"):
        parse_transcript("anything", format="xml")


def test_parse_transcript_wa_alias():
    t = parse_transcript(WA_TEXT, format="wa")
    assert len(t.messages) == 3


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def test_emotion_vector_distance():
    from emotion_analysis.models import EmotionVector

    a = EmotionVector(valence=0.0, arousal=0.5, dominance=0.0, label="neutral")
    b = EmotionVector(valence=1.0, arousal=0.5, dominance=0.0, label="positive")
    dist = a.distance_to(b)
    assert abs(dist - 1.0) < 1e-6


def test_emotion_vector_pad_tuple():
    from emotion_analysis.models import EmotionVector

    v = EmotionVector(valence=0.5, arousal=0.3, dominance=-0.2, label="test")
    assert v.pad_tuple() == (0.5, 0.3, -0.2)


def test_emotion_vector_dominant_basic():
    from emotion_analysis.models import EmotionVector

    v = EmotionVector(
        valence=-0.5, arousal=0.8, dominance=-0.5,
        label="angry", anger=0.9, fear=0.2
    )
    assert v.dominant_basic_emotion() == "anger"
