# emotion_analysis

A Python library for **multi-dimensional emotion analysis of conversational transcripts**.

Domain-agnostic: works equally well for psychotherapy, couples counselling, business negotiations, political discourse, medical consultations, and any other structured human dialogue.

---

## Concepts

### Emotion vectors
Each message is annotated with an `EmotionVector` — a structured, numeric representation of emotional state grounded in established psychology frameworks:

| Dimension | Range | Framework |
|---|---|---|
| `valence` | −1 to +1 | PAD model |
| `arousal` | 0 to 1 | PAD model |
| `dominance` | −1 to +1 | PAD model |
| `joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, `contempt`, `trust`, `anticipation` | 0 to 1 | Plutchik's wheel |
| `assertiveness`, `openness`, `hostility`, `vulnerability` | 0 to 1 | Communicative intent |

### Interaction health
A running score (0–10) tracking the cumulative constructiveness of the conversation at each turn — not an individual message score but a conversational state indicator.

### Emotional dynamics
Computed from the annotated sequence with no additional LLM calls:

- **Trajectories** — per-speaker time series of valence, arousal, dominance
- **Turning points** — statistically significant shifts in emotional tone
- **Reactivity analysis** — how Speaker A's emotional state predicts shifts in Speaker B (Pearson correlation of trigger valence with response delta)
- **Emotional contagion** — degree of cross-speaker valence synchrony
- **Escalation / de-escalation sequences** — monotone runs of falling or rising valence
- **Repair attempts** — moments where a speaker attempts to restore positivity after conflict

---

## Installation

```bash
pip install anthropic pydantic          # minimal
pip install emotion-analysis            # once published
```

Requires Python 3.10+ and an Anthropic API key.

---

## Quick start

```python
from emotion_analysis import EmotionAnalyzer, parse_transcript
from emotion_analysis.models import ConversationDomain

transcript = parse_transcript(
    """
    Alice: I feel like my concerns are never taken seriously in these meetings.
    Bob: That's not true — I always listen to everyone.
    Alice: Really? Because last Tuesday you cut me off mid-sentence.
    Bob: I was just trying to keep us on schedule.
    """,
    format="plain",
    domain=ConversationDomain.BUSINESS,
    context_note="Post-meeting debrief between team members",
)

analyzer = EmotionAnalyzer()   # reads ANTHROPIC_API_KEY from environment
analysis = analyzer.analyze(transcript)

for am in analysis.annotated_messages:
    print(
        f"[{am.message.turn_index}] {am.message.speaker}: "
        f"{am.emotion.label}  "
        f"(valence={am.emotion.valence:+.2f}, health={am.interaction_health:.1f})"
    )

print(analysis.summary.overview)
print(analysis.summary.recommendations)
```

---

## Parsing supported formats

```python
from emotion_analysis import parse_transcript
from pathlib import Path

# Plain text  ("Speaker: message" per line)
t = parse_transcript(text, format="plain")

# WhatsApp export
t = parse_transcript(Path("chat.txt"), format="whatsapp")

# Telegram export
t = parse_transcript(text, format="telegram")

# Slack JSON export
t = parse_transcript(slack_json_list, format="slack", user_map={"U012AB": "Alice"})

# Generic JSON array
t = parse_transcript(json_str, format="json",
                     speaker_field="from", text_field="body")

# CSV
t = parse_transcript(Path("chat.csv"), format="csv",
                     speaker_col="author", text_col="content")

# Python list of dicts
t = parse_transcript(messages, format="list")
```

---

## Advanced dynamics

```python
from emotion_analysis import (
    speaker_influence_score,
    emotional_regulation_index,
    compute_valence_momentum,
    compute_repair_attempts,
    conversation_health_summary,
)

# Directional influence: how does Alice's emotional state affect Bob?
influence = speaker_influence_score(analysis.annotated_messages, "Alice", "Bob")
print(influence["reactivity_r"])        # Pearson r: trigger valence → response delta
print(influence["mean_response_delta"]) # On average, does Bob get more or less negative?

# Does a speaker self-regulate after negative turns?
eri = emotional_regulation_index(analysis.annotated_messages, "Bob")
# +1 = consistently recovers; -1 = spirals; 0 = neutral

# Is the conversation getting better or worse right now?
momentum = compute_valence_momentum(analysis.annotated_messages, window=5)
# Positive slope = improving; negative slope = deteriorating

# Find repair attempts (apologies, validation, positive turn after conflict)
repairs = compute_repair_attempts(analysis.annotated_messages)

# Compact health dashboard dict
dashboard = conversation_health_summary(analysis.dynamics)
```

---

## Supported domains

`ConversationDomain.THERAPY` · `ROMANTIC` · `FAMILY` · `FRIENDSHIP` · `BUSINESS` · `POLITICAL` · `MEDICAL` · `LEGAL` · `EDUCATION` · `GENERAL`

Domain context shapes the model's interpretation and the `interaction_health` scale.

---

## Architecture

```
emotion_analysis/
├── models.py      # Pydantic data models (EmotionVector, ConversationAnalysis, …)
├── parsers.py     # Transcript parsers for all supported formats
├── prompts.py     # Claude prompt templates (annotation + summary)
├── analyzer.py    # LLM-powered analysis engine (Anthropic SDK)
└── dynamics.py    # Pure-Python dynamics algorithms (no LLM)
```

The LLM is used for two things only:
1. Annotating each message with an emotion vector (one API call per `analyze()`)
2. Generating the high-level narrative summary (one API call per `analyze()`)

All dynamics metrics (trajectories, reactivity, turning points, contagion, escalation) are computed deterministically from the annotations — fast, reproducible, and inspectable.

---

## Example scripts

```bash
# Full walkthrough with a therapy transcript
ANTHROPIC_API_KEY=... python examples/basic_usage.py

# Side-by-side comparison across therapy / business / political domains
ANTHROPIC_API_KEY=... python examples/multi_domain_demo.py
```
