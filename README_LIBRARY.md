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
- **Turning points** — statistically significant shifts in emotional tone (local z-score)
- **Reactivity analysis** — how Speaker A's emotional state predicts shifts in Speaker B (Pearson r of trigger valence vs. response delta)
- **Emotional contagion** — degree of cross-speaker valence synchrony
- **Escalation / de-escalation sequences** — monotone runs of falling or rising valence
- **Repair attempts** — moments where a speaker attempts to restore positivity after conflict
- **Emotional regulation index** — does a speaker self-recover after negative turns?
- **Valence momentum** — rolling slope for real-time "is this getting worse?" alerting

---

## Installation

```bash
pip install anthropic openai pydantic
```

Requires Python 3.10+.  At least one LLM API key is required to run analysis
(see [LLM providers](#llm-providers) below).

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

# EmotionAnalyzer auto-detects whichever API key you have set
analyzer = EmotionAnalyzer()
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

## LLM providers

The library works with **any of three providers** — set whichever API key you have:

| Environment variable | Provider | Default model |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic (Claude) | `claude-opus-4-6` |
| `OPENROUTER_API_KEY` | OpenRouter | `openai/gpt-5.2` |
| `OPENAI_API_KEY` | OpenAI | `gpt-4o` |

**Auto-detection priority:** Anthropic → OpenRouter → OpenAI.  The first key found wins.

### Selecting a provider explicitly

```python
from emotion_analysis import EmotionAnalyzer

# Explicit backend strings
analyzer = EmotionAnalyzer(backend="anthropic")
analyzer = EmotionAnalyzer(backend="openai")   # uses OpenRouter key by default

# Override the model without changing code
analyzer = EmotionAnalyzer(backend="openai", model="google/gemini-2.5-pro")
analyzer = EmotionAnalyzer(backend="anthropic", model="claude-haiku-4-5-20251001")
```

### Custom / local endpoints

```python
from emotion_analysis.backends import OpenAIBackend

# Ollama (local)
backend = OpenAIBackend(
    base_url="http://localhost:11434/v1",
    model="llama3.2",
    api_key="ollama",   # Ollama ignores the key but the SDK requires one
)

# Groq
backend = OpenAIBackend(
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
)

analyzer = EmotionAnalyzer(backend=backend)
```

### env-var model override (scripts / CI)

All example scripts respect the `LLM_MODEL` environment variable so you can
switch models without editing code:

```bash
LLM_MODEL=google/gemini-2.5-pro  OPENROUTER_API_KEY=... python examples/basic_usage.py
LLM_MODEL=claude-haiku-4-5-20251001  ANTHROPIC_API_KEY=... python examples/multi_domain_demo.py
```

---

## Parsing transcripts

### Supported formats

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

# Generic JSON array (configurable field names)
t = parse_transcript(json_str, format="json",
                     speaker_field="from", text_field="body")

# CSV (configurable column names)
t = parse_transcript(Path("chat.csv"), format="csv",
                     speaker_col="author", text_col="content")

# Python list of dicts
t = parse_transcript(messages, format="list")
```

### LLM-based parser for unsupported formats

When the source is in a format not covered above — a custom forum export, iMessage
screenshot text, email thread, proprietary log, Discord export with an unusual
layout, etc. — use `format="llm"` (or its alias `"auto"`) to let an LLM extract
the structure automatically.

**Requires `OPENROUTER_API_KEY`.**  Uses `google/gemini-3-flash-preview` via
OpenRouter by default — a fast, inexpensive model well-suited to structured
extraction tasks.

```python
# Any unrecognised format — let the model figure it out
raw = open("mystery_chat.txt").read()
transcript = parse_transcript(raw, format="llm")

# Provide a hint to guide the model
transcript = parse_transcript(raw, format="auto",
                              format_hint="Discord server export, DMs between two users")

# Call the parser directly for full control
from emotion_analysis import parse_llm

transcript = parse_llm(
    raw,
    format_hint="iMessage thread exported from iPhone via third-party tool",
    model="google/gemini-3-flash-preview",   # default
    domain=ConversationDomain.ROMANTIC,
    context_note="argument between partners about holiday plans",
)
```

The detected format is stored in `transcript.metadata["llm_detected_format"]`
so you can inspect what the model inferred.

The LLM parser:
- Normalises speaker names to consistent display names
- Strips formatting artifacts (timestamps embedded in message bodies, delivery
  indicators, join/leave notifications, system messages)
- Preserves original message order exactly
- Parses timestamps to ISO 8601 when present, or leaves them null

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
print(influence["mean_response_delta"]) # On average, does Bob move positively or negatively?
print(influence["positive_responses"])  # Fraction of Bob's responses that improved

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
├── parsers.py     # Transcript parsers — deterministic formats + LLM-based fallback
├── backends.py    # LLM provider abstraction (AnthropicBackend, OpenAIBackend)
├── prompts.py     # Prompt templates for annotation and summary
├── analyzer.py    # LLM-powered analysis engine
└── dynamics.py    # Pure-Python dynamics algorithms (no LLM)
```

**LLM usage per `analyze()` call:**

| Call | Purpose | Required |
|---|---|---|
| 0 | Transcript parsing (`parse_llm` only) | Only for unsupported formats |
| 1 | Per-message emotion annotation | Always |
| 2 | Narrative summary (themes, recommendations, arc) | Always |
| 3 | Natural-language dynamics narrative | Optional (`include_dynamics_narrative=True`) |

All dynamics metrics (trajectories, reactivity, turning points, contagion,
escalation, repair, regulation) are computed deterministically from the
annotations — fast, reproducible, and free of further API calls.

---

## Example scripts

```bash
# Full walkthrough — any provider
ANTHROPIC_API_KEY=...  python examples/basic_usage.py
OPENROUTER_API_KEY=... python examples/basic_usage.py

# Side-by-side comparison across three domains
OPENROUTER_API_KEY=... python examples/multi_domain_demo.py

# Override model without code changes
LLM_MODEL=google/gemini-2.5-pro OPENROUTER_API_KEY=... python examples/multi_domain_demo.py

# Live OpenRouter integration test
OPENROUTER_API_KEY=... python examples/openrouter_smoke_test.py
```

Sample conversations in `examples/sample_conversations/`:
`therapy_session.txt` · `salary_negotiation.txt` · `political_debate.txt` · `family_conflict.txt`

---

## Running tests

No API key required:

```bash
pip install pytest
python -m pytest tests/ -v
# 89 tests: parsers, LLM parser (mocked), dynamics algorithms,
#           backend wiring, env-var auto-detection
```
