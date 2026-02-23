# CLAUDE.md — Project Context for AI Assistants

This file summarises what was built in the founding session of this repository
and provides orientation for future sessions.  Update it whenever significant
new work is done.

---

## What this repository is

`lovegpt` started as a proof-of-concept Node.js/jQuery app (2023, Sam Rahimi)
for AI-powered relationship analysis.  The original app lives in `server/` and
is fully functional but dated.

In the session documented here, the core idea was extracted and reimplemented
as a proper, domain-agnostic **Python library** — `emotion_analysis/`.  The
original app is not being deleted; the Python library is the foundation for
whatever app layer comes next (React + Tailwind frontend, discussed but not
yet built).

---

## The `emotion_analysis` Python library

### Location
`emotion_analysis/` — importable as `from emotion_analysis import ...`

### Purpose
Analyse the emotional content of conversational transcripts using LLMs +
deterministic algorithms.  Deliberately **not** specific to romantic partners:
designed for therapy, business, politics, medical, legal, and general use.

### Architecture

```
emotion_analysis/
├── __init__.py     Public API surface (re-exports everything a caller needs)
├── models.py       Pydantic data models
├── parsers.py      Transcript parsers (7 deterministic formats + LLM fallback)
├── prompts.py      Claude/LLM prompt templates
├── analyzer.py     LLM orchestration — EmotionAnalyzer class
├── backends.py     LLM provider abstraction layer
└── dynamics.py     Pure-Python analytics algorithms
```

### Key design decisions

**EmotionVector** is multi-dimensional, grounded in:
- PAD (Pleasure-Arousal-Dominance) affective space: `valence`, `arousal`, `dominance`
- Plutchik's wheel: `joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`,
  `contempt`, `trust`, `anticipation`
- Communicative intent: `assertiveness`, `openness`, `hostility`, `vulnerability`
- Plus: `label` (natural language), `confidence`

**Dynamics algorithms** (no LLM, pure Python, in `dynamics.py`):
- Emotional trajectory — per-speaker time series of PAD dimensions
- Turning point detection — local z-score on valence delta
- Reactivity analysis — Pearson r between A's trigger valence and B's response delta
- Emotional contagion — cross-speaker valence correlation
- Escalation / de-escalation sequence detection
- Repair attempt detection (explicit speech acts + implicit valence recovery)
- Emotional regulation index — does a speaker recover from their own negativity?
- Valence momentum — rolling linear slope for real-time alerting

**Two LLM calls per `analyze()`**:
1. Per-message emotion annotation (→ `List[AnnotatedMessage]`)
2. Narrative summary with themes, patterns, power dynamics, recommendations,
   emotional arc

Optional third call: natural-language interpretation of dynamics findings.

### Supported transcript formats
`plain` (Speaker: message), `whatsapp`, `telegram`, `slack` (JSON),
`json` (configurable fields), `csv` (configurable columns), `list` (Python dicts),
`llm` / `auto` (LLM-based fallback for any unrecognised format)

**LLM-based parser (`parse_llm` / `format="llm"`):**
- Requires `OPENROUTER_API_KEY`; uses `google/gemini-3-flash-preview` by default
- Accepts a `format_hint` kwarg to guide the model (e.g. `"Discord DM export"`)
- Detected format stored in `transcript.metadata["llm_detected_format"]`
- Normalises speaker names, strips formatting artifacts, parses timestamps

### Supported conversation domains
`THERAPY`, `ROMANTIC`, `FAMILY`, `FRIENDSHIP`, `BUSINESS`, `POLITICAL`,
`MEDICAL`, `LEGAL`, `EDUCATION`, `GENERAL`

Domain affects both the system prompt framing and `interaction_health` scale.

---

## LLM backend system (`backends.py`)

Three backends, all implementing `complete(system, user) -> str`:

| Class | Provider | Default model |
|---|---|---|
| `AnthropicBackend` | Anthropic SDK | `claude-opus-4-6` |
| `OpenAIBackend` | openai SDK (any compatible endpoint) | `openai/gpt-5.2` via OpenRouter |
| `auto_detect_backend()` | picks from env vars | see below |

**Auto-detection priority:**
1. `ANTHROPIC_API_KEY` → AnthropicBackend
2. `OPENROUTER_API_KEY` → OpenAIBackend (OpenRouter, gpt-5.2)
3. `OPENAI_API_KEY` → OpenAIBackend (api.openai.com, gpt-4o)

`EmotionAnalyzer` accepts `backend=` as a string shortcut (`"anthropic"`,
`"openai"`, `"openrouter"`), a backend instance, or `None` (auto-detect).

`LLM_MODEL` environment variable overrides the model in all example scripts
without code changes.

`OpenAIBackend` works with any OpenAI-compatible server: OpenRouter, Groq,
Mistral, Ollama, LM Studio, vLLM, Azure.

---

## Tests

`tests/` — 89 tests, all passing, zero API calls required.

```
tests/test_parsers.py    — parser correctness + model unit tests + parse_llm (mocked)
tests/test_dynamics.py   — dynamics algorithms (math + integration)
tests/test_backends.py   — backend wiring, env var detection, SDK mocking
```

Run with: `python -m pytest tests/ -v`

---

## Example scripts

All scripts support any provider via env vars.  `LLM_MODEL` overrides the model.

| Script | What it does |
|---|---|
| `examples/basic_usage.py` | Full pipeline walkthrough on therapy transcript |
| `examples/multi_domain_demo.py` | Side-by-side comparison of 3 domains |
| `examples/openrouter_smoke_test.py` | Live integration test for OpenRouter |

All scripts support any provider via env vars.  `LLM_MODEL` overrides the model.

```bash
ANTHROPIC_API_KEY=...   python examples/basic_usage.py
OPENROUTER_API_KEY=...  python examples/basic_usage.py
LLM_MODEL=google/gemini-2.5-pro OPENROUTER_API_KEY=... python examples/multi_domain_demo.py
```

Sample conversations live in `examples/sample_conversations/`:
- `therapy_session.txt`
- `salary_negotiation.txt`
- `political_debate.txt`
- `family_conflict.txt`

---

## What has NOT been built yet

- **React + Tailwind frontend** — the user wants to discuss this next.  The
  library is the backend; we need to design the app layer separately.
- **Async support** — both Anthropic and OpenAI SDKs have async clients;
  adding `async def analyze()` variants would be useful for a web server.
- **Streaming** — real-time token streaming would improve UX for long transcripts.
- **Persistent storage** — sessions are currently in-memory only.
- **REST API** — the original app had an Express server; a FastAPI wrapper
  around `EmotionAnalyzer` is the natural next step before the frontend.
- **Batch / chunked analysis** — very long conversations should be windowed.
- **Visualisation** — matplotlib helpers for trajectory plots are sketched
  in `pyproject.toml` extras but not implemented.

---

## Original app (`server/`)

Node.js / Express + jQuery frontend.  Uses OpenAI-compatible API with GPT-4.1
default.  Fully functional as a standalone app.  Not maintained going forward —
the Python library supersedes its analysis logic.

Env vars the original app uses:
- `OPENAI_API_KEY`
- `OPENAI_API_ENDPOINT` (optional, defaults to `https://api.openai.com/v1`)
- `LOVEGPT_MODEL` (optional, defaults to `gpt-4.1`)

---

## Branch convention

Active development branch: `claude/emotion-analysis-library-8vZi1`
Future Claude sessions should create branches prefixed with `claude/` and
suffixed with the session ID (visible in the git push instructions).

---

## Quick orientation for a new session

```bash
# Install deps
pip install anthropic openai pydantic

# Run tests (no API key needed)
python -m pytest tests/ -v

# Run the demo
OPENROUTER_API_KEY=<key> python examples/basic_usage.py

# Import the library
from emotion_analysis import EmotionAnalyzer, parse_transcript
from emotion_analysis.models import ConversationDomain

transcript = parse_transcript("Alice: Hello.\nBob: Hi!", format="plain")
analysis = EmotionAnalyzer().analyze(transcript)
```
