#!/usr/bin/env python3
"""
Basic usage example for the emotion_analysis library.

Demonstrates:
- Parsing a plain-text transcript
- Running the full analysis pipeline
- Inspecting per-message emotion vectors
- Reading the conversation summary and recommendations
- Using the dynamics engine for advanced insights

Supported providers (auto-detected from environment variables):
    ANTHROPIC_API_KEY    → Claude  (claude-opus-4-6 by default)
    OPENROUTER_API_KEY   → OpenRouter  (openai/gpt-5.2 by default)
    OPENAI_API_KEY       → OpenAI  (gpt-4o by default)

Override the model without changing code:
    LLM_MODEL=google/gemini-2.5-pro OPENROUTER_API_KEY=... python examples/basic_usage.py
    LLM_MODEL=claude-haiku-4-5-20251001 ANTHROPIC_API_KEY=... python examples/basic_usage.py

Optional: pip install rich  (for prettier output)
"""

import os
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_analysis import (
    EmotionAnalyzer,
    compute_repair_attempts,
    compute_valence_momentum,
    conversation_health_summary,
    emotional_regulation_index,
    parse_transcript,
    speaker_influence_score,
)
from emotion_analysis.backends import auto_detect_backend
from emotion_analysis.models import ConversationDomain

try:
    from rich import print as rprint
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    rprint = print  # type: ignore


# ---------------------------------------------------------------------------
# Load a sample transcript
# ---------------------------------------------------------------------------

SAMPLE_PATH = Path(__file__).parent / "sample_conversations" / "therapy_session.txt"


def load_sample() -> str:
    return SAMPLE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Provider / key detection
# ---------------------------------------------------------------------------

def check_keys() -> None:
    """Exit with a helpful message if no API key is configured."""
    has_key = any(
        os.environ.get(k)
        for k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
    )
    if not has_key:
        print(
            "Error: no API key found.\n"
            "Set one of:\n"
            "  ANTHROPIC_API_KEY   — for Claude\n"
            "  OPENROUTER_API_KEY  — for OpenRouter (openai/gpt-5.2 default)\n"
            "  OPENAI_API_KEY      — for OpenAI directly\n"
            "\n"
            "Override the model with:  LLM_MODEL=<model-id>"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def sentiment_color(s: str) -> str:
    return {"positive": "green", "negative": "red", "neutral": "yellow"}.get(s, "white")


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    check_keys()

    # Resolve which backend will be used and show it upfront
    model_override = os.environ.get("LLM_MODEL")
    backend = auto_detect_backend(model=model_override)

    print_section("emotion_analysis — demo")
    print(f"Backend : {backend!r}")
    print(f"Transcript: {SAMPLE_PATH.name}")

    # ----- Parse -----
    transcript = parse_transcript(
        load_sample(),
        format="plain",
        domain=ConversationDomain.THERAPY,
        context_note="Individual psychotherapy session, exploring childhood trauma",
    )
    print(f"Parsed {len(transcript.messages)} messages "
          f"from {len(transcript.participants)} participants: "
          f"{', '.join(transcript.participants)}")

    # ----- Analyse -----
    print("\nRunning analysis (2 API calls)...")
    analyzer = EmotionAnalyzer(backend=backend, include_dynamics_narrative=False)
    analysis = analyzer.analyze(transcript)

    # ----- Per-message table -----
    print_section("Per-message annotation")
    if HAS_RICH:
        table = Table(box=box.SIMPLE_HEAVY, show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Speaker", width=12)
        table.add_column("Message (truncated)", width=35)
        table.add_column("Emotion", width=28)
        table.add_column("V", width=6, justify="right")
        table.add_column("A", width=6, justify="right")
        table.add_column("Act", width=14)
        table.add_column("Health", width=7, justify="right")

        for am in analysis.annotated_messages:
            col = sentiment_color(am.sentiment.value)
            table.add_row(
                str(am.message.turn_index),
                f"[{col}]{am.message.speaker}[/{col}]",
                am.message.text[:50] + ("…" if len(am.message.text) > 50 else ""),
                am.emotion.label,
                f"{am.emotion.valence:+.2f}",
                f"{am.emotion.arousal:.2f}",
                am.communicative_act.value,
                f"{am.interaction_health:.1f}",
            )
        rprint(table)
    else:
        for am in analysis.annotated_messages:
            print(
                f"[{am.message.turn_index:>2}] {am.message.speaker:<14} "
                f"val={am.emotion.valence:+.2f}  aro={am.emotion.arousal:.2f}  "
                f"hlth={am.interaction_health:.1f}  "
                f"{am.emotion.label}"
            )

    # ----- Summary -----
    print_section("Conversation summary")
    s = analysis.summary
    print(f"\nOverview:\n  {s.overview}\n")
    print("Key themes:")
    for t in s.key_themes:
        print(f"  • {t}")
    print("\nCommunication patterns:")
    for p in s.communication_patterns:
        print(f"  • {p}")
    print(f"\nPower dynamics:\n  {s.power_dynamics}\n")
    print("Recommendations:")
    for r in s.recommendations:
        print(f"  ✓ {r}")
    print(f"\nEmotional arc:\n  {s.emotional_arc}")

    # ----- Dynamics -----
    print_section("Emotional dynamics")
    d = analysis.dynamics

    print("\nValence trajectories (per speaker):")
    for speaker, vals in d.valence_trajectories.items():
        bar = "  ".join(f"{v:+.2f}" for v in vals)
        print(f"  {speaker:<14} {bar}")

    print(f"\nTurning points detected: {len(d.turning_points)}")
    for tp in d.turning_points:
        print(f"  Turn {tp.turn_index}: {tp.description}")

    print(f"\nEscalation sequences: {len(d.escalation_sequences)}")
    for seq in d.escalation_sequences:
        print(f"  Turns {seq}")

    print(f"\nEmotional contagion score: {d.emotional_contagion_score:.3f}")
    print("  (0 = no synchrony, 1 = speakers perfectly mirror each other)")

    print("\nReactivity matrix (A→B Pearson r or mean |Δ valence|):")
    for sp_a, targets in d.reactivity_matrix.items():
        for sp_b, score in targets.items():
            direction = "positive" if score > 0 else "negative" if score < 0 else "neutral"
            print(f"  {sp_a} → {sp_b}: {score:+.4f}  ({direction} association)")

    # ----- Advanced individual analyses -----
    print_section("Advanced analyses")

    participants = transcript.participants
    if len(participants) >= 2:
        a, b = participants[0], participants[1]
        influence = speaker_influence_score(analysis.annotated_messages, a, b)
        print(f"\n{a} → {b} influence:")
        for k, v in influence.items():
            print(f"  {k}: {v}")

        for sp in participants:
            eri = emotional_regulation_index(analysis.annotated_messages, sp)
            label = "self-regulating" if eri > 0.1 else "dysregulated" if eri < -0.1 else "neutral"
            print(f"\nEmotional regulation index ({sp}): {eri:+.3f}  ({label})")

    momentum = compute_valence_momentum(analysis.annotated_messages, window=4)
    print(f"\nValence momentum (last 5 turns): "
          + "  ".join(f"{m:+.3f}" for m in momentum[-5:]))

    repairs = compute_repair_attempts(analysis.annotated_messages)
    print(f"\nRepair attempts detected: {len(repairs)}")
    for r in repairs:
        print(f"  Turn {r['turn_index']} ({r['speaker']}): {r['reason']}")
        print(f"    \"{r['text'][:70]}\"")

    health_dash = conversation_health_summary(d)
    print("\nHealth dashboard:")
    for k, v in health_dash.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    print()


if __name__ == "__main__":
    main()
