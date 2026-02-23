#!/usr/bin/env python3
"""
Live smoke test against OpenRouter.

Run with:
    OPENROUTER_API_KEY=<your-key> python examples/openrouter_smoke_test.py

Optionally override the model:
    OPENROUTER_API_KEY=<key> OPENROUTER_MODEL=anthropic/claude-3.5-sonnet \
        python examples/openrouter_smoke_test.py

Tests that:
1. OpenAIBackend connects to OpenRouter and returns a valid completion.
2. EmotionAnalyzer with backend="openai" runs the full pipeline end-to-end.
3. Auto-detection picks OpenRouter when OPENROUTER_API_KEY is set and
   ANTHROPIC_API_KEY is not.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_analysis import EmotionAnalyzer, parse_transcript
from emotion_analysis.backends import OpenAIBackend, auto_detect_backend
from emotion_analysis.models import ConversationDomain

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.2")

SHORT_TRANSCRIPT = """\
Alice: I feel like you never listen when I bring up concerns about the project timeline.
Bob: That's not fair. I attended every single planning meeting.
Alice: Attending isn't the same as actually engaging with what I'm saying.
Bob: Fine. What specifically do you think I missed?
Alice: The risk around the API integration — I raised it three times and it went nowhere.
Bob: Okay. That one I admit I underestimated. That's on me.
"""


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        sys.exit(1)


def main() -> None:
    if not OPENROUTER_KEY:
        print("OPENROUTER_API_KEY is not set. Skipping live test.")
        sys.exit(0)

    print(f"=== OpenRouter smoke test  (model: {MODEL}) ===\n")

    # ------------------------------------------------------------------
    # Test 1: raw backend completion
    # ------------------------------------------------------------------
    print("1. Raw OpenAIBackend.complete() …")
    backend = OpenAIBackend(
        api_key=OPENROUTER_KEY,
        model=MODEL,
        extra_headers={
            "HTTP-Referer": "https://github.com/samrahimi/lovegpt",
            "X-Title": "emotion-analysis smoke test",
        },
    )
    result = backend.complete(
        system="You are a helpful assistant. Reply with exactly one word.",
        user="Say 'hello'.",
    )
    check(isinstance(result, str) and len(result) > 0, f"Got non-empty response: {result!r}")

    # ------------------------------------------------------------------
    # Test 2: full EmotionAnalyzer pipeline with explicit backend="openai"
    # ------------------------------------------------------------------
    print("\n2. EmotionAnalyzer(backend='openai') full pipeline …")
    transcript = parse_transcript(
        SHORT_TRANSCRIPT,
        format="plain",
        domain=ConversationDomain.BUSINESS,
        context_note="Project retrospective between two team leads",
    )
    analyzer = EmotionAnalyzer(
        backend="openai",
        model=MODEL,
    )
    check(repr(analyzer.backend).startswith("OpenAIBackend"), "backend is OpenAIBackend")

    analysis = analyzer.analyze(transcript)

    check(len(analysis.annotated_messages) == len(transcript.messages),
          f"All {len(transcript.messages)} messages annotated")
    check(all(-1.0 <= am.emotion.valence <= 1.0 for am in analysis.annotated_messages),
          "All valence scores in [-1, 1]")
    check(all(0.0 <= am.interaction_health <= 10.0 for am in analysis.annotated_messages),
          "All health scores in [0, 10]")
    check(bool(analysis.summary.overview), "Summary overview non-empty")
    check(len(analysis.summary.recommendations) > 0, "Got at least one recommendation")
    check(len(analysis.dynamics.turning_points) >= 0, "Turning points computed (may be 0)")
    check("reactivity_matrix" in analysis.dynamics.model_fields_set
          or analysis.dynamics.reactivity_matrix is not None,
          "Reactivity matrix present")

    print("\n  Sample annotations:")
    for am in analysis.annotated_messages:
        print(f"    [{am.message.turn_index}] {am.message.speaker}: "
              f"{am.emotion.label} | v={am.emotion.valence:+.2f} "
              f"h={am.interaction_health:.1f} | {am.communicative_act.value}")

    print(f"\n  Overview: {analysis.summary.overview[:120]}…")
    print(f"  Recommendations:")
    for r in analysis.summary.recommendations:
        print(f"    • {r}")

    # ------------------------------------------------------------------
    # Test 3: auto-detection picks OpenRouter when ANTHROPIC_API_KEY absent
    # ------------------------------------------------------------------
    print("\n3. auto_detect_backend() with no ANTHROPIC_API_KEY …")
    original_ant = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        detected = auto_detect_backend()
        check(isinstance(detected, OpenAIBackend),
              f"Auto-detected OpenAIBackend (got {type(detected).__name__})")
    finally:
        if original_ant:
            os.environ["ANTHROPIC_API_KEY"] = original_ant

    print("\n=== All tests passed ===\n")


if __name__ == "__main__":
    main()
