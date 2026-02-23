#!/usr/bin/env python3
"""
Multi-domain demo: run the same analysis pipeline across three different
conversation types (therapy, business negotiation, political debate) and
compare their dynamics profiles.

Run with:
    ANTHROPIC_API_KEY=<your-key> python examples/multi_domain_demo.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_analysis import (
    EmotionAnalyzer,
    conversation_health_summary,
    parse_transcript,
)
from emotion_analysis.models import ConversationDomain

SAMPLE_DIR = Path(__file__).parent / "sample_conversations"

CONFIGS = [
    {
        "file": "therapy_session.txt",
        "format": "plain",
        "domain": ConversationDomain.THERAPY,
        "context_note": "Individual psychotherapy session",
        "label": "Therapy Session",
    },
    {
        "file": "salary_negotiation.txt",
        "format": "plain",
        "domain": ConversationDomain.BUSINESS,
        "context_note": "Annual performance review and compensation negotiation",
        "label": "Salary Negotiation",
    },
    {
        "file": "political_debate.txt",
        "format": "plain",
        "domain": ConversationDomain.POLITICAL,
        "context_note": "Senate committee debate on climate legislation",
        "label": "Political Debate",
    },
]


def hr(char: str = "─", width: int = 70) -> None:
    print(char * width)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    analyzer = EmotionAnalyzer()
    results = []

    for cfg in CONFIGS:
        path = SAMPLE_DIR / cfg["file"]
        print(f"\n[{cfg['label']}] Parsing and analysing…")
        transcript = parse_transcript(
            path.read_text(encoding="utf-8"),
            format=cfg["format"],
            domain=cfg["domain"],
            context_note=cfg["context_note"],
        )
        analysis = analyzer.analyze(transcript)
        results.append((cfg["label"], analysis))
        print(f"  Done. {len(transcript.messages)} turns, "
              f"{len(transcript.participants)} participants.")

    # ----- Comparative summary table -----
    print("\n")
    hr("═")
    print("  COMPARATIVE DYNAMICS ACROSS DOMAINS")
    hr("═")

    header = f"{'Metric':<35}"
    for label, _ in results:
        header += f"  {label[:22]:<22}"
    print(header)
    hr()

    # Collect health dashboards
    dashboards = [(label, conversation_health_summary(a.dynamics)) for label, a in results]

    metrics = [
        ("final_health", "Final health score (0–10)"),
        ("mean_health", "Mean health score"),
        ("min_health", "Lowest point"),
        ("n_turning_points", "Turning points detected"),
        ("n_escalation_sequences", "Escalation sequences"),
        ("emotional_contagion", "Emotional contagion score"),
    ]

    for key, display_name in metrics:
        row = f"  {display_name:<33}"
        for _, dash in dashboards:
            val = dash.get(key, "—")
            row += f"  {str(val):<22}"
        print(row)

    hr()

    # Per-speaker mean valence
    print("\n  MEAN VALENCE PER SPEAKER")
    hr()
    for label, analysis in results:
        d = analysis.dynamics
        print(f"\n  {label}:")
        for sp, mv in d.mean_valence.items():
            bar = "█" * int((mv + 1) * 10) + "░" * int((1 - mv) * 10)
            print(f"    {sp:<16} {mv:+.3f}  {bar}")

    # Dominant themes
    print("\n\n  KEY THEMES")
    hr()
    for label, analysis in results:
        print(f"\n  {label}:")
        for theme in analysis.summary.key_themes[:4]:
            print(f"    • {theme}")

    # Recommendations
    print("\n\n  TOP RECOMMENDATION PER DOMAIN")
    hr()
    for label, analysis in results:
        recs = analysis.summary.recommendations
        print(f"\n  {label}:")
        if recs:
            print(f"    → {recs[0]}")

    print()


if __name__ == "__main__":
    main()
