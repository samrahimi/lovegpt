"""
emotion_analysis
================

A Python library for multi-dimensional emotion analysis of conversational
transcripts — domain-agnostic and applicable to therapy, business, politics,
personal relationships, and any structured human dialogue.

Quick start
-----------
::

    from emotion_analysis import EmotionAnalyzer, parse_transcript
    from emotion_analysis.models import ConversationDomain

    transcript = parse_transcript(
        \"\"\"
        Alice: I feel like my concerns are never taken seriously in these meetings.
        Bob: That's not true — I always listen to everyone.
        Alice: Really? Because last Tuesday you cut me off mid-sentence.
        Bob: I was just trying to keep us on schedule.
        \"\"\",
        format="plain",
        domain=ConversationDomain.BUSINESS,
        context_note="Post-meeting debrief between team members",
    )

    analyzer = EmotionAnalyzer()  # uses ANTHROPIC_API_KEY env var
    analysis = analyzer.analyze(transcript)

    for am in analysis.annotated_messages:
        print(f"[{am.message.turn_index}] {am.message.speaker}: "
              f"{am.emotion.label} (valence={am.emotion.valence:+.2f}, "
              f"health={am.interaction_health:.1f})")

    print(analysis.summary.overview)
    print(analysis.summary.recommendations)

Public API surface
------------------
Core classes and functions re-exported here for convenient top-level access.
Everything else (prompts, internal helpers) lives in the sub-modules.
"""

from .analyzer import AnalysisError, EmotionAnalyzer
from .backends import AnthropicBackend, LLMBackend, OpenAIBackend, auto_detect_backend
from .dynamics import (
    compute_dynamics,
    compute_repair_attempts,
    compute_valence_momentum,
    conversation_health_summary,
    emotional_regulation_index,
    speaker_influence_score,
)
from .models import (
    AnnotatedMessage,
    CommunicativeAct,
    ConversationAnalysis,
    ConversationDomain,
    ConversationSummary,
    ConversationTranscript,
    EmotionalDynamics,
    EmotionVector,
    Message,
    ReactivityEvent,
    Sentiment,
    TurningPoint,
)
from .parsers import (
    parse_csv,
    parse_json,
    parse_list,
    parse_plain,
    parse_slack_json,
    parse_telegram,
    parse_transcript,
    parse_whatsapp,
)

__all__ = [
    # Analyser
    "EmotionAnalyzer",
    "AnalysisError",
    # Backends
    "LLMBackend",
    "AnthropicBackend",
    "OpenAIBackend",
    "auto_detect_backend",
    # Parsers
    "parse_transcript",
    "parse_plain",
    "parse_whatsapp",
    "parse_telegram",
    "parse_slack_json",
    "parse_json",
    "parse_csv",
    "parse_list",
    # Dynamics
    "compute_dynamics",
    "compute_valence_momentum",
    "compute_repair_attempts",
    "speaker_influence_score",
    "emotional_regulation_index",
    "conversation_health_summary",
    # Models
    "ConversationAnalysis",
    "ConversationSummary",
    "ConversationTranscript",
    "EmotionalDynamics",
    "EmotionVector",
    "AnnotatedMessage",
    "Message",
    "TurningPoint",
    "ReactivityEvent",
    "Sentiment",
    "CommunicativeAct",
    "ConversationDomain",
]

__version__ = "0.1.0"
