"""
Core analysis engine.

EmotionAnalyzer is the primary entry point for callers.  It takes a
ConversationTranscript and returns a full ConversationAnalysis, including:
  - per-message emotion vectors and communicative labels
  - a conversation-level summary with themes, patterns, and recommendations
  - computed emotional dynamics (trajectories, reactivity, turning points, etc.)

All LLM calls go through the Anthropic SDK.  The model is configurable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import anthropic

from .dynamics import compute_dynamics
from .models import (
    AnnotatedMessage,
    CommunicativeAct,
    ConversationAnalysis,
    ConversationSummary,
    ConversationTranscript,
    EmotionVector,
    Sentiment,
)
from .prompts import (
    SYSTEM_PROMPT,
    _format_transcript,
    build_annotation_prompt,
    build_dynamics_narrative_prompt,
    build_summary_prompt,
)

logger = logging.getLogger(__name__)

# Default model — can be overridden per-instance or per-call
DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_MAX_TOKENS = 8192


class AnalysisError(Exception):
    """Raised when the LLM returns output that cannot be parsed."""


class EmotionAnalyzer:
    """
    Analyse conversational transcripts for emotion, dynamics, and narrative summary.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY`` env var if
        not provided.
    model:
        Claude model ID to use.  Defaults to ``claude-opus-4-6``.
    max_tokens:
        Token budget for each LLM call.
    include_dynamics_narrative:
        If True, a third LLM call generates a natural-language interpretation of
        the computed dynamics (turning points, reactivity).  Useful for display;
        can be skipped to reduce latency and cost.
    client:
        Optionally pass a pre-configured ``anthropic.Anthropic`` client (useful
        for testing or custom retry policies).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        include_dynamics_narrative: bool = False,
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.include_dynamics_narrative = include_dynamics_narrative
        self._client = client or anthropic.Anthropic(
            **({"api_key": api_key} if api_key else {})
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        transcript: ConversationTranscript,
        *,
        include_dynamics_narrative: Optional[bool] = None,
    ) -> ConversationAnalysis:
        """
        Run the full analysis pipeline on a transcript.

        Steps
        -----
        1. LLM call: annotate every message with emotion vectors.
        2. Compute emotional dynamics from the annotations (pure Python — no LLM).
        3. LLM call: generate a high-level narrative summary.
        4. (Optional) LLM call: generate natural-language dynamics narrative.

        Parameters
        ----------
        transcript:
            The conversation to analyse.
        include_dynamics_narrative:
            Override instance default for whether to generate a narrative
            interpretation of the computed dynamics.

        Returns
        -------
        ConversationAnalysis
        """
        do_narrative = (
            include_dynamics_narrative
            if include_dynamics_narrative is not None
            else self.include_dynamics_narrative
        )

        logger.info(
            "Analysing transcript with %d messages, domain=%s",
            len(transcript.messages),
            transcript.domain.value,
        )

        # Step 1: per-message annotation
        annotated = self._annotate_messages(transcript)

        # Step 2: compute dynamics (no LLM)
        dynamics = compute_dynamics(annotated, transcript.participants)

        # Step 3: conversation-level summary
        summary = self._generate_summary(transcript, annotated)

        # Step 4: optional dynamics narrative
        if do_narrative and (dynamics.turning_points or dynamics.reactivity_events):
            narrative = self._generate_dynamics_narrative(
                transcript, dynamics.turning_points, dynamics.reactivity_events
            )
            dynamics = dynamics.model_copy(
                update={"_narrative": narrative}  # stored as private attr if desired
            )

        return ConversationAnalysis(
            transcript=transcript,
            annotated_messages=annotated,
            dynamics=dynamics,
            summary=summary,
        )

    def annotate_messages(
        self, transcript: ConversationTranscript
    ) -> List[AnnotatedMessage]:
        """Expose just the message-level annotation step (no dynamics/summary)."""
        return self._annotate_messages(transcript)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call(self, user_prompt: str, system: str = SYSTEM_PROMPT) -> str:
        """Make a single Anthropic API call and return the text content."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _parse_json(self, raw: str, context: str = "") -> Any:
        """Extract and parse JSON from a model response, with error context."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise AnalysisError(
                f"Failed to parse JSON from model output ({context}): {exc}\n"
                f"Raw output:\n{raw[:500]}"
            ) from exc

    def _annotate_messages(
        self, transcript: ConversationTranscript
    ) -> List[AnnotatedMessage]:
        """Call the LLM to annotate every message with emotion vectors."""
        prompt = build_annotation_prompt(
            messages=transcript.messages,
            domain=transcript.domain,
            context_note=transcript.context_note,
            participants=transcript.participants,
        )
        raw = self._call(prompt)
        data = self._parse_json(raw, context="message annotation")

        annotations_raw: List[Dict[str, Any]] = data.get("annotations", data)
        if not isinstance(annotations_raw, list):
            raise AnalysisError(
                "Expected a list of annotations; got: " + repr(type(annotations_raw))
            )

        # Build a quick lookup from the transcript
        msg_by_index = {m.turn_index: m for m in transcript.messages}

        annotated: List[AnnotatedMessage] = []
        for item in annotations_raw:
            turn_index = item.get("turn_index")
            msg = msg_by_index.get(turn_index)
            if msg is None:
                logger.warning("Annotation references unknown turn_index=%r — skipping", turn_index)
                continue

            emotion_raw = item.get("emotion", {})
            emotion = EmotionVector(
                valence=_clamp(emotion_raw.get("valence", 0.0), -1.0, 1.0),
                arousal=_clamp(emotion_raw.get("arousal", 0.5), 0.0, 1.0),
                dominance=_clamp(emotion_raw.get("dominance", 0.0), -1.0, 1.0),
                joy=_clamp01(emotion_raw.get("joy", 0.0)),
                sadness=_clamp01(emotion_raw.get("sadness", 0.0)),
                anger=_clamp01(emotion_raw.get("anger", 0.0)),
                fear=_clamp01(emotion_raw.get("fear", 0.0)),
                surprise=_clamp01(emotion_raw.get("surprise", 0.0)),
                disgust=_clamp01(emotion_raw.get("disgust", 0.0)),
                contempt=_clamp01(emotion_raw.get("contempt", 0.0)),
                trust=_clamp01(emotion_raw.get("trust", 0.0)),
                anticipation=_clamp01(emotion_raw.get("anticipation", 0.0)),
                assertiveness=_clamp01(emotion_raw.get("assertiveness", 0.0)),
                openness=_clamp01(emotion_raw.get("openness", 0.0)),
                hostility=_clamp01(emotion_raw.get("hostility", 0.0)),
                vulnerability=_clamp01(emotion_raw.get("vulnerability", 0.0)),
                label=str(emotion_raw.get("label", "unspecified")),
                confidence=_clamp01(emotion_raw.get("confidence", 1.0)),
            )

            sentiment_raw = item.get("sentiment", "neutral").lower()
            try:
                sentiment = Sentiment(sentiment_raw)
            except ValueError:
                sentiment = Sentiment.NEUTRAL

            act_raw = item.get("communicative_act", "other").lower()
            try:
                act = CommunicativeAct(act_raw)
            except ValueError:
                act = CommunicativeAct.OTHER

            annotated.append(
                AnnotatedMessage(
                    message=msg,
                    emotion=emotion,
                    sentiment=sentiment,
                    communicative_act=act,
                    interaction_health=_clamp(
                        float(item.get("interaction_health", 5.0)), 0.0, 10.0
                    ),
                    notes=str(item.get("notes", "")),
                )
            )

        if not annotated:
            raise AnalysisError("No annotations were produced. Check the model response.")

        return annotated

    def _generate_summary(
        self,
        transcript: ConversationTranscript,
        annotated: List[AnnotatedMessage],
    ) -> ConversationSummary:
        """Generate the high-level narrative summary."""
        transcript_block = _format_transcript(transcript.messages)
        annotations_json = json.dumps(
            [
                {
                    "turn_index": am.message.turn_index,
                    "speaker": am.message.speaker,
                    "sentiment": am.sentiment.value,
                    "communicative_act": am.communicative_act.value,
                    "interaction_health": am.interaction_health,
                    "emotion_label": am.emotion.label,
                    "valence": am.emotion.valence,
                    "notes": am.notes,
                }
                for am in annotated
            ],
            indent=2,
        )
        prompt = build_summary_prompt(
            transcript_block=transcript_block,
            annotations_json=annotations_json,
            domain=transcript.domain,
            participants=transcript.participants,
            context_note=transcript.context_note,
        )
        raw = self._call(prompt)
        data = self._parse_json(raw, context="summary generation")

        return ConversationSummary(
            overview=str(data.get("overview", "")),
            key_themes=list(data.get("key_themes", [])),
            communication_patterns=list(data.get("communication_patterns", [])),
            power_dynamics=str(data.get("power_dynamics", "")),
            recommendations=list(data.get("recommendations", [])),
            emotional_arc=str(data.get("emotional_arc", "")),
        )

    def _generate_dynamics_narrative(
        self,
        transcript: ConversationTranscript,
        turning_points: list,
        reactivity_events: list,
    ) -> str:
        """Generate an optional natural-language narrative about dynamics."""
        transcript_block = _format_transcript(transcript.messages)
        tp_json = json.dumps(
            [tp.model_dump(exclude_none=True) for tp in turning_points], indent=2
        )
        re_json = json.dumps(
            [re_.model_dump(exclude_none=True) for re_ in reactivity_events], indent=2
        )
        prompt = build_dynamics_narrative_prompt(
            transcript_block=transcript_block,
            turning_points_json=tp_json,
            reactivity_json=re_json,
            domain=transcript.domain,
            participants=transcript.participants,
        )
        raw = self._call(prompt)
        data = self._parse_json(raw, context="dynamics narrative")
        return str(data.get("dynamics_narrative", ""))


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return (lo + hi) / 2.0


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)
