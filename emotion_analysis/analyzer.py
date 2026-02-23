"""
Core analysis engine.

EmotionAnalyzer is the primary entry point for callers.  It takes a
ConversationTranscript and returns a full ConversationAnalysis, including:
  - per-message emotion vectors and communicative labels
  - a conversation-level summary with themes, patterns, and recommendations
  - computed emotional dynamics (trajectories, reactivity, turning points, etc.)

LLM provider is configurable via the ``backend`` parameter — any backend that
implements ``complete(system, user) -> str`` is accepted.  Built-in backends:
  AnthropicBackend  — Anthropic Messages API (default when ANTHROPIC_API_KEY set)
  OpenAIBackend     — any OpenAI-compatible endpoint (default when OPENROUTER_API_KEY set)

When no backend is supplied, auto_detect_backend() picks one from env vars.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from .backends import (
    AnthropicBackend,
    LLMBackend,
    OpenAIBackend,
    auto_detect_backend,
)
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

DEFAULT_MAX_TOKENS = 8192


class AnalysisError(Exception):
    """Raised when the LLM returns output that cannot be parsed."""


class EmotionAnalyzer:
    """
    Analyse conversational transcripts for emotion, dynamics, and narrative summary.

    Parameters
    ----------
    backend:
        How to reach an LLM.  Three forms are accepted:

        * ``None`` (default) — auto-detect from environment variables
          (ANTHROPIC_API_KEY → Claude, OPENROUTER_API_KEY → OpenRouter,
          OPENAI_API_KEY → OpenAI).
        * A string shortcut: ``"anthropic"`` or ``"openai"``.
        * An ``LLMBackend`` instance (``AnthropicBackend``, ``OpenAIBackend``,
          or any custom object with a ``complete(system, user) -> str`` method).

    api_key:
        API key forwarded to the auto-detected or shortcut backend.  Ignored
        when a fully-configured backend instance is passed.
    model:
        Model ID forwarded to the auto-detected or shortcut backend.
    max_tokens:
        Token budget for each LLM call.
    include_dynamics_narrative:
        If True, a third LLM call generates a natural-language interpretation of
        the computed dynamics (turning points, reactivity).
    client:
        Pre-configured SDK client (``anthropic.Anthropic`` or ``openai.OpenAI``).
        When supplied, it is wrapped in the appropriate backend automatically.

    Examples
    --------
    Auto-detect (recommended)::

        analyzer = EmotionAnalyzer()

    Explicit Anthropic::

        analyzer = EmotionAnalyzer(backend="anthropic", model="claude-haiku-4-5-20251001")

    OpenRouter with a specific model::

        analyzer = EmotionAnalyzer(backend="openai", model="google/gemini-2.5-pro")

    Ollama (local)::

        from emotion_analysis.backends import OpenAIBackend
        backend = OpenAIBackend(
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key="ollama",
        )
        analyzer = EmotionAnalyzer(backend=backend)
    """

    def __init__(
        self,
        backend: Union[str, LLMBackend, None] = None,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        include_dynamics_narrative: bool = False,
        # Legacy convenience: pass a pre-built SDK client
        client: Optional[Any] = None,
    ) -> None:
        self.include_dynamics_narrative = include_dynamics_narrative
        self._backend = self._resolve_backend(
            backend, api_key=api_key, model=model,
            max_tokens=max_tokens, client=client,
        )
        logger.debug("EmotionAnalyzer ready: backend=%r", self._backend)

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
        2. Compute emotional dynamics (pure Python, no LLM).
        3. LLM call: generate a high-level narrative summary.
        4. (Optional) LLM call: natural-language dynamics narrative.

        Parameters
        ----------
        transcript:
            The conversation to analyse.
        include_dynamics_narrative:
            Override the instance-level setting.

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
            "Analysing transcript with %d messages, domain=%s, backend=%r",
            len(transcript.messages),
            transcript.domain.value,
            self._backend,
        )

        annotated = self._annotate_messages(transcript)
        dynamics = compute_dynamics(annotated, transcript.participants)
        summary = self._generate_summary(transcript, annotated)

        if do_narrative and (dynamics.turning_points or dynamics.reactivity_events):
            self._generate_dynamics_narrative(
                transcript, dynamics.turning_points, dynamics.reactivity_events
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

    @property
    def backend(self) -> LLMBackend:
        """The active LLM backend."""
        return self._backend

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(
        backend: Union[str, LLMBackend, None],
        *,
        api_key: Optional[str],
        model: Optional[str],
        max_tokens: int,
        client: Optional[Any],
    ) -> LLMBackend:
        if isinstance(backend, LLMBackend):
            return backend

        if backend is None:
            return auto_detect_backend(
                api_key=api_key, model=model,
                max_tokens=max_tokens, client=client,
            )

        if backend == "anthropic":
            return AnthropicBackend(
                api_key=api_key,
                model=model or AnthropicBackend.DEFAULT_MODEL,
                max_tokens=max_tokens,
                client=client,
            )

        if backend in ("openai", "openrouter"):
            return OpenAIBackend(
                api_key=api_key,
                model=model or OpenAIBackend.DEFAULT_MODEL,
                max_tokens=max_tokens,
            )

        raise ValueError(
            f"Unknown backend string {backend!r}. "
            "Use 'anthropic', 'openai', or pass an LLMBackend instance."
        )

    def _call(self, user_prompt: str, system: str = SYSTEM_PROMPT) -> str:
        return self._backend.complete(system, user_prompt)

    def _parse_json(self, raw: str, context: str = "") -> Any:
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

        msg_by_index = {m.turn_index: m for m in transcript.messages}
        annotated: List[AnnotatedMessage] = []

        for item in annotations_raw:
            turn_index = item.get("turn_index")
            msg = msg_by_index.get(turn_index)
            if msg is None:
                logger.warning(
                    "Annotation references unknown turn_index=%r — skipping", turn_index
                )
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
