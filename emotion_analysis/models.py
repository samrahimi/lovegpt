"""
Core data models for the emotion_analysis library.

Emotion representation is grounded in established psychological frameworks:
  - PAD (Pleasure-Arousal-Dominance) affective space
  - Plutchik's wheel of emotions (basic emotion categories)
  - Communicative intent dimensions for discourse analysis
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class CommunicativeAct(str, Enum):
    """
    Speech act taxonomy broadly applicable across therapeutic, business,
    political, and personal contexts.
    """
    ASSERTION = "assertion"           # stating a fact or belief
    REQUEST = "request"               # asking for something
    QUESTION = "question"             # seeking information
    APOLOGY = "apology"               # expressing regret
    COMPLAINT = "complaint"           # expressing dissatisfaction
    CRITICISM = "criticism"           # negative evaluation of other
    PRAISE = "praise"                 # positive evaluation of other
    THREAT = "threat"                 # signalling negative consequence
    PROMISE = "promise"               # commitment to future action
    CONCESSION = "concession"         # yielding a point
    DEFLECTION = "deflection"         # avoiding direct response
    DISCLOSURE = "disclosure"         # sharing personal information
    VALIDATION = "validation"         # affirming the other's perspective
    ACCUSATION = "accusation"         # attributing blame
    JUSTIFICATION = "justification"   # defending own position
    COMMAND = "command"               # directive with authority
    EMPATHY = "empathy"               # expressing understanding
    SARCASM = "sarcasm"               # ironic/mocking statement
    OTHER = "other"


class ConversationDomain(str, Enum):
    """Broad domain context for domain-informed analysis."""
    THERAPY = "therapy"
    ROMANTIC = "romantic"
    FAMILY = "family"
    FRIENDSHIP = "friendship"
    BUSINESS = "business"
    POLITICAL = "political"
    MEDICAL = "medical"
    LEGAL = "legal"
    EDUCATION = "education"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Emotion representation
# ---------------------------------------------------------------------------

class EmotionVector(BaseModel):
    """
    Multi-dimensional representation of emotional state.

    Core affective dimensions follow the PAD (Pleasure-Arousal-Dominance)
    model, which is both theoretically grounded and useful for computation.

    Basic emotion scores (0–1) follow Plutchik's primary emotion categories.

    Communicative dimensions (0–1) capture how the emotion is expressed
    in discourse — useful for conflict, therapy, and business contexts.
    """

    # ---- PAD affective space ------------------------------------------------
    valence: float = Field(..., ge=-1.0, le=1.0,
        description="Hedonic tone: -1 (very negative) to +1 (very positive)")
    arousal: float = Field(..., ge=0.0, le=1.0,
        description="Activation level: 0 (very calm) to 1 (very excited/agitated)")
    dominance: float = Field(..., ge=-1.0, le=1.0,
        description="Social dominance: -1 (submissive/powerless) to +1 (dominant/in-control)")

    # ---- Basic emotions (Plutchik) ------------------------------------------
    joy: float = Field(0.0, ge=0.0, le=1.0)
    sadness: float = Field(0.0, ge=0.0, le=1.0)
    anger: float = Field(0.0, ge=0.0, le=1.0)
    fear: float = Field(0.0, ge=0.0, le=1.0)
    surprise: float = Field(0.0, ge=0.0, le=1.0)
    disgust: float = Field(0.0, ge=0.0, le=1.0)
    contempt: float = Field(0.0, ge=0.0, le=1.0)
    trust: float = Field(0.0, ge=0.0, le=1.0)
    anticipation: float = Field(0.0, ge=0.0, le=1.0)

    # ---- Communicative dimensions -------------------------------------------
    assertiveness: float = Field(0.0, ge=0.0, le=1.0,
        description="Degree of confident, direct self-expression")
    openness: float = Field(0.0, ge=0.0, le=1.0,
        description="Receptivity, willingness to hear the other party")
    hostility: float = Field(0.0, ge=0.0, le=1.0,
        description="Adversarial or aggressive orientation toward the listener")
    vulnerability: float = Field(0.0, ge=0.0, le=1.0,
        description="Emotional exposure, willingness to show weakness or need")

    # ---- Meta ---------------------------------------------------------------
    label: str = Field(...,
        description="Concise natural-language description of this emotional state")
    confidence: float = Field(1.0, ge=0.0, le=1.0,
        description="Model confidence in this classification (0–1)")

    def distance_to(self, other: "EmotionVector") -> float:
        """Euclidean distance in PAD space — useful for measuring emotional shift."""
        return (
            (self.valence - other.valence) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        ) ** 0.5

    def pad_tuple(self) -> Tuple[float, float, float]:
        """Return (valence, arousal, dominance) for vectorised operations."""
        return (self.valence, self.arousal, self.dominance)

    def dominant_basic_emotion(self) -> str:
        """Return the name of the highest-scoring basic emotion."""
        candidates = {
            "joy": self.joy, "sadness": self.sadness, "anger": self.anger,
            "fear": self.fear, "surprise": self.surprise, "disgust": self.disgust,
            "contempt": self.contempt, "trust": self.trust,
            "anticipation": self.anticipation,
        }
        return max(candidates, key=candidates.__getitem__)


# ---------------------------------------------------------------------------
# Transcript / Message primitives
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single conversational turn, format-agnostic."""
    turn_index: int = Field(..., description="Zero-based position in the conversation")
    speaker: str = Field(..., description="Speaker identifier (name, role, or alias)")
    text: str = Field(..., description="Raw message content")
    timestamp: Optional[datetime] = Field(None, description="Wall-clock time if available")
    metadata: Dict[str, Any] = Field(default_factory=dict,
        description="Arbitrary extra data (platform, channel, message ID, etc.)")


class ConversationTranscript(BaseModel):
    """
    An ordered sequence of messages ready for analysis.
    All participants are enumerated; order is preserved.
    """
    messages: List[Message]
    participants: List[str] = Field(default_factory=list)
    domain: ConversationDomain = ConversationDomain.GENERAL
    context_note: Optional[str] = Field(None,
        description="Optional free-text context provided by the caller "
                    "(e.g. 'negotiation over budget', 'couples therapy intake')")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _infer_participants(self) -> "ConversationTranscript":
        if not self.participants:
            seen: List[str] = []
            for m in self.messages:
                if m.speaker not in seen:
                    seen.append(m.speaker)
            self.participants = seen
        return self

    def turns_by(self, speaker: str) -> List[Message]:
        return [m for m in self.messages if m.speaker == speaker]

    def turn_pairs(self) -> List[Tuple[Message, Message]]:
        """
        Return all consecutive (A→B) turn pairs where A != B.
        Used in reactivity analysis.
        """
        pairs = []
        for i in range(len(self.messages) - 1):
            a, b = self.messages[i], self.messages[i + 1]
            if a.speaker != b.speaker:
                pairs.append((a, b))
        return pairs


# ---------------------------------------------------------------------------
# Analysis output types
# ---------------------------------------------------------------------------

class AnnotatedMessage(BaseModel):
    """A raw message enriched with emotion analysis."""
    message: Message
    emotion: EmotionVector
    sentiment: Sentiment
    communicative_act: CommunicativeAct
    interaction_health: float = Field(..., ge=0.0, le=10.0,
        description="Running health score for the conversation at this point "
                    "(0 = severely dysfunctional, 10 = highly constructive)")
    notes: str = Field("",
        description="Analyst commentary explaining the score and emotion")


class TurningPoint(BaseModel):
    """
    A moment in the conversation where the emotional trajectory shifts
    significantly — positively or negatively.
    """
    turn_index: int
    magnitude: float = Field(..., ge=0.0,
        description="Size of the emotional shift in PAD space")
    direction: Sentiment = Field(...,
        description="Whether the shift was toward positive or negative valence")
    trigger_speaker: Optional[str] = None
    description: str = ""


class ReactivityEvent(BaseModel):
    """
    One detected instance of emotional reactivity:
    Speaker A says something → Speaker B's next turn shows a measurable
    emotional change attributable to A's statement.
    """
    trigger_turn_index: int
    trigger_speaker: str
    trigger_emotion: EmotionVector
    response_turn_index: int
    response_speaker: str
    response_emotion: EmotionVector
    baseline_emotion: Optional[EmotionVector] = Field(None,
        description="B's emotional baseline before this event (previous B turn)")
    valence_delta: float = Field(...,
        description="Change in B's valence relative to baseline (positive = improvement)")
    arousal_delta: float = Field(...,
        description="Change in B's arousal relative to baseline")
    reactivity_score: float = Field(..., ge=0.0, le=1.0,
        description="Normalised magnitude of the reactive shift (0 = no change, 1 = maximal)")
    description: str = ""


class EmotionalDynamics(BaseModel):
    """
    Aggregate temporal and relational statistics derived from the sequence
    of annotated messages.  All indices refer to turn_index values.
    """

    # ---- Per-speaker temporal trajectories ----------------------------------
    valence_trajectories: Dict[str, List[float]] = Field(default_factory=dict,
        description="speaker → ordered list of valence values across their turns")
    arousal_trajectories: Dict[str, List[float]] = Field(default_factory=dict)
    dominance_trajectories: Dict[str, List[float]] = Field(default_factory=dict)
    interaction_health_trajectory: List[float] = Field(default_factory=list,
        description="Conversation-level health score at each turn")

    # ---- Turning points -----------------------------------------------------
    turning_points: List[TurningPoint] = Field(default_factory=list)

    # ---- Reactivity ---------------------------------------------------------
    reactivity_events: List[ReactivityEvent] = Field(default_factory=list)
    reactivity_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict,
        description="speaker_a → speaker_b → average reactivity score. "
                    "Measures how strongly A's turns shift B's valence.")

    # ---- Escalation / de-escalation sequences --------------------------------
    escalation_sequences: List[List[int]] = Field(default_factory=list,
        description="Each inner list is a run of turn indices where valence "
                    "falls monotonically (escalating negativity)")
    de_escalation_sequences: List[List[int]] = Field(default_factory=list,
        description="Runs of turn indices where valence rises monotonically")

    # ---- Aggregate statistics -----------------------------------------------
    mean_valence: Dict[str, float] = Field(default_factory=dict,
        description="Average valence per speaker across all their turns")
    valence_variance: Dict[str, float] = Field(default_factory=dict,
        description="Variance in valence per speaker (high = emotionally volatile)")
    emotional_contagion_score: float = Field(0.0, ge=0.0, le=1.0,
        description="Overall degree to which speakers mirror each other's valence "
                    "(0 = no synchrony, 1 = perfect mirroring)")
    dominant_sentiment: Dict[str, Sentiment] = Field(default_factory=dict,
        description="Most frequent sentiment label per speaker")


class ConversationSummary(BaseModel):
    """High-level narrative summary of a conversation."""
    overview: str = Field(..., description="What this conversation is about")
    key_themes: List[str] = Field(default_factory=list)
    communication_patterns: List[str] = Field(default_factory=list,
        description="Recurring behavioural patterns observed (positive and negative)")
    power_dynamics: str = Field("",
        description="Analysis of dominance/submission patterns between speakers")
    recommendations: List[str] = Field(default_factory=list,
        description="Actionable suggestions for improving future interactions")
    emotional_arc: str = Field("",
        description="Narrative description of how the emotional tone evolved")


class ConversationAnalysis(BaseModel):
    """
    Complete analysis output.  This is the top-level object returned by
    EmotionAnalyzer.analyze().
    """
    transcript: ConversationTranscript
    annotated_messages: List[AnnotatedMessage]
    dynamics: EmotionalDynamics
    summary: ConversationSummary

    def message_at(self, turn_index: int) -> Optional[AnnotatedMessage]:
        for am in self.annotated_messages:
            if am.message.turn_index == turn_index:
                return am
        return None

    def speaker_arc(self, speaker: str) -> List[AnnotatedMessage]:
        """Return all annotated turns for a given speaker, in order."""
        return [am for am in self.annotated_messages
                if am.message.speaker == speaker]

    def health_at(self, turn_index: int) -> Optional[float]:
        am = self.message_at(turn_index)
        return am.interaction_health if am else None
