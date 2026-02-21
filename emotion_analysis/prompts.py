"""
Prompt templates used by the analysis engine.

Keeping prompts here (rather than inline in analyzer.py) makes it easy to
swap, version, or fine-tune them independently of the API call logic.
"""

from __future__ import annotations

from typing import List

from .models import ConversationDomain, Message


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert computational psycholinguist specialising in affective analysis \
of human conversation. Your role is to annotate conversational transcripts with \
precise, multi-dimensional emotion vectors and contextual commentary.

You work across domains: psychotherapy, couples counselling, political discourse, \
organisational behaviour, medical consultation, legal proceedings, and general \
interpersonal communication. Your analysis is domain-aware but always grounded in \
empirically validated emotion frameworks (PAD model, Plutchik's wheel, speech-act theory).

Output must always be valid JSON, exactly matching the schema requested. \
Do not include any text outside the JSON structure.
"""


# ---------------------------------------------------------------------------
# Message-level annotation prompt
# ---------------------------------------------------------------------------

def build_annotation_prompt(
    messages: List[Message],
    domain: ConversationDomain,
    context_note: str | None,
    participants: List[str],
) -> str:
    """
    Build the user-turn prompt for per-message emotion annotation.

    The model receives the full transcript and is asked to return a JSON array
    where each element corresponds to one message, in order.
    """
    domain_guidance = _domain_guidance(domain)
    participant_list = ", ".join(participants)
    context_block = f"\nContext provided by the requester: {context_note}" if context_note else ""

    transcript_block = _format_transcript(messages)

    schema = """
{
  "annotations": [
    {
      "turn_index": <int>,
      "speaker": "<string>",
      "emotion": {
        "valence": <float -1.0 to 1.0>,
        "arousal": <float 0.0 to 1.0>,
        "dominance": <float -1.0 to 1.0>,
        "joy": <float 0.0 to 1.0>,
        "sadness": <float 0.0 to 1.0>,
        "anger": <float 0.0 to 1.0>,
        "fear": <float 0.0 to 1.0>,
        "surprise": <float 0.0 to 1.0>,
        "disgust": <float 0.0 to 1.0>,
        "contempt": <float 0.0 to 1.0>,
        "trust": <float 0.0 to 1.0>,
        "anticipation": <float 0.0 to 1.0>,
        "assertiveness": <float 0.0 to 1.0>,
        "openness": <float 0.0 to 1.0>,
        "hostility": <float 0.0 to 1.0>,
        "vulnerability": <float 0.0 to 1.0>,
        "label": "<10 words or fewer describing the emotional state>",
        "confidence": <float 0.0 to 1.0>
      },
      "sentiment": "<positive|negative|neutral>",
      "communicative_act": "<assertion|request|question|apology|complaint|criticism|praise|threat|promise|concession|deflection|disclosure|validation|accusation|justification|command|empathy|sarcasm|other>",
      "interaction_health": <float 0.0 to 10.0>,
      "notes": "<2-4 sentences of analyst commentary explaining the annotation>"
    }
  ]
}"""

    return f"""\
Domain: {domain.value}
Participants: {participant_list}{context_block}

{domain_guidance}

## Task

Annotate every message in the transcript below with:

1. **Emotion vector** — multi-dimensional scores as defined in the schema.
   - `valence`: overall hedonic tone (-1 = extremely negative, +1 = extremely positive)
   - `arousal`: activation/energy level (0 = very calm, 1 = extremely agitated or excited)
   - `dominance`: social power stance (-1 = powerless/submissive, +1 = dominant/in-control)
   - Basic emotions (0–1 each): score the degree to which each is present.
   - Communicative dimensions (0–1 each): how the speaker expresses themselves.
   - `label`: a concise phrase capturing the overall emotional state (≤ 10 words).
   - `confidence`: how certain you are in this annotation (0–1).

2. **Sentiment** — overall relational impact: positive, negative, or neutral.

3. **Communicative act** — the primary speech act performed by this utterance.

4. **Interaction health** — a running score (0–10) reflecting the cumulative \
health and constructiveness of the conversation up to and including this turn. \
This should evolve as the conversation progresses; it is NOT a per-message score \
but a cumulative conversational state indicator.
   - 0–2: severely dysfunctional (abuse, complete breakdown)
   - 3–4: highly problematic (escalating conflict, stonewalling)
   - 5–6: mixed or neutral (some friction but also some connection)
   - 7–8: largely constructive (disagreement handled respectfully)
   - 9–10: highly productive (deep understanding, mutual respect)

5. **Notes** — 2–4 sentences of analytical commentary explaining the annotation \
in plain language, referencing context from earlier in the conversation where relevant.

## Transcript

{transcript_block}

## Output

Return a JSON object matching this schema exactly:
{schema}
"""


# ---------------------------------------------------------------------------
# Summary / narrative prompt
# ---------------------------------------------------------------------------

def build_summary_prompt(
    transcript_block: str,
    annotations_json: str,
    domain: ConversationDomain,
    participants: List[str],
    context_note: str | None,
) -> str:
    """Build the prompt for generating the high-level conversation summary."""
    domain_guidance = _domain_guidance(domain)
    participant_list = ", ".join(participants)
    context_block = f"\nContext: {context_note}" if context_note else ""

    schema = """
{
  "overview": "<paragraph summarising what this conversation is about>",
  "key_themes": ["<theme 1>", "<theme 2>", ...],
  "communication_patterns": [
    "<observed pattern 1, e.g. 'Alice consistently deflects when challenged'>",
    ...
  ],
  "power_dynamics": "<paragraph describing dominance/submission patterns>",
  "recommendations": [
    "<actionable recommendation 1>",
    ...
  ],
  "emotional_arc": "<paragraph describing how the emotional tone evolved across the conversation>"
}"""

    return f"""\
Domain: {domain.value}
Participants: {participant_list}{context_block}

{domain_guidance}

## Annotated Transcript

{transcript_block}

## Per-message Emotion Annotations (JSON)

{annotations_json}

## Task

Using both the transcript and the emotion annotations, produce a high-level \
narrative analysis of this conversation. Address:

1. **Overview**: What is this conversation about? What are the stakes for each participant?
2. **Key themes**: What recurring topics, concerns, or values are at play?
3. **Communication patterns**: Identify 3–6 specific, evidence-based behavioural patterns \
   (e.g. stonewalling, deflection, empathy bids, power plays, vulnerability, escalation).
4. **Power dynamics**: Who holds more discourse power? How does dominance shift? \
   Are there asymmetries in emotional labour or vulnerability?
5. **Recommendations**: 3–5 concrete, actionable suggestions tailored to the domain \
   and participants for improving future interactions.
6. **Emotional arc**: Describe the journey of the conversation's emotional tone from \
   start to finish — where did it peak negatively, where did it recover, and where did it end?

Output must be valid JSON matching this schema:
{schema}
"""


# ---------------------------------------------------------------------------
# Optional: natural-language descriptions of dynamics findings
# ---------------------------------------------------------------------------

def build_dynamics_narrative_prompt(
    transcript_block: str,
    turning_points_json: str,
    reactivity_json: str,
    domain: ConversationDomain,
    participants: List[str],
) -> str:
    """
    Prompt for generating a natural-language description of the computed
    emotional dynamics (turning points and reactivity).  Optional — callers
    may skip this for a purely computational output.
    """
    participant_list = ", ".join(participants)

    return f"""\
Domain: {domain.value}
Participants: {participant_list}

## Transcript
{transcript_block}

## Computed Turning Points
{turning_points_json}

## Computed Reactivity Events
{reactivity_json}

## Task

Write a concise analytical narrative (3–5 paragraphs) that interprets these \
computed findings for a non-technical audience. Focus on:
- The most significant emotional turning points and what triggered them
- Which speaker is more emotionally reactive to the other, and in what direction
- Whether there is evidence of emotional contagion (one speaker's state spreading)
- The overall pattern of emotional regulation or dysregulation
- Any moments of particular concern or particular promise

Return a JSON object:
{{"dynamics_narrative": "<narrative text here>"}}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_transcript(messages: List[Message]) -> str:
    lines = []
    for m in messages:
        ts = f" [{m.timestamp.strftime('%H:%M')}]" if m.timestamp else ""
        lines.append(f"[{m.turn_index}]{ts} {m.speaker}: {m.text}")
    return "\n".join(lines)


def _domain_guidance(domain: ConversationDomain) -> str:
    """Return domain-specific framing notes for the model."""
    guidance = {
        ConversationDomain.THERAPY: (
            "This is a psychotherapeutic context. Weight emotional disclosure, "
            "therapeutic alliance, resistance, and transference patterns highly. "
            "Interaction health reflects safety, trust, and therapeutic progress."
        ),
        ConversationDomain.ROMANTIC: (
            "This is an intimate partnership context. Attachment styles, bids for "
            "connection, contempt (highly destructive), and repair attempts are key "
            "signals. Use Gottman's four horsemen framework as a reference."
        ),
        ConversationDomain.FAMILY: (
            "This is a family context. Consider generational dynamics, parentification, "
            "enmeshment, triangulation, and boundary violations alongside emotional tone."
        ),
        ConversationDomain.BUSINESS: (
            "This is a professional/organisational context. Power differentials related "
            "to hierarchy, passive aggression, coalition-building, and face-saving are "
            "particularly salient. Interaction health reflects psychological safety and "
            "productive collaboration."
        ),
        ConversationDomain.POLITICAL: (
            "This is a political or public-affairs context. Rhetorical strategies, "
            "framing, in-group/out-group signalling, emotional manipulation, and "
            "deliberative quality are key. Be attentive to coded language."
        ),
        ConversationDomain.MEDICAL: (
            "This is a medical or health context. Patient autonomy, anxiety, information "
            "asymmetry, and clinician empathy are central. Interaction health reflects "
            "shared decision-making and patient comfort."
        ),
        ConversationDomain.LEGAL: (
            "This is a legal context (negotiation, deposition, courtroom, etc.). "
            "Strategic positioning, implied threats, face management, and emotional "
            "suppression or performance are all relevant."
        ),
        ConversationDomain.EDUCATION: (
            "This is an educational context. Power asymmetry between instructor and "
            "learner, motivation, shame, curiosity, and authority are key dimensions."
        ),
        ConversationDomain.FRIENDSHIP: (
            "This is a peer friendship context. Reciprocity, mutual support, social "
            "comparison, and in-group belonging are particularly salient."
        ),
        ConversationDomain.GENERAL: (
            "Analyse this conversation with no specific domain assumptions. "
            "Apply general principles of interpersonal communication."
        ),
    }
    return guidance.get(domain, guidance[ConversationDomain.GENERAL])
