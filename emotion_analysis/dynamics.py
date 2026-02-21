"""
Emotional dynamics analysis.

This module operates purely on annotated messages — no LLM calls.
All algorithms are interpretable, deterministic, and work on small to
medium-sized datasets (typical conversation lengths).

Key analyses
------------
- Emotional trajectory: per-speaker time series of valence/arousal/dominance
- Turning point detection: significant shifts in emotional state
- Reactivity analysis: how A's emotional state predicts shifts in B's state
- Emotional contagion: degree of cross-speaker valence synchrony
- Escalation / de-escalation sequence detection
- Aggregate per-speaker statistics

Mathematical notes
------------------
Reactivity is computed as the *lagged influence* of speaker A on speaker B:
  For every consecutive A→B turn pair, we record:
    trigger_valence  = A's valence on that turn
    response_delta   = B's valence on that turn − B's valence on the previous B turn

  The reactivity coefficient A→B is the Pearson correlation of trigger_valence
  with response_delta across all A→B pairs.  Positive = A's negativity induces
  B's negativity; negative = A's negativity induces B's positivity (resilience).

Turning points are detected with a sliding-window z-score algorithm applied to
the conversation's valence sequence.  A point t is a turning point if the
absolute change in smoothed valence exceeds `sigma_threshold` standard
deviations of changes in the local window.

Emotional contagion is measured as the average absolute cross-speaker
correlation of valence sequences, sampled at shared time steps.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .models import (
    AnnotatedMessage,
    EmotionalDynamics,
    EmotionVector,
    ReactivityEvent,
    Sentiment,
    TurningPoint,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_dynamics(
    annotated: List[AnnotatedMessage],
    participants: List[str],
    *,
    turning_point_window: int = 3,
    turning_point_threshold: float = 1.25,
    min_escalation_length: int = 3,
) -> EmotionalDynamics:
    """
    Compute all emotional dynamics metrics from a list of annotated messages.

    Parameters
    ----------
    annotated:
        Ordered list of AnnotatedMessage objects (output of EmotionAnalyzer).
    participants:
        Ordered list of speaker names.
    turning_point_window:
        Number of turns on each side of a candidate turning point used to
        compute the local standard deviation.
    turning_point_threshold:
        Minimum z-score (in local-window SD units) required to classify a
        valence shift as a turning point.
    min_escalation_length:
        Minimum number of consecutive turns required to form an
        escalation/de-escalation sequence.

    Returns
    -------
    EmotionalDynamics
    """
    if not annotated:
        return EmotionalDynamics()

    # ---- Trajectories -------------------------------------------------------
    val_traj, aro_traj, dom_traj = _build_per_speaker_trajectories(annotated)
    health_traj = [am.interaction_health for am in annotated]

    # ---- Turning points -----------------------------------------------------
    all_valences = [am.emotion.valence for am in annotated]
    turning_points = _detect_turning_points(
        annotated, all_valences, turning_point_window, turning_point_threshold
    )

    # ---- Reactivity ---------------------------------------------------------
    reactivity_events, reactivity_matrix = _compute_reactivity(annotated, participants)

    # ---- Escalation sequences -----------------------------------------------
    esc_seqs, de_esc_seqs = _find_monotone_runs(
        annotated, min_length=min_escalation_length
    )

    # ---- Aggregate stats ----------------------------------------------------
    mean_val, val_var = _per_speaker_stats(val_traj)
    dominant_sentiment = _dominant_sentiment(annotated, participants)

    # ---- Contagion ----------------------------------------------------------
    contagion = _compute_contagion(val_traj, participants)

    return EmotionalDynamics(
        valence_trajectories=val_traj,
        arousal_trajectories=aro_traj,
        dominance_trajectories=dom_traj,
        interaction_health_trajectory=health_traj,
        turning_points=turning_points,
        reactivity_events=reactivity_events,
        reactivity_matrix=reactivity_matrix,
        escalation_sequences=esc_seqs,
        de_escalation_sequences=de_esc_seqs,
        mean_valence=mean_val,
        valence_variance=val_var,
        emotional_contagion_score=contagion,
        dominant_sentiment=dominant_sentiment,
    )


# ---------------------------------------------------------------------------
# Per-speaker trajectory builders
# ---------------------------------------------------------------------------

def _build_per_speaker_trajectories(
    annotated: List[AnnotatedMessage],
) -> Tuple[
    Dict[str, List[float]],
    Dict[str, List[float]],
    Dict[str, List[float]],
]:
    val_traj: Dict[str, List[float]] = defaultdict(list)
    aro_traj: Dict[str, List[float]] = defaultdict(list)
    dom_traj: Dict[str, List[float]] = defaultdict(list)

    for am in annotated:
        sp = am.message.speaker
        val_traj[sp].append(am.emotion.valence)
        aro_traj[sp].append(am.emotion.arousal)
        dom_traj[sp].append(am.emotion.dominance)

    return dict(val_traj), dict(aro_traj), dict(dom_traj)


# ---------------------------------------------------------------------------
# Turning point detection
# ---------------------------------------------------------------------------

def _detect_turning_points(
    annotated: List[AnnotatedMessage],
    valences: List[float],
    window: int,
    threshold: float,
) -> List[TurningPoint]:
    """
    Detect positions where the valence trajectory shifts significantly.

    Uses a local z-score: for each position t, compute the change
    Δv[t] = v[t] − v[t−1] and compare it to the SD of changes in the
    surrounding window.  Points exceeding the threshold are turning points.
    """
    if len(valences) < 3:
        return []

    deltas = [valences[i] - valences[i - 1] for i in range(1, len(valences))]
    turning_points: List[TurningPoint] = []

    for i, delta in enumerate(deltas):
        lo = max(0, i - window)
        hi = min(len(deltas), i + window + 1)
        window_deltas = deltas[lo:hi]
        if len(window_deltas) < 2:
            continue
        try:
            sd = statistics.stdev(window_deltas)
        except statistics.StatisticsError:
            continue
        if sd == 0:
            continue

        z = abs(delta) / sd
        if z >= threshold:
            turn_index = i + 1  # delta[i] = v[i+1] - v[i], so the *new* position is i+1
            am = annotated[turn_index]
            direction = Sentiment.POSITIVE if delta > 0 else Sentiment.NEGATIVE

            # Identify likely trigger: the previous speaker's turn
            trigger_speaker: Optional[str] = None
            if turn_index > 0:
                trigger_speaker = annotated[turn_index - 1].message.speaker

            turning_points.append(
                TurningPoint(
                    turn_index=am.message.turn_index,
                    magnitude=abs(delta),
                    direction=direction,
                    trigger_speaker=trigger_speaker,
                    description=(
                        f"{am.message.speaker}'s valence shifted "
                        f"{'up' if delta > 0 else 'down'} by {abs(delta):.2f} "
                        f"(z={z:.2f}) — {am.emotion.label}"
                    ),
                )
            )

    return turning_points


# ---------------------------------------------------------------------------
# Reactivity analysis
# ---------------------------------------------------------------------------

def _compute_reactivity(
    annotated: List[AnnotatedMessage],
    participants: List[str],
) -> Tuple[List[ReactivityEvent], Dict[str, Dict[str, float]]]:
    """
    For each consecutive (A→B) turn pair, measure how A's valence predicts
    the *change* in B's valence relative to B's previous turn.

    Returns
    -------
    events:
        Individual ReactivityEvent objects for every A→B pair.
    matrix:
        speaker_a → speaker_b → Pearson r (or mean |delta| if < 2 data points)
    """
    # Build per-speaker index of their turns (in order)
    speaker_turns: Dict[str, List[int]] = defaultdict(list)
    for idx, am in enumerate(annotated):
        speaker_turns[am.message.speaker].append(idx)

    events: List[ReactivityEvent] = []

    # Iterate through the annotated sequence looking for cross-speaker transitions
    for seq_idx in range(1, len(annotated)):
        cur = annotated[seq_idx]
        prev = annotated[seq_idx - 1]

        if cur.message.speaker == prev.message.speaker:
            continue  # same speaker — skip

        # Find B's baseline: the last turn B took before this one
        sp_b = cur.message.speaker
        b_turns_before = [
            i for i in speaker_turns[sp_b] if i < seq_idx
        ]
        if not b_turns_before:
            baseline_emotion: Optional[EmotionVector] = None
            valence_delta = 0.0
            arousal_delta = 0.0
        else:
            baseline_am = annotated[b_turns_before[-1]]
            baseline_emotion = baseline_am.emotion
            valence_delta = cur.emotion.valence - baseline_am.emotion.valence
            arousal_delta = cur.emotion.arousal - baseline_am.emotion.arousal

        # Reactivity score: how large is the shift in PAD space?
        reactivity_score = min(1.0, abs(valence_delta) / 2.0 + abs(arousal_delta) / 2.0)

        events.append(
            ReactivityEvent(
                trigger_turn_index=prev.message.turn_index,
                trigger_speaker=prev.message.speaker,
                trigger_emotion=prev.emotion,
                response_turn_index=cur.message.turn_index,
                response_speaker=sp_b,
                response_emotion=cur.emotion,
                baseline_emotion=baseline_emotion,
                valence_delta=valence_delta,
                arousal_delta=arousal_delta,
                reactivity_score=reactivity_score,
                description=(
                    f"{prev.message.speaker} [{prev.emotion.label}] → "
                    f"{sp_b} valence shifted "
                    f"{'+' if valence_delta >= 0 else ''}{valence_delta:.2f} "
                    f"({cur.emotion.label})"
                ),
            )
        )

    # Build aggregated reactivity matrix from events
    # key: (trigger_speaker, response_speaker) → [trigger_valences], [deltas]
    pairs: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = defaultdict(
        lambda: ([], [])
    )
    for ev in events:
        key = (ev.trigger_speaker, ev.response_speaker)
        pairs[key][0].append(ev.trigger_emotion.valence)
        pairs[key][1].append(ev.valence_delta)

    matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (sp_a, sp_b), (triggers, deltas) in pairs.items():
        if len(triggers) >= 2:
            r = _pearson_r(triggers, deltas)
            matrix[sp_a][sp_b] = round(r, 4)
        else:
            # Not enough data for correlation — use mean absolute delta
            matrix[sp_a][sp_b] = round(
                sum(abs(d) for d in deltas) / len(deltas) if deltas else 0.0, 4
            )

    return events, {k: dict(v) for k, v in matrix.items()}


# ---------------------------------------------------------------------------
# Escalation / de-escalation sequence detection
# ---------------------------------------------------------------------------

def _find_monotone_runs(
    annotated: List[AnnotatedMessage],
    min_length: int = 3,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Find runs of consecutive turns where valence falls (escalation) or
    rises (de-escalation) monotonically for at least `min_length` turns.

    Returns (escalation_sequences, de_escalation_sequences) where each
    sequence is a list of turn indices.
    """
    if len(annotated) < min_length:
        return [], []

    valences = [(am.message.turn_index, am.emotion.valence) for am in annotated]
    esc_seqs: List[List[int]] = []
    de_esc_seqs: List[List[int]] = []

    def flush(run: List[int], direction: int) -> None:
        if len(run) >= min_length:
            if direction == -1:
                esc_seqs.append(run[:])
            else:
                de_esc_seqs.append(run[:])

    current_run: List[int] = [valences[0][0]]
    current_dir: Optional[int] = None  # -1 = falling, +1 = rising

    for i in range(1, len(valences)):
        idx, v = valences[i]
        prev_v = valences[i - 1][1]
        delta = v - prev_v

        if abs(delta) < 0.05:
            # Treat very small changes as continuation of current trend
            current_run.append(idx)
            continue

        new_dir = -1 if delta < 0 else 1
        if new_dir == current_dir:
            current_run.append(idx)
        else:
            flush(current_run, current_dir or new_dir)
            current_run = [valences[i - 1][0], idx]
            current_dir = new_dir

    flush(current_run, current_dir or 1)
    return esc_seqs, de_esc_seqs


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def _per_speaker_stats(
    val_traj: Dict[str, List[float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    mean_val: Dict[str, float] = {}
    val_var: Dict[str, float] = {}
    for sp, vals in val_traj.items():
        mean_val[sp] = round(statistics.mean(vals), 4) if vals else 0.0
        val_var[sp] = round(statistics.variance(vals), 4) if len(vals) > 1 else 0.0
    return mean_val, val_var


def _dominant_sentiment(
    annotated: List[AnnotatedMessage],
    participants: List[str],
) -> Dict[str, Sentiment]:
    counts: Dict[str, Dict[Sentiment, int]] = {
        sp: {s: 0 for s in Sentiment} for sp in participants
    }
    for am in annotated:
        sp = am.message.speaker
        if sp in counts:
            counts[sp][am.sentiment] += 1
    result: Dict[str, Sentiment] = {}
    for sp, cnts in counts.items():
        result[sp] = max(cnts, key=cnts.__getitem__)
    return result


# ---------------------------------------------------------------------------
# Emotional contagion
# ---------------------------------------------------------------------------

def _compute_contagion(
    val_traj: Dict[str, List[float]],
    participants: List[str],
) -> float:
    """
    Overall emotional contagion score (0–1).

    Computed as the mean |Pearson r| between all pairs of speakers' valence
    trajectories, sampled at even indices to align them (crude but effective
    for conversational data where turns alternate).

    Returns 0.0 if fewer than 2 speakers or insufficient data.
    """
    speakers = [p for p in participants if p in val_traj and len(val_traj[p]) >= 2]
    if len(speakers) < 2:
        return 0.0

    correlations: List[float] = []
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            a_vals = val_traj[speakers[i]]
            b_vals = val_traj[speakers[j]]
            # Align by taking the minimum length
            n = min(len(a_vals), len(b_vals))
            if n < 2:
                continue
            r = _pearson_r(a_vals[:n], b_vals[:n])
            correlations.append(abs(r))

    return round(statistics.mean(correlations), 4) if correlations else 0.0


# ---------------------------------------------------------------------------
# Additional high-level analyses (importable individually)
# ---------------------------------------------------------------------------

def compute_valence_momentum(
    annotated: List[AnnotatedMessage],
    window: int = 5,
) -> List[float]:
    """
    Compute a rolling valence momentum for the whole conversation.

    Returns a list of values (same length as `annotated`) where each value
    is the slope of a linear regression of valence over the preceding
    `window` turns.  Positive = improving, negative = deteriorating.

    Useful for real-time alerting: "the conversation has been declining for
    the last N turns."
    """
    valences = [am.emotion.valence for am in annotated]
    momentum = []
    for i in range(len(valences)):
        lo = max(0, i - window + 1)
        segment = valences[lo: i + 1]
        momentum.append(_linear_slope(segment))
    return momentum


def speaker_influence_score(
    annotated: List[AnnotatedMessage],
    speaker_a: str,
    speaker_b: str,
) -> Dict[str, float]:
    """
    Compute directional influence metrics from speaker_a's turns onto speaker_b.

    Returns a dict with:
      - ``reactivity_r``: Pearson r between A's valence and B's subsequent delta
      - ``mean_trigger_valence``: average valence of A's triggering turns
      - ``mean_response_delta``: average change in B's valence after A speaks
      - ``n_pairs``: number of A→B pairs observed
      - ``positive_responses``: fraction of B's responses that moved positively
      - ``negative_responses``: fraction that moved negatively
    """
    speaker_turns: Dict[str, List[int]] = defaultdict(list)
    for idx, am in enumerate(annotated):
        speaker_turns[am.message.speaker].append(idx)

    trigger_vals: List[float] = []
    response_deltas: List[float] = []

    for seq_idx in range(1, len(annotated)):
        cur = annotated[seq_idx]
        prev = annotated[seq_idx - 1]
        if prev.message.speaker != speaker_a or cur.message.speaker != speaker_b:
            continue

        b_turns_before = [i for i in speaker_turns[speaker_b] if i < seq_idx]
        if not b_turns_before:
            continue
        baseline = annotated[b_turns_before[-1]].emotion.valence
        delta = cur.emotion.valence - baseline

        trigger_vals.append(prev.emotion.valence)
        response_deltas.append(delta)

    n = len(trigger_vals)
    if n == 0:
        return {
            "reactivity_r": 0.0,
            "mean_trigger_valence": 0.0,
            "mean_response_delta": 0.0,
            "n_pairs": 0,
            "positive_responses": 0.0,
            "negative_responses": 0.0,
        }

    r = _pearson_r(trigger_vals, response_deltas) if n >= 2 else 0.0
    pos_frac = sum(1 for d in response_deltas if d > 0.05) / n
    neg_frac = sum(1 for d in response_deltas if d < -0.05) / n

    return {
        "reactivity_r": round(r, 4),
        "mean_trigger_valence": round(statistics.mean(trigger_vals), 4),
        "mean_response_delta": round(statistics.mean(response_deltas), 4),
        "n_pairs": n,
        "positive_responses": round(pos_frac, 3),
        "negative_responses": round(neg_frac, 3),
    }


def emotional_regulation_index(
    annotated: List[AnnotatedMessage],
    speaker: str,
) -> float:
    """
    Measure a speaker's ability to maintain or recover emotional stability.

    Returns a value from -1 (consistently dysregulated) to +1 (consistently
    self-regulating), computed as the mean of self-directed valence changes:
    positive change after a negative turn = regulation; continued negativity = dysregulation.
    """
    speaker_turns = [am for am in annotated if am.message.speaker == speaker]
    if len(speaker_turns) < 2:
        return 0.0

    recoveries: List[float] = []
    for i in range(1, len(speaker_turns)):
        prev_val = speaker_turns[i - 1].emotion.valence
        curr_val = speaker_turns[i].emotion.valence
        # If previously negative, does the speaker recover?
        if prev_val < -0.1:
            recoveries.append(curr_val - prev_val)

    if not recoveries:
        return 0.0
    # Normalise to [-1, 1]
    mean_r = statistics.mean(recoveries)
    return round(max(-1.0, min(1.0, mean_r)), 4)


def compute_repair_attempts(
    annotated: List[AnnotatedMessage],
) -> List[Dict]:
    """
    Identify turns that represent emotional repair attempts — moments where
    a speaker follows a negative turn (their own or their partner's) with a
    positive or neutral one, or where the communicative act is apology/validation.

    Returns a list of dicts describing each detected repair attempt.
    """
    from .models import CommunicativeAct

    repair_acts = {CommunicativeAct.APOLOGY, CommunicativeAct.VALIDATION,
                   CommunicativeAct.EMPATHY, CommunicativeAct.CONCESSION}
    repairs = []

    for i, am in enumerate(annotated):
        is_repair = False
        reason = ""

        # Explicit repair communicative acts
        if am.communicative_act in repair_acts:
            is_repair = True
            reason = f"communicative act: {am.communicative_act.value}"

        # Positive turn after a negative one (same or different speaker)
        elif i > 0:
            prev = annotated[i - 1]
            if prev.emotion.valence < -0.2 and am.emotion.valence > prev.emotion.valence + 0.25:
                is_repair = True
                reason = (
                    f"valence recovery: {prev.emotion.valence:.2f} → "
                    f"{am.emotion.valence:.2f}"
                )

        if is_repair:
            repairs.append({
                "turn_index": am.message.turn_index,
                "speaker": am.message.speaker,
                "text": am.message.text,
                "emotion_label": am.emotion.label,
                "reason": reason,
            })

    return repairs


def conversation_health_summary(
    dynamics: "EmotionalDynamics",
) -> Dict[str, object]:
    """
    Produce a compact health dashboard for a conversation.

    Returns a dict suitable for display or logging.
    """
    health_scores = dynamics.interaction_health_trajectory
    if not health_scores:
        return {}

    return {
        "final_health": round(health_scores[-1], 2),
        "mean_health": round(statistics.mean(health_scores), 2),
        "min_health": round(min(health_scores), 2),
        "max_health": round(max(health_scores), 2),
        "health_variance": round(
            statistics.variance(health_scores) if len(health_scores) > 1 else 0.0, 3
        ),
        "n_turning_points": len(dynamics.turning_points),
        "n_escalation_sequences": len(dynamics.escalation_sequences),
        "n_repair_attempts": 0,  # caller can add from compute_repair_attempts
        "emotional_contagion": dynamics.emotional_contagion_score,
        "mean_valence_per_speaker": dynamics.mean_valence,
        "dominant_sentiment_per_speaker": {
            k: v.value for k, v in dynamics.dominant_sentiment.items()
        },
    }


# ---------------------------------------------------------------------------
# Mathematical utilities (no external deps required)
# ---------------------------------------------------------------------------

def _pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 on degenerate input."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x = list(x[:n])
    y = list(y[:n])
    mx = statistics.mean(x)
    my = statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return num / (sx * sy)


def _linear_slope(values: List[float]) -> float:
    """Slope of a least-squares line through the values. Returns 0.0 if < 2 points."""
    n = len(values)
    if n < 2:
        return 0.0
    x = list(range(n))
    mx = statistics.mean(x)
    my = statistics.mean(values)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, values))
    den = sum((xi - mx) ** 2 for xi in x)
    return num / den if den != 0 else 0.0
