"""Tests for the dynamics engine — no API calls required."""

from datetime import datetime

import pytest

from emotion_analysis.dynamics import (
    _pearson_r,
    _linear_slope,
    compute_dynamics,
    compute_repair_attempts,
    compute_valence_momentum,
    conversation_health_summary,
    emotional_regulation_index,
    speaker_influence_score,
)
from emotion_analysis.models import (
    AnnotatedMessage,
    CommunicativeAct,
    EmotionVector,
    Message,
    Sentiment,
)


# ---------------------------------------------------------------------------
# Test fixture builders
# ---------------------------------------------------------------------------

def _ev(valence: float, arousal: float = 0.5, dominance: float = 0.0,
        label: str = "test", **kwargs) -> EmotionVector:
    return EmotionVector(
        valence=valence, arousal=arousal, dominance=dominance,
        label=label, **kwargs
    )


def _msg(turn_index: int, speaker: str, text: str = "x") -> Message:
    return Message(turn_index=turn_index, speaker=speaker, text=text)


def _am(
    turn_index: int,
    speaker: str,
    valence: float,
    arousal: float = 0.5,
    health: float = 5.0,
    sentiment: Sentiment = Sentiment.NEUTRAL,
    act: CommunicativeAct = CommunicativeAct.ASSERTION,
) -> AnnotatedMessage:
    return AnnotatedMessage(
        message=_msg(turn_index, speaker),
        emotion=_ev(valence, arousal),
        sentiment=sentiment,
        communicative_act=act,
        interaction_health=health,
        notes="",
    )


# ---------------------------------------------------------------------------
# _pearson_r
# ---------------------------------------------------------------------------

def test_pearson_r_perfect_positive():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]
    r = _pearson_r(x, y)
    assert abs(r - 1.0) < 1e-6


def test_pearson_r_perfect_negative():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [8.0, 6.0, 4.0, 2.0]
    r = _pearson_r(x, y)
    assert abs(r + 1.0) < 1e-6


def test_pearson_r_zero_variance():
    # Constant series → correlation undefined → 0.0
    r = _pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert r == 0.0


def test_pearson_r_too_few_points():
    assert _pearson_r([1.0], [1.0]) == 0.0
    assert _pearson_r([], []) == 0.0


# ---------------------------------------------------------------------------
# _linear_slope
# ---------------------------------------------------------------------------

def test_linear_slope_rising():
    assert _linear_slope([0.0, 1.0, 2.0, 3.0]) > 0


def test_linear_slope_falling():
    assert _linear_slope([3.0, 2.0, 1.0, 0.0]) < 0


def test_linear_slope_flat():
    s = _linear_slope([1.0, 1.0, 1.0, 1.0])
    assert s == 0.0


def test_linear_slope_single_point():
    assert _linear_slope([5.0]) == 0.0


# ---------------------------------------------------------------------------
# compute_dynamics
# ---------------------------------------------------------------------------

def _make_alternating_annotated(n_pairs: int = 4) -> list:
    """Create an alternating A/B conversation with known valence pattern."""
    annotated = []
    for i in range(n_pairs * 2):
        speaker = "Alice" if i % 2 == 0 else "Bob"
        # Alice goes negative, Bob mirrors
        valence = -0.5 if speaker == "Alice" else -0.3
        annotated.append(_am(i, speaker, valence))
    return annotated


def test_compute_dynamics_returns_object():
    annotated = _make_alternating_annotated()
    dynamics = compute_dynamics(annotated, ["Alice", "Bob"])
    assert dynamics is not None


def test_compute_dynamics_trajectories_keys():
    annotated = _make_alternating_annotated(3)
    d = compute_dynamics(annotated, ["Alice", "Bob"])
    assert "Alice" in d.valence_trajectories
    assert "Bob" in d.valence_trajectories


def test_compute_dynamics_trajectory_lengths():
    annotated = _make_alternating_annotated(4)  # 8 turns, 4 per speaker
    d = compute_dynamics(annotated, ["Alice", "Bob"])
    assert len(d.valence_trajectories["Alice"]) == 4
    assert len(d.valence_trajectories["Bob"]) == 4


def test_compute_dynamics_health_trajectory():
    annotated = [_am(i, "A" if i % 2 == 0 else "B", 0.0, health=float(i)) for i in range(6)]
    d = compute_dynamics(annotated, ["A", "B"])
    assert d.interaction_health_trajectory == [float(i) for i in range(6)]


def test_compute_dynamics_mean_valence():
    # All Alice turns at 0.5, all Bob turns at -0.5
    annotated = []
    for i in range(6):
        sp = "Alice" if i % 2 == 0 else "Bob"
        val = 0.5 if sp == "Alice" else -0.5
        annotated.append(_am(i, sp, val))
    d = compute_dynamics(annotated, ["Alice", "Bob"])
    assert abs(d.mean_valence["Alice"] - 0.5) < 1e-4
    assert abs(d.mean_valence["Bob"] - (-0.5)) < 1e-4


def test_compute_dynamics_empty():
    d = compute_dynamics([], [])
    assert d.valence_trajectories == {}


def test_compute_dynamics_reactivity_matrix_keys():
    annotated = _make_alternating_annotated(3)
    d = compute_dynamics(annotated, ["Alice", "Bob"])
    # Both directions should appear in the matrix
    assert "Alice" in d.reactivity_matrix or "Bob" in d.reactivity_matrix


# ---------------------------------------------------------------------------
# Turning points
# ---------------------------------------------------------------------------

def test_turning_points_detected_on_large_shift():
    # Build a sequence with a dramatic drop in the middle
    annotated = [
        _am(0, "A", 0.8), _am(1, "B", 0.7),
        _am(2, "A", 0.6), _am(3, "B", -0.9),   # big drop here
        _am(4, "A", -0.8), _am(5, "B", -0.7),
    ]
    d = compute_dynamics(annotated, ["A", "B"])
    # Should detect at least the dramatic drop
    assert len(d.turning_points) >= 1


def test_turning_points_empty_on_flat():
    annotated = [_am(i, "A" if i % 2 == 0 else "B", 0.0) for i in range(6)]
    d = compute_dynamics(annotated, ["A", "B"])
    # Flat valence → no significant shifts
    assert len(d.turning_points) == 0


# ---------------------------------------------------------------------------
# Escalation sequences
# ---------------------------------------------------------------------------

def test_escalation_detected():
    # Monotone decline across 4 turns
    annotated = [
        _am(0, "A", 0.6), _am(1, "B", 0.3),
        _am(2, "A", 0.0), _am(3, "B", -0.4),
        _am(4, "A", -0.7),
    ]
    d = compute_dynamics(annotated, ["A", "B"], min_escalation_length=3)
    assert len(d.escalation_sequences) >= 1


def test_de_escalation_detected():
    # Monotone rise across 4 turns
    annotated = [
        _am(0, "A", -0.7), _am(1, "B", -0.4),
        _am(2, "A", 0.0), _am(3, "B", 0.3),
        _am(4, "A", 0.6),
    ]
    d = compute_dynamics(annotated, ["A", "B"], min_escalation_length=3)
    assert len(d.de_escalation_sequences) >= 1


# ---------------------------------------------------------------------------
# compute_valence_momentum
# ---------------------------------------------------------------------------

def test_valence_momentum_length():
    annotated = [_am(i, "A", float(i) * 0.1) for i in range(8)]
    momentum = compute_valence_momentum(annotated, window=3)
    assert len(momentum) == 8


def test_valence_momentum_positive_for_rising():
    # All rising
    annotated = [_am(i, "A", float(i) * 0.1) for i in range(6)]
    momentum = compute_valence_momentum(annotated, window=3)
    # Last entry should be positive
    assert momentum[-1] > 0


# ---------------------------------------------------------------------------
# speaker_influence_score
# ---------------------------------------------------------------------------

def test_speaker_influence_score_returns_dict():
    annotated = _make_alternating_annotated(4)
    result = speaker_influence_score(annotated, "Alice", "Bob")
    assert "reactivity_r" in result
    assert "n_pairs" in result
    assert "mean_response_delta" in result


def test_speaker_influence_score_no_pairs():
    # Single speaker — no cross-speaker pairs
    annotated = [_am(i, "Alice", 0.0) for i in range(4)]
    result = speaker_influence_score(annotated, "Alice", "Bob")
    assert result["n_pairs"] == 0
    assert result["reactivity_r"] == 0.0


# ---------------------------------------------------------------------------
# emotional_regulation_index
# ---------------------------------------------------------------------------

def test_emotional_regulation_positive_for_recovery():
    # Alice has a negative turn then improves each time
    annotated = [
        _am(0, "Alice", -0.8), _am(1, "Bob", 0.0),
        _am(2, "Alice", -0.2), _am(3, "Bob", 0.0),  # recovered
        _am(4, "Alice", -0.6), _am(5, "Bob", 0.0),
        _am(6, "Alice", 0.1),                         # recovered again
    ]
    eri = emotional_regulation_index(annotated, "Alice")
    assert eri > 0  # positive = recovering


def test_emotional_regulation_single_turn():
    annotated = [_am(0, "Alice", -0.5)]
    assert emotional_regulation_index(annotated, "Alice") == 0.0


# ---------------------------------------------------------------------------
# compute_repair_attempts
# ---------------------------------------------------------------------------

def test_repair_attempt_from_act():
    annotated = [
        _am(0, "A", -0.8, act=CommunicativeAct.CRITICISM),
        _am(1, "B", 0.2, act=CommunicativeAct.APOLOGY),   # explicit repair
    ]
    repairs = compute_repair_attempts(annotated)
    assert any(r["turn_index"] == 1 for r in repairs)


def test_repair_attempt_from_valence_recovery():
    annotated = [
        _am(0, "A", -0.7),
        _am(1, "B", 0.4),   # large positive shift → implicit repair
    ]
    repairs = compute_repair_attempts(annotated)
    assert len(repairs) >= 1


def test_no_repair_when_stable_negative():
    annotated = [
        _am(0, "A", -0.5, act=CommunicativeAct.CRITICISM),
        _am(1, "B", -0.5, act=CommunicativeAct.ASSERTION),
    ]
    repairs = compute_repair_attempts(annotated)
    assert len(repairs) == 0


# ---------------------------------------------------------------------------
# conversation_health_summary
# ---------------------------------------------------------------------------

def test_health_summary_keys():
    annotated = _make_alternating_annotated(3)
    d = compute_dynamics(annotated, ["Alice", "Bob"])
    summary = conversation_health_summary(d)
    for key in ("final_health", "mean_health", "min_health", "max_health",
                "n_turning_points", "emotional_contagion"):
        assert key in summary


def test_health_summary_empty():
    from emotion_analysis.models import EmotionalDynamics
    assert conversation_health_summary(EmotionalDynamics()) == {}
