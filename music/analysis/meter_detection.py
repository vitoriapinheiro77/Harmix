"""Select a time signature from competing Madmom beat paths."""

from __future__ import annotations

from statistics import mean, median

import numpy as np


METER_CONFIG = {
    2: {'beat_unit': 4, 'label': '2/4', 'accent_positions': (1,)},
    3: {'beat_unit': 4, 'label': '3/4', 'accent_positions': (1,)},
    4: {'beat_unit': 4, 'label': '4/4', 'accent_positions': (1, 3)},
    6: {'beat_unit': 8, 'label': '6/8', 'accent_positions': (1, 4)},
}
MIN_BARS = 3
MIN_SIX_EIGHT_SCORE = 0.58
MIN_SIX_EIGHT_MARGIN = 0.06
MIN_RELATIVE_SIX_EIGHT_SCORE = 0.22
MIN_COMPOUND_ACCENT_ADVANTAGE = 0.15
MIN_DOWNBEAT_ADVANTAGE = 0.05


def _activation_at(activations, time_value, column, fps):
    if activations is None or len(activations) == 0:
        return 0.0
    index = max(0, min(len(activations) - 1, int(round(float(time_value) * fps))))
    return float(activations[index][column])


def score_meter_path(path, activations, beats_per_measure, fps=100):
    """Score a DBN path using activation, regularity and internal accents."""
    rows = [
        (float(row[0]), int(row[1]))
        for row in np.atleast_2d(path)
        if len(row) >= 2 and 1 <= int(row[1]) <= beats_per_measure
    ]
    bar_count = sum(position == 1 for _time, position in rows)
    if len(rows) < beats_per_measure * MIN_BARS or bar_count < MIN_BARS:
        return {
            'score': 0.0,
            'bar_count': bar_count,
            'event_count': len(rows),
            'status': 'insufficient_evidence',
        }

    intervals = [right[0] - left[0] for left, right in zip(rows, rows[1:]) if right[0] > left[0]]
    if not intervals:
        return {'score': 0.0, 'status': 'insufficient_evidence', 'bar_count': bar_count}
    typical_interval = median(intervals)
    # Compare neighboring intervals so sustained tempo changes are not
    # penalized throughout the entire recording.
    deviations = [
        abs(right - left) / max(left, right)
        for left, right in zip(intervals, intervals[1:])
    ]
    regularity = max(0.0, 1.0 - mean(deviations) / 0.20) if deviations else 0.0

    downbeat_strength = mean(
        _activation_at(activations, time_value, 1, fps)
        for time_value, position in rows
        if position == 1
    )
    beat_strength = mean(
        _activation_at(activations, time_value, 0, fps)
        for time_value, position in rows
        if position != 1
    )

    accent_positions = METER_CONFIG[beats_per_measure]['accent_positions']
    accented = [
        _activation_at(activations, time_value, 1 if position == 1 else 0, fps)
        for time_value, position in rows
        if position in accent_positions
    ]
    unaccented = [
        _activation_at(activations, time_value, 0, fps)
        for time_value, position in rows
        if position not in accent_positions
    ]
    accent_mean = mean(accented) if accented else 0.0
    unaccented_mean = mean(unaccented) if unaccented else 0.0
    accent_contrast = max(0.0, min(1.0, 0.5 + accent_mean - unaccented_mean))

    # The score deliberately balances four independent signals. BPM is not
    # included: it can be represented at dotted-quarter or eighth-note level.
    score = (
        0.30 * downbeat_strength
        + 0.20 * beat_strength
        + 0.30 * regularity
        + 0.20 * accent_contrast
    )
    return {
        'score': round(max(0.0, min(1.0, score)), 4),
        'bar_count': bar_count,
        'event_count': len(rows),
        'downbeat_strength': round(downbeat_strength, 4),
        'beat_strength': round(beat_strength, 4),
        'regularity': round(regularity, 4),
        'accent_contrast': round(accent_contrast, 4),
        'median_subdivision_seconds': round(typical_interval, 4),
            'status': 'scored',
    }


def select_meter(hypotheses):
    """Require a clear lead before choosing a formula other than 4/4."""
    four = hypotheses.get(4, {})
    six = hypotheses.get(6, {})
    four_score = float(four.get('score') or 0.0)
    six_score = float(six.get('score') or 0.0)
    margin = six_score - four_score
    accent_advantage = float(six.get('accent_contrast') or 0.0) - float(
        four.get('accent_contrast') or 0.0
    )
    downbeat_advantage = float(six.get('downbeat_strength') or 0.0) - float(
        four.get('downbeat_strength') or 0.0
    )
    relative_compound_evidence = (
        six_score >= MIN_RELATIVE_SIX_EIGHT_SCORE
        and accent_advantage >= MIN_COMPOUND_ACCENT_ADVANTAGE
        and downbeat_advantage >= MIN_DOWNBEAT_ADVANTAGE
    )
    six_has_evidence = (
        six.get('status') == 'scored'
        and int(six.get('bar_count') or 0) >= MIN_BARS
        and margin >= MIN_SIX_EIGHT_MARGIN
        and (six_score >= MIN_SIX_EIGHT_SCORE or relative_compound_evidence)
    )
    beats_per_measure = 6 if six_has_evidence else 4
    for candidate in (2, 3):
        evidence = hypotheses.get(candidate, {})
        score = float(evidence.get('score') or 0.0)
        competing_score = max(
            float(other.get('score') or 0.0)
            for meter, other in hypotheses.items() if meter != candidate
        ) if len(hypotheses) > 1 else 0.0
        if (
            evidence.get('status') == 'scored'
            and score >= MIN_SIX_EIGHT_SCORE
            and score - competing_score >= MIN_SIX_EIGHT_MARGIN
            and evidence.get('downbeat_strength', 0) - four.get('downbeat_strength', 0)
            >= MIN_DOWNBEAT_ADVANTAGE
        ):
            beats_per_measure = candidate
    if beats_per_measure == 6 and any(
        hypotheses.get(candidate, {}).get('score', 0.0) > six_score - MIN_SIX_EIGHT_MARGIN
        for candidate in (2, 3) if hypotheses.get(candidate, {}).get('status') == 'scored'
    ):
        beats_per_measure = 4
    selected = hypotheses.get(beats_per_measure, {})
    selected_score = float(selected.get('score') or 0.0)
    selected_margin = selected_score - max(
        (float(other.get('score') or 0.0) for meter, other in hypotheses.items()
         if meter != beats_per_measure),
        default=0.0,
    )
    ambiguous = selected_margin < MIN_SIX_EIGHT_MARGIN
    confidence = max(0.0, min(1.0, selected_score * (0.75 + min(abs(selected_margin), 0.25))))
    if selected_margin < 0:
        confidence *= 0.7
    return {
        'beats_per_measure': beats_per_measure,
        'beat_unit': METER_CONFIG[beats_per_measure]['beat_unit'],
        'label': METER_CONFIG[beats_per_measure]['label'],
        'confidence': round(confidence, 3),
        'low_confidence': confidence < 0.60 or ambiguous,
        'decision': {
            2: 'two_four_evidence', 3: 'three_four_evidence',
            4: 'four_four_default_or_evidence', 6: 'six_eight_evidence',
        }[beats_per_measure],
        'selected_score_margin': round(selected_margin, 4),
        'score_margin_6_over_4': round(margin, 4),
        'compound_accent_advantage': round(accent_advantage, 4),
        'downbeat_advantage': round(downbeat_advantage, 4),
        'hypotheses': {str(key): value for key, value in hypotheses.items()},
    }


def detect_meter(paths, activations, fps=100):
    """Score the available DBN paths and return the selected meter."""
    hypotheses = {
        meter: score_meter_path(paths.get(meter, []), activations, meter, fps=fps)
        for meter in METER_CONFIG
    }
    return select_meter(hypotheses)

