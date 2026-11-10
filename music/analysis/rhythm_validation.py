"""Validate detector events before they become the musical beat grid."""

import logging
from collections import Counter
from dataclasses import dataclass
from statistics import median


logger = logging.getLogger('harmix.analysis')


@dataclass(frozen=True)
class RhythmValidationThresholds:
    duplicate_max_ratio: float = 0.68
    missing_min_ratio: float = 1.55
    missing_max_ratio: float = 2.45
    tempo_change_min_ratio: float = 0.70
    tempo_change_max_ratio: float = 1.30
    downbeat_match_max_ratio: float = 0.30
    downbeat_phase_margin: float = 0.10
    local_interval_window: int = 8
    tempo_change_confirmation_intervals: int = 3
    tempo_change_consistency_ratio: float = 0.15


DEFAULT_RHYTHM_THRESHOLDS = RhythmValidationThresholds()


def _local_interval(beats, nominal, thresholds):
    intervals = [
        right - left for left, right in zip(beats, beats[1:])
        if nominal * thresholds.tempo_change_min_ratio
        <= right - left
        <= nominal * thresholds.tempo_change_max_ratio
    ]
    return median(intervals[-thresholds.local_interval_window:]) if intervals else nominal


def validate_rhythm_grid(rhythm, thresholds=DEFAULT_RHYTHM_THRESHOLDS):
    """Clean beat events and select canonical downbeats for the chosen meter.

    Madmom events remain available in raw_beats/raw_downbeats. This function
    does not double the beat count when the DBN reports half-bar downbeats.
    """
    raw_beats = sorted({float(value) for value in rhythm.beats if float(value) >= 0})
    raw_downbeats = sorted({float(value) for value in rhythm.downbeats if float(value) >= 0})
    rhythm.raw_beats = raw_beats
    rhythm.raw_downbeats = raw_downbeats
    selected_hypothesis = (
        (rhythm.meter.get('hypotheses') or {}).get(str(rhythm.beats_per_measure), {})
        if rhythm.meter else {}
    )
    nominal = (
        selected_hypothesis.get('median_subdivision_seconds')
        if rhythm.beats_per_measure == 6
        else None
    )
    if not nominal:
        nominal = 60.0 / rhythm.bpm if rhythm.bpm and rhythm.bpm > 0 else None
    if nominal is None and len(raw_beats) > 1:
        nominal = median(right - left for left, right in zip(raw_beats, raw_beats[1:]))
    if not raw_beats or not nominal:
        rhythm.validation = {'status': 'insufficient_beats', 'confidence': 0.0}
        return rhythm

    kept = [raw_beats[0]]
    discarded = []
    inferred = []
    tempo_reference = nominal
    for position, candidate in enumerate(raw_beats[1:], start=1):
        local = _local_interval(kept, tempo_reference, thresholds)
        gap = candidate - kept[-1]
        if gap < local * thresholds.duplicate_max_ratio:
            discarded.append(candidate)
            continue
        if local * thresholds.missing_min_ratio <= gap <= local * thresholds.missing_max_ratio:
            lookahead = raw_beats[
                position:position + thresholds.tempo_change_confirmation_intervals + 1
            ]
            future_intervals = [
                right - left for left, right in zip(lookahead, lookahead[1:])
            ]
            sustained_change = (
                len(future_intervals) == thresholds.tempo_change_confirmation_intervals
                and all(
                    abs(interval - gap) <= gap * thresholds.tempo_change_consistency_ratio
                    for interval in future_intervals
                )
            )
            if sustained_change:
                tempo_reference = gap
            else:
                inserted = kept[-1] + gap / 2.0
                inferred.append(inserted)
                kept.append(inserted)
        kept.append(candidate)

    matches = []
    for raw_downbeat in raw_downbeats:
        index = min(range(len(kept)), key=lambda item: abs(kept[item] - raw_downbeat))
        if abs(kept[index] - raw_downbeat) <= nominal * thresholds.downbeat_match_max_ratio:
            matches.append(index)

    if matches:
        phase_counts = Counter(index % rhythm.beats_per_measure for index in matches)
        # A tied half-bar pattern keeps the first DBN anchor instead of
        # shifting the whole song by two beats.
        first_phase = matches[0] % rhythm.beats_per_measure
        best_phase = max(
            phase_counts,
            key=lambda value: (phase_counts[value], value == first_phase),
        )
        phase = (
            first_phase
            if phase_counts[best_phase] - phase_counts[first_phase]
            <= len(matches) * thresholds.downbeat_phase_margin
            else best_phase
        )
        anchor = next(index for index in matches if index % rhythm.beats_per_measure == phase)
        phase_agreement = phase_counts[phase] / len(matches)
    else:
        anchor = 0
        phase_agreement = 0.0

    canonical_downbeats = [
        kept[index]
        for index in range(anchor, len(kept), rhythm.beats_per_measure)
    ]
    rhythm.beats = kept
    rhythm.downbeats = canonical_downbeats
    rhythm.validation = {
        'status': 'validated',
        'time_signature': f'{rhythm.beats_per_measure}/{rhythm.beat_unit}',
        'meter_confidence': rhythm.meter.get('confidence', 0.0) if rhythm.meter else 0.0,
        'meter_low_confidence': rhythm.meter.get('low_confidence', True) if rhythm.meter else True,
        'meter_evidence': rhythm.meter,
        'confidence': round(
            max(0.0, min(1.0, (
                (len(kept) - len(inferred)) / max(len(kept), 1)
            ) * (0.5 + 0.5 * phase_agreement))),
            3,
        ),
        'nominal_beat_seconds': round(nominal, 4),
        'raw_beat_count': len(raw_beats),
        'kept_beat_count': len(kept) - len(inferred),
        'discarded_subdivisions': len(discarded),
        'discarded_sample': [round(value, 3) for value in discarded[:20]],
        'inferred_beats': len(inferred),
        'inferred_sample': [round(value, 3) for value in inferred[:20]],
        'raw_downbeat_count': len(raw_downbeats),
        'canonical_downbeat_count': len(canonical_downbeats),
        'downbeat_phase_agreement': round(phase_agreement, 3),
        'anchor_seconds': round(kept[anchor], 3),
        'canonical_downbeats_sample': [round(value, 3) for value in canonical_downbeats[:20]],
    }
    logger.info('Grade ritmica validada: %s', rhythm.validation)
    return rhythm

