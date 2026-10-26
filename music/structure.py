from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import logging
from math import ceil, floor

from .analysis.chord_stability import harmonic_family


DEFAULT_BEATS_PER_MEASURE = 4
DEFAULT_BEAT_UNIT = 4
COMPLEX_CHORD_CONSENSUS = 0.75
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeasureDecisionThresholds:
    """Initial calibration values for measure-level harmony decisions."""

    boundary_tolerance_beats: float = 0.5
    small_anticipation_beats: float = 1.0
    late_entry_window_beats: float = 1.5
    min_real_change_beats: float = 1.5
    min_real_change_ratio: float = 0.35
    next_measure_confirmation_beats: float = 3.0
    near_tie_gap_beats: float = 0.35
    near_tie_gap_ratio: float = 0.10
    balanced_split_min_ratio: float = 0.40
    balanced_split_max_ratio: float = 0.60
    entry_priority_window_beats: float = 0.75
    strong_real_change_beats: float = 2.25
    strong_real_change_ratio: float = 0.55
    pattern_cycle_lengths: tuple[int, ...] = (2, 4, 8)
    pattern_min_repetitions: float = 3.0
    pattern_min_slot_observations: int = 2
    pattern_min_coverage: float = 0.75
    pattern_min_global_agreement: float = 0.70
    pattern_reliable_primary_ratio: float = 0.65
    pattern_min_secondary_start_beat: float = 1.5
    pattern_max_ambiguous_beats: float = 2.5
    pattern_max_ambiguous_ratio: float = 0.55
    pattern_section_window_measures: int = 8
    pattern_section_min_window_measures: int = 3
    pattern_section_break_match_rate: float = 0.5
    phase_min_global_agreement: float = 0.70
    phase_local_window_measures: int = 12
    phase_min_local_agreement: float = 0.80
    phase_min_margin: float = 0.12
    phase_min_reliable_anchors: int = 6
    phase_anchor_max_distance: int = 3
    phase_override_min_global_agreement: float = 0.85
    phase_override_min_repetitions: float = 8.0
    phase_override_min_local_agreement: float = 0.70
    single_local_contradiction_beats: float = 2.5
    single_local_contradiction_ratio: float = 0.65
    single_expected_min_support_beats: float = 1.5
    single_expected_min_support_ratio: float = 0.35


@dataclass(frozen=True)
class MeasureHarmonyDecision:
    measure: Measure
    evidence: dict
    eligible: dict
    selected_families: tuple
    duration_winner_family: tuple | None
    entry_family: tuple | None
    primary_family: tuple | None
    pattern_learning_family: tuple | None
    primary_ratio: float
    secondary_family: tuple | None
    secondary_duration_beats: float
    secondary_ratio: float
    secondary_start_beat: float | None
    status: str
    corrected_by_pattern: bool = False
    pattern_reason: str | None = None


@dataclass(frozen=True)
class HarmonicPatternCandidate:
    cycle_length: int
    phase: int
    expected_primary_families: tuple
    global_agreement: float
    coverage: float
    repetition_count: float
    ambiguity_support: int


DEFAULT_MEASURE_DECISION_THRESHOLDS = MeasureDecisionThresholds()


@dataclass(frozen=True)
class TimeSignature:
    beats_per_measure: int = DEFAULT_BEATS_PER_MEASURE
    beat_unit: int = DEFAULT_BEAT_UNIT

    def as_dict(self):
        return {
            'beats_per_measure': self.beats_per_measure,
            'beat_unit': self.beat_unit,
            'label': f'{self.beats_per_measure}/{self.beat_unit}',
        }


@dataclass(frozen=True)
class ChordPlacement:
    chord_name: str
    start_time: float
    end_time: float
    absolute_start_beat: float
    absolute_end_beat: float
    beat_start: float
    beat_duration: float

    def as_dict(self):
        return {
            'chord_name': self.chord_name,
            'start_time': round(self.start_time, 2),
            'end_time': round(self.end_time, 2),
            'absolute_start_beat': round(self.absolute_start_beat, 3),
            'absolute_end_beat': round(self.absolute_end_beat, 3),
            'beat_start': round(self.beat_start, 3),
            'beat_duration': round(self.beat_duration, 3),
        }


@dataclass(frozen=True)
class Measure:
    number: int
    start_beat: float
    end_beat: float
    chords: list[ChordPlacement]

    def as_dict(self):
        return {
            'number': self.number,
            'start_beat': round(self.start_beat, 3),
            'end_beat': round(self.end_beat, 3),
            'chords': [chord.as_dict() for chord in self.chords],
        }


class MusicalStructureBuilder:
    def __init__(
        self,
        bpm,
        time_signature=None,
        rhythm_grid=None,
        decision_thresholds=None,
    ):
        if not bpm or bpm <= 0:
            raise ValueError('BPM valido e necessario para converter segundos em compassos.')
        self.bpm = float(bpm)
        self.time_signature = time_signature or TimeSignature()
        self.rhythm_grid = rhythm_grid or {}
        self.decision_thresholds = (
            decision_thresholds or DEFAULT_MEASURE_DECISION_THRESHOLDS
        )
        self.beat_offset = 0.0
        downbeats = self.rhythm_grid.get('downbeats') or []
        if downbeats:
            self.beat_offset = float(downbeats[0])
        self.beat_times = sorted(
            float(value)
            for value in (self.rhythm_grid.get('beats') or [])
        )
        self.beat_anchor_index = (
            min(
                range(len(self.beat_times)),
                key=lambda index: abs(self.beat_times[index] - self.beat_offset),
            )
            if self.beat_times
            else 0
        )

    def seconds_to_beats(self, seconds):
        value = float(seconds)
        if not self.beat_times:
            adjusted = max(0.0, value - self.beat_offset)
            return adjusted * self.bpm / 60.0

        anchor_time = self.beat_times[self.beat_anchor_index]
        if value <= anchor_time:
            return 0.0

        right_index = bisect_right(self.beat_times, value)
        if right_index >= len(self.beat_times):
            last_index = len(self.beat_times) - 1
            seconds_per_beat = (
                self.beat_times[-1] - self.beat_times[-2]
                if len(self.beat_times) > 1
                else 60.0 / self.bpm
            )
            return (
                last_index
                - self.beat_anchor_index
                + ((value - self.beat_times[-1]) / seconds_per_beat)
            )

        left_index = max(0, right_index - 1)
        left_time = self.beat_times[left_index]
        right_time = self.beat_times[right_index]
        fraction = (
            (value - left_time) / (right_time - left_time)
            if right_time > left_time
            else 0.0
        )
        return max(0.0, left_index - self.beat_anchor_index + fraction)

    def beats_to_measure_number(self, beat_position):
        return floor(float(beat_position) / self.time_signature.beats_per_measure) + 1

    def build(self, chord_events):
        normalized_chords = self._normalize_chords(chord_events)
        family_labels = self._choose_family_labels(normalized_chords)
        last_end_beat = self._last_end_beat(normalized_chords)
        measure_count = max(1, ceil(last_end_beat / self.time_signature.beats_per_measure))
        measures = [
            Measure(
                number=number,
                start_beat=(number - 1) * self.time_signature.beats_per_measure,
                end_beat=number * self.time_signature.beats_per_measure,
                chords=[],
            )
            for number in range(1, measure_count + 1)
        ]

        for chord in normalized_chords:
            self._place_chord_in_measures(chord, measures)

        measure_evidence = [
            self._collect_measure_evidence(measure)
            for measure in measures
        ]
        decisions = [
            self._decide_measure_harmony(
                measure,
                measure_evidence[index],
                measures[index + 1] if index + 1 < len(measures) else None,
                measure_evidence[index + 1]
                if index + 1 < len(measure_evidence)
                else None,
            )
            for index, measure in enumerate(measures)
        ]
        provisional_sequence = ' | '.join(
            self._family_label(decision.primary_family, family_labels)
            for decision in decisions
        )
        reliable_measure_count = sum(
            1 for decision in decisions if decision.pattern_learning_family
        )
        logger.info(
            'Provisional primary chord sequence: %s',
            provisional_sequence,
        )
        logger.info(
            'Harmonic pattern learning: reliable_measures=%s total_measures=%s excluded_non_reliable=%s.',
            reliable_measure_count,
            len(decisions),
            len(decisions) - reliable_measure_count,
        )
        pattern = self._detect_harmonic_pattern(decisions, family_labels)
        if pattern:
            decisions = self._apply_harmonic_pattern(
                decisions,
                pattern,
                family_labels,
            )
            decisions = self._apply_phase_based_single_corrections(
                decisions,
                pattern,
                family_labels,
            )
        measures = [
            self._render_measure_harmony(
                decision.measure,
                family_labels,
                decision.eligible,
                decision.selected_families,
            )
            for decision in decisions
        ]

        return {
            'bpm': round(self.bpm, 2),
            'time_signature': self.time_signature.as_dict(),
            'measure_count': len(measures),
            'measures': [measure.as_dict() for measure in measures],
            'rhythm_grid': {
                'source': self.rhythm_grid.get('source'),
                'beat_offset_seconds': round(self.beat_offset, 3),
                'beats_per_measure': self.rhythm_grid.get('beats_per_measure', self.time_signature.beats_per_measure),
                'total_beats': self.rhythm_grid.get('total_beats'),
                'total_downbeats': self.rhythm_grid.get('total_downbeats'),
            },
        }

    def _normalize_chords(self, chord_events):
        normalized = []
        for event in chord_events:
            try:
                start_time = float(event['start_time'])
                end_time = float(event['end_time'])
            except (KeyError, TypeError, ValueError):
                continue
            chord_name = str(event.get('chord_name') or '').strip()
            if not chord_name or end_time <= start_time:
                continue
            normalized.append(
                {
                    'chord_name': chord_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_beat': self.seconds_to_beats(start_time),
                    'end_beat': self.seconds_to_beats(end_time),
                    'family': harmonic_family(chord_name),
                }
            )
        return sorted(normalized, key=lambda item: item['start_beat'])

    def _last_end_beat(self, normalized_chords):
        if not normalized_chords:
            return self.time_signature.beats_per_measure
        return max(chord['end_beat'] for chord in normalized_chords)

    def _choose_family_labels(self, normalized_chords):
        """Choose stable labels using duration consensus across the whole song."""
        by_family = {}
        for chord in normalized_chords:
            duration = max(0.0, chord['end_beat'] - chord['start_beat'])
            if duration <= 0:
                continue
            labels = by_family.setdefault(chord['family'], {})
            labels[chord['chord_name']] = labels.get(chord['chord_name'], 0.0) + duration

        chosen = {}
        for family, labels in by_family.items():
            total = sum(labels.values())
            strongest = max(labels, key=labels.get)
            if labels[strongest] / total >= COMPLEX_CHORD_CONSENSUS:
                chosen[family] = strongest
            else:
                chosen[family] = self._base_family_label(family)
        return chosen

    def _base_family_label(self, family):
        root, quality = family
        suffix = {
            'minor': 'm',
            'diminished': 'dim',
            'augmented': 'aug',
        }.get(quality, '')
        return f'{root}{suffix}'

    def _place_chord_in_measures(self, chord, measures):
        """Split detections only for measuring their real overlap per bar."""
        beats_per_measure = self.time_signature.beats_per_measure
        first_measure = self.beats_to_measure_number(chord['start_beat'])
        last_measure = self.beats_to_measure_number(
            max(chord['end_beat'] - 0.0001, chord['start_beat'])
        )

        for measure_number in range(first_measure, last_measure + 1):
            if not 1 <= measure_number <= len(measures):
                continue
            measure = measures[measure_number - 1]
            segment_start = max(chord['start_beat'], measure.start_beat)
            segment_end = min(chord['end_beat'], measure.end_beat)
            if segment_end <= segment_start:
                continue
            measure.chords.append(
                ChordPlacement(
                    chord_name=chord['chord_name'],
                    start_time=chord['start_time'],
                    end_time=chord['end_time'],
                    absolute_start_beat=segment_start,
                    absolute_end_beat=segment_end,
                    beat_start=segment_start - measure.start_beat,
                    beat_duration=segment_end - segment_start,
                )
            )

    def _collect_measure_evidence(self, measure):
        """Aggregate real overlap by harmonic family inside one measure."""
        evidence = {}
        for placement in measure.chords:
            family = harmonic_family(placement.chord_name)
            family_evidence = evidence.setdefault(
                family,
                {
                    'intervals': [],
                    'source_label': placement.chord_name,
                },
            )
            family_evidence['intervals'].append(
                (placement.absolute_start_beat, placement.absolute_end_beat)
            )

        for family_evidence in evidence.values():
            intervals = sorted(family_evidence.pop('intervals'))
            merged = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            family_evidence['duration'] = sum(
                end - start for start, end in merged
            )
            family_evidence['start'] = merged[0][0]
            family_evidence['end'] = merged[-1][1]
        return evidence

    def _predominant_family(self, evidence):
        if not evidence:
            return None
        return max(
            evidence,
            key=lambda family: (
                evidence[family]['duration'],
                -evidence[family]['start'],
            ),
        )

    def _is_confirmed_anticipation(
        self,
        family,
        measure,
        evidence,
        next_measure,
        next_evidence,
    ):
        """Identify a late fragment that belongs harmonically to the next bar."""
        if not next_measure or not next_evidence or len(evidence) < 2:
            return False

        thresholds = self.decision_thresholds
        current = evidence[family]
        following = next_evidence.get(family)
        if not following:
            return False

        starts_late = current['start'] >= (
            measure.end_beat - thresholds.late_entry_window_beats
        )
        reaches_boundary = current['end'] >= (
            measure.end_beat - thresholds.boundary_tolerance_beats
        )
        starts_near_next_downbeat = following['start'] <= (
            next_measure.start_beat + thresholds.boundary_tolerance_beats
        )
        next_predominant = self._predominant_family(next_evidence)
        strongly_confirmed_next = (
            next_predominant == family
            and following['duration']
            >= thresholds.next_measure_confirmation_beats
        )
        if not (
            starts_late
            and reaches_boundary
            and starts_near_next_downbeat
            and strongly_confirmed_next
        ):
            return False

        if current['duration'] <= thresholds.small_anticipation_beats:
            return True

        # The interval between a clear anticipation and a clear real change is
        # intentionally resolved using the strong evidence from the next bar.
        return current['duration'] <= thresholds.min_real_change_beats

    def _decide_measure_harmony(
        self,
        measure,
        evidence,
        next_measure=None,
        next_evidence=None,
    ):
        """Choose local measure harmony while consulting the following measure."""
        if not evidence:
            return MeasureHarmonyDecision(
                measure=measure,
                evidence=evidence,
                eligible={},
                selected_families=(),
                duration_winner_family=None,
                entry_family=None,
                primary_family=None,
                pattern_learning_family=None,
                primary_ratio=0.0,
                secondary_family=None,
                secondary_duration_beats=0.0,
                secondary_ratio=0.0,
                secondary_start_beat=None,
                status='empty',
            )

        suppressed_anticipations = {
            family
            for family in evidence
            if self._is_confirmed_anticipation(
                family,
                measure,
                evidence,
                next_measure,
                next_evidence,
            )
        }
        eligible = {
            family: details
            for family, details in evidence.items()
            if family not in suppressed_anticipations
        }
        if not eligible:
            eligible = evidence

        duration_winner = self._predominant_family(eligible)
        entry_family = min(
            eligible,
            key=lambda family: (
                eligible[family]['start'],
                -eligible[family]['duration'],
            ),
        )
        selected = [duration_winner]
        thresholds = self.decision_thresholds
        for family, details in eligible.items():
            if family == duration_winner:
                continue
            duration = details['duration']
            if (
                duration >= thresholds.min_real_change_beats
                and duration / self.time_signature.beats_per_measure
                >= thresholds.min_real_change_ratio
            ):
                selected.append(family)
        selected.sort(key=lambda family: eligible[family]['start'])

        primary_family = duration_winner
        pattern_learning_family = duration_winner
        status = 'single'
        is_ambiguous_balanced_split = False
        if len(selected) == 2:
            ranked_by_duration = sorted(
                selected,
                key=lambda family: eligible[family]['duration'],
                reverse=True,
            )
            winner_duration = eligible[ranked_by_duration[0]]['duration']
            runner_up_duration = eligible[ranked_by_duration[1]]['duration']
            winner_ratio = winner_duration / self.time_signature.beats_per_measure
            runner_up_ratio = (
                runner_up_duration / self.time_signature.beats_per_measure
            )
            duration_gap = winner_duration - runner_up_duration
            duration_gap_ratio = (
                duration_gap / self.time_signature.beats_per_measure
            )
            is_near_tie = (
                duration_gap <= thresholds.near_tie_gap_beats
                or duration_gap_ratio <= thresholds.near_tie_gap_ratio
            )
            is_balanced = (
                thresholds.balanced_split_min_ratio
                <= winner_ratio
                <= thresholds.balanced_split_max_ratio
                and thresholds.balanced_split_min_ratio
                <= runner_up_ratio
                <= thresholds.balanced_split_max_ratio
            )
            entry_is_clear = eligible[entry_family]['start'] <= (
                measure.start_beat + thresholds.entry_priority_window_beats
            )
            late_family = selected[1]
            late_duration = eligible[late_family]['duration']
            late_ratio = late_duration / self.time_signature.beats_per_measure
            has_strong_late_change = (
                late_duration >= thresholds.strong_real_change_beats
                or late_ratio >= thresholds.strong_real_change_ratio
            )
            is_ambiguous_balanced_split = (
                is_near_tie
                and is_balanced
                and entry_is_clear
                and not has_strong_late_change
            )
            if is_ambiguous_balanced_split:
                primary_family = entry_family
                pattern_learning_family = None
                status = 'ambiguous_balanced_split'

        primary_ratio = (
            eligible[primary_family]['duration'] / self.time_signature.beats_per_measure
            if primary_family
            else 0.0
        )
        secondary_family = next(
            (
                family
                for family in selected
                if family != primary_family
            ),
            None,
        )
        secondary_duration_beats = (
            eligible[secondary_family]['duration']
            if secondary_family
            else 0.0
        )
        secondary_ratio = (
            secondary_duration_beats / self.time_signature.beats_per_measure
            if secondary_family
            else 0.0
        )
        secondary_start_beat = (
            max(0.0, eligible[secondary_family]['start'] - measure.start_beat)
            if secondary_family
            else None
        )

        if len(selected) > 1 and not is_ambiguous_balanced_split:
            starts_late = (
                secondary_start_beat is not None
                and secondary_start_beat >= thresholds.pattern_min_secondary_start_beat
            )
            is_pattern_ambiguous = (
                starts_late
                and secondary_duration_beats <= thresholds.pattern_max_ambiguous_beats
                and secondary_ratio <= thresholds.pattern_max_ambiguous_ratio
            )
            status = 'ambiguous' if is_pattern_ambiguous else 'multi'
            if status == 'ambiguous':
                pattern_learning_family = None
        elif primary_ratio < thresholds.pattern_reliable_primary_ratio:
            if not is_ambiguous_balanced_split:
                status = 'ambiguous'
                pattern_learning_family = None

        if is_ambiguous_balanced_split:
            logger.info(
                'Measure %s classified as ambiguous_balanced_split: entry=%s duration_winner=%s entry_beats=%.3f winner_beats=%.3f gap_beats=%.3f gap_ratio=%.3f; excluded from harmonic pattern learning.',
                measure.number,
                eligible[entry_family]['source_label'],
                eligible[duration_winner]['source_label'],
                eligible[entry_family]['duration'],
                eligible[duration_winner]['duration'],
                duration_gap,
                duration_gap_ratio,
            )

        return MeasureHarmonyDecision(
            measure=measure,
            evidence=evidence,
            eligible=eligible,
            selected_families=tuple(selected),
            duration_winner_family=duration_winner,
            entry_family=entry_family,
            primary_family=primary_family,
            pattern_learning_family=pattern_learning_family,
            primary_ratio=primary_ratio,
            secondary_family=secondary_family,
            secondary_duration_beats=secondary_duration_beats,
            secondary_ratio=secondary_ratio,
            secondary_start_beat=secondary_start_beat,
            status=status,
        )

    def _render_measure_harmony(
        self,
        measure,
        family_labels,
        eligible,
        selected_families,
    ):
        if not eligible or not selected_families:
            return measure

        placements = []
        for index, family in enumerate(selected_families):
            beat_start = (
                0.0
                if index == 0
                else max(0.0, eligible[family]['start'] - measure.start_beat)
            )
            next_start = (
                max(
                    beat_start,
                    eligible[selected_families[index + 1]]['start'] - measure.start_beat,
                )
                if index + 1 < len(selected_families)
                else self.time_signature.beats_per_measure
            )
            beat_duration = max(0.0, next_start - beat_start)
            if beat_duration <= 0:
                continue
            placements.append(
                ChordPlacement(
                    chord_name=family_labels.get(
                        family,
                        eligible[family]['source_label'],
                    ),
                    start_time=measure.start_beat * 60.0 / self.bpm,
                    end_time=measure.end_beat * 60.0 / self.bpm,
                    absolute_start_beat=measure.start_beat + beat_start,
                    absolute_end_beat=measure.start_beat + next_start,
                    beat_start=beat_start,
                    beat_duration=beat_duration,
                )
            )

        return Measure(
            number=measure.number,
            start_beat=measure.start_beat,
            end_beat=measure.end_beat,
            chords=placements,
        )

    def _detect_harmonic_pattern(self, decisions, family_labels):
        candidates = []
        for cycle_length in self.decision_thresholds.pattern_cycle_lengths:
            if not cycle_length or cycle_length > len(decisions):
                continue
            for phase in range(cycle_length):
                candidate = self._score_harmonic_pattern_candidate(
                    decisions,
                    cycle_length,
                    phase,
                )
                if candidate:
                    candidates.append(candidate)

        if not candidates:
            logger.info('No harmonic cycle pattern detected for current structure.')
            return None

        pattern = max(
            candidates,
            key=lambda item: (
                item.global_agreement,
                item.ambiguity_support,
                item.repetition_count,
                -item.cycle_length,
            ),
        )
        display_families = self._pattern_families_for_display(pattern)
        sequence = ' | '.join(
            self._family_label(family, family_labels)
            for family in display_families
        )
        logger.info(
            'Detected harmonic cycle: length=%s phase=%s agreement=%.2f coverage=%.2f repetitions=%.2f pattern=%s',
            pattern.cycle_length,
            pattern.phase,
            pattern.global_agreement,
            pattern.coverage,
            pattern.repetition_count,
            sequence,
        )
        logger.info(
            'Detected cycle phase: phase=%s expected_measure_one=%s global_agreement=%.2f.',
            pattern.phase,
            self._family_label(
                self._expected_pattern_family(pattern, 0),
                family_labels,
            ),
            pattern.global_agreement,
        )
        return pattern

    def _pattern_families_for_display(self, pattern):
        families = pattern.expected_primary_families
        for index, family in enumerate(families):
            if family == families[(index + 1) % pattern.cycle_length]:
                return families[index:] + families[:index]
        return families

    def _score_harmonic_pattern_candidate(self, decisions, cycle_length, phase):
        slots = [
            {'weights': {}, 'observations': 0}
            for _ in range(cycle_length)
        ]
        total_weight = 0.0
        matched_weight = 0.0

        for index, decision in enumerate(decisions):
            learning_family = decision.pattern_learning_family
            if not learning_family:
                continue
            slot_index = index % cycle_length
            weight = self._pattern_learning_weight(decision)
            slot = slots[slot_index]
            slot['weights'][learning_family] = (
                slot['weights'].get(learning_family, 0.0) + weight
            )
            slot['observations'] += 1

        expected = []
        covered_slots = 0
        for slot in slots:
            if not slot['weights']:
                expected.append(None)
                continue
            family = max(slot['weights'], key=slot['weights'].get)
            expected.append(family)
            if slot['observations'] >= self.decision_thresholds.pattern_min_slot_observations:
                covered_slots += 1

        expected = self._canonicalize_harmonic_cycle(tuple(expected))

        observed_measures = sum(
            1 for decision in decisions if decision.pattern_learning_family
        )
        if observed_measures == 0:
            return None

        for index, decision in enumerate(decisions):
            learning_family = decision.pattern_learning_family
            if not learning_family:
                continue
            slot_index = (index + phase) % cycle_length
            expected_family = expected[slot_index]
            if expected_family is None:
                continue
            weight = self._pattern_learning_weight(decision)
            total_weight += weight
            if learning_family == expected_family:
                matched_weight += weight

        if total_weight <= 0:
            return None

        coverage = covered_slots / cycle_length
        global_agreement = matched_weight / total_weight
        repetition_count = observed_measures / cycle_length
        if (
            repetition_count < self.decision_thresholds.pattern_min_repetitions
            or coverage < self.decision_thresholds.pattern_min_coverage
            or global_agreement < self.decision_thresholds.pattern_min_global_agreement
        ):
            return None

        candidate = HarmonicPatternCandidate(
            cycle_length=cycle_length,
            phase=phase,
            expected_primary_families=tuple(expected),
            global_agreement=global_agreement,
            coverage=coverage,
            repetition_count=repetition_count,
            ambiguity_support=0,
        )
        ambiguity_support = sum(
            1
            for index, decision in enumerate(decisions)
            if self._can_pattern_correct_measure(
                decisions,
                index,
                candidate,
                require_active_section=False,
            )
        )
        return HarmonicPatternCandidate(
            cycle_length=cycle_length,
            phase=phase,
            expected_primary_families=candidate.expected_primary_families,
            global_agreement=candidate.global_agreement,
            coverage=candidate.coverage,
            repetition_count=candidate.repetition_count,
            ambiguity_support=ambiguity_support,
        )

    def _canonicalize_harmonic_cycle(self, families):
        """Rotate a cycle once so phase scoring has a stable reference."""
        if not families:
            return families
        for index, family in enumerate(families):
            if family == families[(index + 1) % len(families)]:
                return families[index:] + families[:index]

        rotations = [
            families[index:] + families[:index]
            for index in range(len(families))
        ]
        return min(
            rotations,
            key=lambda rotation: tuple(
                self._base_family_label(family) if family else '~'
                for family in rotation
            ),
        )

    def _pattern_learning_weight(self, decision):
        return 2.0 if decision.status in {'single', 'multi'} else 1.0

    def _apply_harmonic_pattern(self, decisions, pattern, family_labels):
        corrected = list(decisions)
        for index, decision in enumerate(decisions):
            if not self._can_pattern_correct_measure(corrected, index, pattern):
                continue
            expected_now = decision.primary_family
            expected_next = decision.secondary_family
            corrected[index] = MeasureHarmonyDecision(
                measure=decision.measure,
                evidence=decision.evidence,
                eligible=decision.eligible,
                selected_families=(expected_now,),
                duration_winner_family=decision.duration_winner_family,
                entry_family=decision.entry_family,
                primary_family=expected_now,
                pattern_learning_family=expected_now,
                primary_ratio=decision.primary_ratio,
                secondary_family=None,
                secondary_duration_beats=0.0,
                secondary_ratio=0.0,
                secondary_start_beat=None,
                status='single',
                corrected_by_pattern=True,
                pattern_reason=(
                    f'{self._family_label(decision.secondary_family, family_labels)} '
                    f'was treated as anticipation of next slot '
                    f'{self._family_label(expected_next, family_labels)}'
                ),
            )
            logger.info(
                'Pattern correction on measure %s (%s): kept %s and suppressed late %s as anticipation of next measure.',
                decision.measure.number,
                decision.status,
                self._family_label(expected_now, family_labels),
                self._family_label(decision.secondary_family, family_labels),
            )
        return corrected

    def _apply_phase_based_single_corrections(
        self,
        decisions,
        pattern,
        family_labels,
    ):
        corrected = list(decisions)
        thresholds = self.decision_thresholds
        for index, decision in enumerate(decisions):
            if decision.status != 'single' or decision.corrected_by_pattern:
                continue

            expected_now = self._expected_pattern_family(pattern, index)
            if not expected_now or decision.primary_family == expected_now:
                continue

            if pattern.global_agreement < thresholds.phase_min_global_agreement:
                logger.info(
                    'Phase-based single correction blocked on measure %s: global_pattern_confidence.',
                    decision.measure.number,
                )
                continue

            phase_context = self._phase_context_at(decisions, index, pattern)
            has_override_phase_context = self._has_override_phase_context(
                pattern,
                phase_context,
            )
            if not phase_context['is_determined'] and not has_override_phase_context:
                logger.info(
                    'Phase-based single correction blocked on measure %s: phase_context anchors=%s agreement=%.2f margin=%.2f.',
                    decision.measure.number,
                    phase_context['anchor_count'],
                    phase_context['agreement'],
                    phase_context['margin'],
                )
                continue

            strong_local_contradiction = self._has_strong_local_contradiction(
                decision,
                expected_now,
            )
            can_override_strong_local = self._can_override_strong_single(
                decision,
                pattern,
                phase_context,
                expected_now,
            )
            if strong_local_contradiction and not can_override_strong_local:
                logger.info(
                    'Phase-based single correction blocked on measure %s: strong_local_evidence detected=%s expected=%s.',
                    decision.measure.number,
                    self._family_label(decision.primary_family, family_labels),
                    self._family_label(expected_now, family_labels),
                )
                continue

            if (
                not strong_local_contradiction
                and not self._has_expected_local_support(decision, expected_now)
            ):
                logger.info(
                    'Phase-based single correction blocked on measure %s: expected_local_support.',
                    decision.measure.number,
                )
                continue

            if not self._has_phase_anchors(decisions, index, pattern):
                logger.info(
                    'Phase-based single correction blocked on measure %s: neighboring_phase_anchors.',
                    decision.measure.number,
                )
                continue

            corrected_evidence = dict(decision.evidence)
            corrected_evidence.setdefault(
                expected_now,
                {
                    'duration': 0.0,
                    'start': decision.measure.start_beat,
                    'end': decision.measure.end_beat,
                    'source_label': self._family_label(expected_now, family_labels),
                },
            )
            corrected[index] = MeasureHarmonyDecision(
                measure=decision.measure,
                evidence=corrected_evidence,
                eligible=corrected_evidence,
                selected_families=(expected_now,),
                duration_winner_family=decision.duration_winner_family,
                entry_family=decision.entry_family,
                primary_family=expected_now,
                pattern_learning_family=expected_now,
                primary_ratio=(
                    corrected_evidence[expected_now]['duration']
                    / self.time_signature.beats_per_measure
                ),
                secondary_family=None,
                secondary_duration_beats=0.0,
                secondary_ratio=0.0,
                secondary_start_beat=None,
                status='single',
                corrected_by_pattern=True,
                pattern_reason=(
                    'high-confidence phase override'
                    if can_override_strong_local
                    else 'phase-aligned single correction'
                ),
            )
            logger.info(
                'Phase-based single correction on measure %s: %s -> %s phase=%s local_agreement=%.2f anchors=%s override_strong_local=%s.',
                decision.measure.number,
                self._family_label(decision.primary_family, family_labels),
                self._family_label(expected_now, family_labels),
                pattern.phase,
                phase_context['agreement'],
                phase_context['anchor_count'],
                can_override_strong_local,
            )
        return corrected

    def _can_override_strong_single(
        self,
        decision,
        pattern,
        phase_context,
        expected_family,
    ):
        """Permit an isolated full-measure misread only under strong cycle evidence."""
        expected_evidence = decision.evidence.get(expected_family)
        thresholds = self.decision_thresholds
        expected_is_weak = (
            not expected_evidence
            or expected_evidence['duration']
            < thresholds.single_expected_min_support_beats
            or (
                expected_evidence['duration'] / self.time_signature.beats_per_measure
                < thresholds.single_expected_min_support_ratio
            )
        )
        return (
            expected_is_weak
            and pattern.global_agreement >= thresholds.phase_override_min_global_agreement
            and pattern.repetition_count >= thresholds.phase_override_min_repetitions
            and phase_context['agreement'] >= thresholds.phase_override_min_local_agreement
        )

    def _has_override_phase_context(self, pattern, phase_context):
        """Allow a strong long-form pattern to resolve an early isolated misread."""
        thresholds = self.decision_thresholds
        return (
            phase_context['anchor_count'] >= thresholds.phase_min_reliable_anchors
            and phase_context['agreement'] >= thresholds.phase_override_min_local_agreement
            and phase_context['margin'] >= thresholds.phase_min_margin
            and pattern.global_agreement >= thresholds.phase_override_min_global_agreement
            and pattern.repetition_count >= thresholds.phase_override_min_repetitions
        )

    def _has_strong_local_contradiction(self, decision, expected_family):
        primary = decision.evidence.get(decision.primary_family)
        expected = decision.evidence.get(expected_family)
        if not primary:
            return True

        primary_ratio = (
            primary['duration'] / self.time_signature.beats_per_measure
        )
        primary_is_strong = (
            primary['duration']
            >= self.decision_thresholds.single_local_contradiction_beats
            or primary_ratio
            >= self.decision_thresholds.single_local_contradiction_ratio
        )
        expected_is_weak = (
            not expected
            or expected['duration']
            < self.decision_thresholds.single_expected_min_support_beats
            or (
                expected['duration'] / self.time_signature.beats_per_measure
                < self.decision_thresholds.single_expected_min_support_ratio
            )
        )
        return primary_is_strong and expected_is_weak

    def _has_expected_local_support(self, decision, expected_family):
        expected = decision.evidence.get(expected_family)
        if not expected:
            return False
        return (
            expected['duration']
            >= self.decision_thresholds.single_expected_min_support_beats
            and expected['duration'] / self.time_signature.beats_per_measure
            >= self.decision_thresholds.single_expected_min_support_ratio
        )

    def _phase_context_at(self, decisions, index, pattern):
        half_window = self.decision_thresholds.phase_local_window_measures // 2
        start = max(0, index - half_window)
        end = min(len(decisions), index + half_window + 1)
        anchors = [
            (anchor_index, decision.pattern_learning_family)
            for anchor_index, decision in enumerate(decisions[start:end], start)
            if anchor_index != index and self._is_phase_reliable_anchor(decision)
        ]
        if not anchors:
            return {'is_determined': False, 'anchor_count': 0, 'agreement': 0.0, 'margin': 0.0}

        phase_scores = []
        for phase in range(pattern.cycle_length):
            matches = sum(
                1
                for anchor_index, family in anchors
                if family == pattern.expected_primary_families[
                    (anchor_index + phase) % pattern.cycle_length
                ]
            )
            phase_scores.append(matches / len(anchors))

        agreement = phase_scores[pattern.phase]
        second_best = max(
            score
            for phase, score in enumerate(phase_scores)
            if phase != pattern.phase
        )
        margin = agreement - second_best
        thresholds = self.decision_thresholds
        return {
            'is_determined': (
                len(anchors) >= thresholds.phase_min_reliable_anchors
                and agreement >= thresholds.phase_min_local_agreement
                and margin >= thresholds.phase_min_margin
            ),
            'anchor_count': len(anchors),
            'agreement': agreement,
            'margin': margin,
        }

    def _is_phase_reliable_anchor(self, decision):
        return (
            not decision.corrected_by_pattern
            and decision.status in {'single', 'multi'}
            and bool(decision.pattern_learning_family)
        )

    def _has_phase_anchors(self, decisions, index, pattern):
        max_distance = self.decision_thresholds.phase_anchor_max_distance
        previous = self._nearest_matching_phase_anchor(
            decisions,
            index,
            pattern,
            range(index - 1, max(-1, index - max_distance - 1), -1),
        )
        following = self._nearest_matching_phase_anchor(
            decisions,
            index,
            pattern,
            range(index + 1, min(len(decisions), index + max_distance + 1)),
        )
        if not previous or not following:
            return False
        return True

    def _nearest_matching_phase_anchor(self, decisions, index, pattern, candidates):
        for anchor_index in candidates:
            decision = decisions[anchor_index]
            if (
                self._is_phase_reliable_anchor(decision)
                and decision.pattern_learning_family
                == self._expected_pattern_family(pattern, anchor_index)
            ):
                return anchor_index, decision
        return None

    def _can_pattern_correct_measure(
        self,
        decisions,
        index,
        pattern,
        require_active_section=True,
    ):
        decision = decisions[index]
        if decision.status not in {'ambiguous', 'ambiguous_balanced_split'}:
            return False
        if len(decision.selected_families) != 2:
            return False
        if index + 1 >= len(decisions):
            return False
        next_decision = decisions[index + 1]
        current_family = decision.primary_family
        following_family = decision.secondary_family
        if not current_family or not following_family:
            return False
        if (
            not self._pattern_contains_transition(
                pattern,
                current_family,
                following_family,
            )
            or next_decision.primary_family != following_family
        ):
            return False
        if decision.secondary_start_beat is None:
            return False
        if decision.secondary_duration_beats > self.decision_thresholds.pattern_max_ambiguous_beats:
            return False
        if decision.secondary_ratio > self.decision_thresholds.pattern_max_ambiguous_ratio:
            return False
        starts_late = decision.secondary_start_beat >= (
            self.decision_thresholds.pattern_min_secondary_start_beat
        )
        if not starts_late:
            return False
        if require_active_section and not self._pattern_is_active_at(decisions, index, pattern):
            return False
        return True

    def _pattern_is_active_at(self, decisions, index, pattern):
        half_window = self.decision_thresholds.pattern_section_window_measures // 2
        start = max(0, index - half_window)
        end = min(len(decisions), index + half_window + 1)
        observations = []
        for window_index in range(start, end):
            decision = decisions[window_index]
            learning_family = decision.pattern_learning_family
            if not learning_family:
                continue
            observations.append((window_index, learning_family))
        if len(observations) < self.decision_thresholds.pattern_section_min_window_measures:
            return True

        matches = 0
        comparisons = 0
        for (previous_index, previous_family), (
            current_index,
            current_family,
        ) in zip(observations, observations[1:]):
            comparisons += 1
            if self._pattern_contains_transition(
                pattern,
                previous_family,
                current_family,
                steps=current_index - previous_index,
            ):
                matches += 1
        if comparisons == 0:
            return True
        return (
            matches / comparisons
            >= self.decision_thresholds.pattern_section_break_match_rate
        )

    def _pattern_contains_transition(
        self,
        pattern,
        current_family,
        following_family,
        steps=1,
    ):
        expected = pattern.expected_primary_families
        if not expected:
            return False
        return any(
            family == current_family
            and expected[(slot_index + steps) % pattern.cycle_length]
            == following_family
            for slot_index, family in enumerate(expected)
        )

    def _expected_pattern_family(self, pattern, decision_index):
        slot_index = (decision_index + pattern.phase) % pattern.cycle_length
        return pattern.expected_primary_families[slot_index]

    def _family_label(self, family, family_labels):
        if family is None:
            return '?'
        return family_labels.get(family, self._base_family_label(family))


def build_musical_structure(
    chords,
    bpm,
    time_signature=None,
    rhythm_grid=None,
    decision_thresholds=None,
):
    return MusicalStructureBuilder(
        bpm=bpm,
        time_signature=time_signature,
        rhythm_grid=rhythm_grid,
        decision_thresholds=decision_thresholds,
    ).build(chords)
