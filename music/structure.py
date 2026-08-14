from bisect import bisect_right
from dataclasses import dataclass
from math import ceil, floor

from .analysis.chord_stability import harmonic_family


DEFAULT_BEATS_PER_MEASURE = 4
DEFAULT_BEAT_UNIT = 4
MIN_SECONDARY_CHORD_BEATS = 1.5
MIN_SECONDARY_CHORD_RATIO = 0.35
COMPLEX_CHORD_CONSENSUS = 0.75


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
    def __init__(self, bpm, time_signature=None, rhythm_grid=None):
        if not bpm or bpm <= 0:
            raise ValueError('BPM valido e necessario para converter segundos em compassos.')
        self.bpm = float(bpm)
        self.time_signature = time_signature or TimeSignature()
        self.rhythm_grid = rhythm_grid or {}
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

        measures = [
            self._summarize_measure_harmony(measure, family_labels)
            for measure in measures
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

    def _summarize_measure_harmony(self, measure, family_labels):
        """Keep predominant harmony, plus a stable secondary half-bar change."""
        if not measure.chords:
            return measure

        durations = {}
        starts = {}
        ends = {}
        source_labels = {}
        for placement in measure.chords:
            family = harmonic_family(placement.chord_name)
            durations[family] = durations.get(family, 0.0) + placement.beat_duration
            starts[family] = min(
                starts.get(family, placement.absolute_start_beat),
                placement.absolute_start_beat,
            )
            ends[family] = max(
                ends.get(family, placement.absolute_end_beat),
                placement.absolute_end_beat,
            )
            source_labels.setdefault(family, placement.chord_name)

        predominant = max(durations, key=durations.get)
        selected = [predominant]
        for family, duration in durations.items():
            if family == predominant:
                continue
            if (
                duration >= MIN_SECONDARY_CHORD_BEATS
                and duration / self.time_signature.beats_per_measure
                >= MIN_SECONDARY_CHORD_RATIO
            ):
                selected.append(family)
        selected.sort(key=lambda family: starts[family])

        placements = []
        for index, family in enumerate(selected):
            beat_start = (
                0.0
                if len(selected) == 1
                else max(0.0, starts[family] - measure.start_beat)
            )
            next_start = (
                max(beat_start, starts[selected[index + 1]] - measure.start_beat)
                if index + 1 < len(selected)
                else self.time_signature.beats_per_measure
            )
            beat_duration = max(0.0, next_start - beat_start)
            if beat_duration <= 0:
                continue
            placements.append(
                ChordPlacement(
                    chord_name=family_labels.get(family, source_labels[family]),
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


def build_musical_structure(chords, bpm, time_signature=None, rhythm_grid=None):
    return MusicalStructureBuilder(
        bpm=bpm,
        time_signature=time_signature,
        rhythm_grid=rhythm_grid,
    ).build(chords)
