import time
from contextlib import nullcontext
from pathlib import Path
from statistics import median

from . import logging_utils
from .beat_detector import detect_beats
from .beat_this_tracker import compare_beat_this
from .bpm_detector import detect_bpm
from .chord_detector import detect_chords
from .rhythm_validation import validate_rhythm_grid
from .types import MusicalAnalysisResult, RhythmGrid


def _stage(diagnostics, name, details=None):
    return diagnostics.stage(name, details=details) if diagnostics else nullcontext({})


def _interval_summary(values):
    intervals = [right - left for left, right in zip(values, values[1:]) if right > left]
    if not intervals:
        return {'count': 0}
    return {
        'count': len(intervals),
        'minimum_seconds': round(min(intervals), 4),
        'median_seconds': round(median(intervals), 4),
        'maximum_seconds': round(max(intervals), 4),
    }


def run_rhythm_analysis(audio_path: Path, diagnostics=None) -> tuple[float | None, RhythmGrid]:
    logging_utils.log_analysis_start(audio_path)
    rhythm = RhythmGrid()
    with _stage(diagnostics, 'bpm_detection', {'audio_file': audio_path.name}) as metrics:
        bpm, rhythm = detect_bpm(audio_path, rhythm)
        metrics.update({
            'bpm': bpm,
            'source': rhythm.source,
            'errors': list(rhythm.errors),
        })
    logging_utils.log_bpm_result(bpm, rhythm.source, rhythm.errors)
    with _stage(diagnostics, 'beat_downbeat_detection', {'bpm_input': bpm}) as metrics:
        rhythm = detect_beats(audio_path, rhythm, bpm=bpm)
        metrics.update({
            'source': rhythm.source,
            'total_beats': len(rhythm.beats),
            'total_downbeats': len(rhythm.downbeats),
            'beats_per_measure': rhythm.beats_per_measure,
            'beat_unit': rhythm.beat_unit,
            'time_signature': f'{rhythm.beats_per_measure}/{rhythm.beat_unit}',
            'meter': rhythm.meter,
            'beat_intervals': _interval_summary(rhythm.beats),
            'downbeat_intervals': _interval_summary(rhythm.downbeats),
            'beats_sample': rhythm.beats[:12],
            'downbeats_sample': rhythm.downbeats[:12],
            'errors': list(rhythm.errors),
        })
    logging_utils.log_beat_result(rhythm)
    with _stage(diagnostics, 'rhythm_grid_validation') as metrics:
        rhythm = validate_rhythm_grid(rhythm)
        metrics.update(rhythm.validation)
    if diagnostics:
        with _stage(diagnostics, 'beat_this_comparison') as metrics:
            try:
                metrics.update(compare_beat_this(audio_path, rhythm))
            except Exception as exc:
                metrics['comparison_error'] = str(exc)
    return bpm, rhythm


def run_chord_analysis(
    audio_path: Path,
    duration: float,
    diagnostics=None,
) -> MusicalAnalysisResult:
    started = time.perf_counter()
    with _stage(diagnostics, 'chord_detection', {'duration_seconds': duration}) as metrics:
        chord_result = detect_chords(
            audio_path,
            duration,
            diagnostics=diagnostics,
        )
        metrics.update({
            'source': chord_result.source,
            'total_chords': len(chord_result.chords),
            'errors': list(chord_result.errors),
            'sample': [item.as_segment() for item in chord_result.chords[:10]],
        })
    logging_utils.log_chord_result(chord_result)
    result = MusicalAnalysisResult(
        duration=duration,
        chords=chord_result,
        processing_seconds=time.perf_counter() - started,
    )
    logging_utils.log_analysis_complete(result.processing_seconds, result.errors)
    return result
