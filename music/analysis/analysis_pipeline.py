import time
from pathlib import Path

from . import logging_utils
from .beat_detector import detect_beats
from .bpm_detector import detect_bpm
from .chord_detector import detect_chords
from .types import MusicalAnalysisResult, RhythmGrid


def run_rhythm_analysis(audio_path: Path) -> tuple[float | None, RhythmGrid]:
    logging_utils.log_analysis_start(audio_path)
    rhythm = RhythmGrid()
    bpm, rhythm = detect_bpm(audio_path, rhythm)
    logging_utils.log_bpm_result(bpm, rhythm.source, rhythm.errors)
    rhythm = detect_beats(audio_path, rhythm, bpm=bpm)
    logging_utils.log_beat_result(rhythm)
    return bpm, rhythm


def run_chord_analysis(audio_path: Path, duration: float, fallback_fn) -> MusicalAnalysisResult:
    started = time.perf_counter()
    chord_result = detect_chords(audio_path, duration, fallback_fn)
    logging_utils.log_chord_result(chord_result)
    result = MusicalAnalysisResult(
        duration=duration,
        chords=chord_result,
        processing_seconds=time.perf_counter() - started,
    )
    logging_utils.log_analysis_complete(result.processing_seconds, result.errors)
    return result
