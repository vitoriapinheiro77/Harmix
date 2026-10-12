import logging
from contextlib import nullcontext

from .chordino_labels import chord_changes_to_events
from .types import ChordAnalysisResult, ChordEvent

logger = logging.getLogger('harmix.analysis')


def _stage(diagnostics, name, details=None):
    return diagnostics.stage(name, details=details) if diagnostics else nullcontext({})


def detect_chords(audio_path, duration, fallback_fn, diagnostics=None):
    result = ChordAnalysisResult()
    try:
        with _stage(diagnostics, 'chordino_detection') as metrics:
            result = _detect_chords_chordino(audio_path, duration)
            metrics['total_chords'] = len(result.chords)
        if result.chords:
            result.source = 'chordino'
            return result
        result.errors.append('chordino: nenhum acorde valido retornado')
    except Exception as exc:
        logger.warning('Chordino indisponivel, usando fallback librosa: %s', exc)
        result.errors.append(f'chordino: {exc}')

    try:
        with _stage(diagnostics, 'librosa_chord_fallback') as metrics:
            fallback_events = fallback_fn()
            metrics['total_chords'] = len(fallback_events)
        result.chords = [
            ChordEvent(
                start_time=item['start_time'],
                end_time=item['end_time'],
                chord_name=item['chord_name'],
                confidence=item.get('confidence', 0.0),
                source='librosa',
            )
            for item in fallback_events
        ]
        result.source = 'librosa'
    except Exception as exc:
        result.errors.append(f'librosa_fallback: {exc}')
        raise

    return result


def _detect_chords_chordino(audio_path, duration):
    from chord_extractor.extractors import Chordino

    extractor = Chordino(roll_on=1)
    changes = extractor.extract(str(audio_path), sr=22050, mono=True)
    events = chord_changes_to_events(changes, duration, source='chordino')
    return ChordAnalysisResult(chords=events, source='chordino')
