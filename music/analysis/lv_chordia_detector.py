"""LV-Chordia adapter for the Harmix analysis pipeline."""

from .lv_chordia_labels import harte_to_conventional
from .types import ChordAnalysisResult, ChordEvent


def detect_lv_chordia(audio_path, duration):
    """Return validated LV-Chordia segments without detector fallbacks."""
    from lv_chordia import chord_recognition

    try:
        changes = chord_recognition(str(audio_path), device='cpu')
    except TypeError as exc:
        if 'device' not in str(exc):
            raise
        changes = chord_recognition(str(audio_path))

    events = []
    for change in changes:
        chord_name = harte_to_conventional(change.get('chord', change.get('label')))
        start = change.get('start_time', change.get('start'))
        end = change.get('end_time', change.get('end'))
        if not chord_name or start is None or end is None:
            continue

        start = max(0.0, float(start))
        end = min(float(duration), float(end)) if duration > 0 else float(end)
        if end <= start:
            continue
        events.append(ChordEvent(
            start_time=round(start, 2),
            end_time=round(end, 2),
            chord_name=chord_name,
            confidence=0.0,
            source='lv_chordia',
        ))

    events.sort(key=lambda event: (event.start_time, event.end_time))
    return ChordAnalysisResult(chords=events, source='lv_chordia')
