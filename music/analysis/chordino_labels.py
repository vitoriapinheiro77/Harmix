import re

CHORDINO_QUALITY_MAP = {
    'maj': '',
    'major': '',
    'min': 'm',
    'minor': 'm',
    'maj7': 'maj7',
    'major7': 'maj7',
    'min7': 'm7',
    'minor7': 'm7',
    '7': '7',
    'dim': 'dim',
    'dim7': 'dim7',
    'aug': 'aug',
    'sus4': 'sus4',
    'sus2': 'sus2',
    'm7b5': 'm7b5',
    'hdim7': 'm7b5',
    '6': '6',
    'min6': 'm6',
    'minor6': 'm6',
}


def normalize_chordino_label(label):
    raw = (label or '').strip()
    if not raw or raw.upper() in {'N', 'X', 'NC', 'N:C', 'NONE'}:
        return None

    if ':' not in raw:
        return raw.replace(' ', '')

    root, remainder = raw.split(':', 1)
    root = root.strip()
    bass = ''
    if '/' in remainder:
        quality, bass_note = remainder.split('/', 1)
        bass = f'/{_normalize_bass(bass_note.strip())}'
    else:
        quality = remainder.strip()

    quality_key = quality.lower()
    suffix = CHORDINO_QUALITY_MAP.get(quality_key, quality)
    return f'{root}{suffix}{bass}'


def _normalize_bass(value):
    if not value:
        return value
    if value.isdigit():
        note_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        index = int(value) % 12
        return note_map[index]
    return value.replace('b', 'b').replace(' ', '')


def _read_chord_change(change):
    """Lê timestamp e label de ChordChange (NamedTuple) ou dict legado."""
    if isinstance(change, dict):
        timestamp = change.get('timestamp', 0)
        label = change.get('chord', change.get('label', ''))
    else:
        timestamp = change.timestamp
        label = change.chord
    return float(timestamp), label


def chord_changes_to_events(changes, duration, source='chordino'):
    from .types import ChordEvent

    timeline = []
    for change in changes:
        timestamp, label = _read_chord_change(change)
        timeline.append((timestamp, normalize_chordino_label(label)))

    if not timeline:
        return []

    events = []
    for index, (start, chord_name) in enumerate(timeline):
        # Keep no-chord entries in the timeline so they terminate the previous
        # harmony instead of making it spill over silence or non-harmonic audio.
        end = timeline[index + 1][0] if index + 1 < len(timeline) else duration
        if not chord_name or end <= start:
            continue
        events.append(
            ChordEvent(
                start_time=round(start, 2),
                end_time=round(end, 2),
                chord_name=chord_name,
                confidence=0.85,
                source=source,
            )
        )
    return events
