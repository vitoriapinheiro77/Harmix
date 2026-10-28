"""Convert LV-Chordia Harte labels to Harmix chord symbols."""

NOTE_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
HARTE_INTERVALS = {
    '1': 0, 'b2': 1, '2': 2, 'b3': 3, '3': 4, '4': 5, '#4': 6,
    'b5': 6, '5': 7, '#5': 8, 'b6': 8, '6': 9, 'bb7': 9, 'b7': 10, '7': 11,
}
QUALITY_MAP = {
    'maj': '', 'min': 'm', '7': '7', 'maj7': 'maj7', 'min7': 'm7',
    'dim': 'dim', 'dim7': 'dim7', 'hdim7': 'm7b5', 'aug': 'aug',
    'sus2': 'sus2', 'sus4': 'sus4', '6': '6', 'min6': 'm6',
}


def _transpose(root, semitones):
    aliases = {
        'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E',
        'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    }
    normalized_root = aliases.get(root, root)
    return NOTE_NAMES[(NOTE_NAMES.index(normalized_root) + semitones) % 12]


def harte_to_conventional(label):
    """Convert labels such as ``G:maj/3`` to Harmix's ``G/B`` form."""
    raw = (label or '').strip()
    if not raw or raw.upper() in {'N', 'X', 'NC', 'N.C.'}:
        return None
    if ':' not in raw:
        return raw.replace(' ', '')

    root, description = raw.split(':', 1)
    root = root.strip()[0].upper() + root.strip()[1:]
    quality, separator, bass = description.partition('/')
    suffix = QUALITY_MAP.get(quality.strip().lower(), quality.strip())
    bass_note = None
    if separator and bass.strip():
        bass = bass.strip()
        bass_note = _transpose(root, HARTE_INTERVALS[bass]) if bass in HARTE_INTERVALS else bass
    return f'{root}{suffix}' + (f'/{bass_note}' if bass_note and bass_note != root else '')
