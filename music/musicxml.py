import math
import re
from xml.etree import ElementTree as ET


MUSICXML_DIVISIONS = 10080
MEASURES_PER_SYSTEM = 4

_KEY_FIFTHS = {
    'Cb': -7, 'Gb': -6, 'Db': -5, 'Ab': -4, 'Eb': -3, 'Bb': -2, 'F': -1,
    'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
    'Abm': -7, 'Ebm': -6, 'Bbm': -5, 'Fm': -4, 'Cm': -3, 'Gm': -2, 'Dm': -1,
    'Am': 0, 'Em': 1, 'Bm': 2, 'F#m': 3, 'C#m': 4, 'G#m': 5, 'D#m': 6, 'A#m': 7,
}

_CHORD_KINDS = {
    '': 'major',
    'maj': 'major',
    'm': 'minor',
    'min': 'minor',
    '7': 'dominant',
    'maj7': 'major-seventh',
    'm7': 'minor-seventh',
    'min7': 'minor-seventh',
    'mM7': 'major-minor',
    'dim': 'diminished',
    'dim7': 'diminished-seventh',
    'm7b5': 'half-diminished',
    'aug': 'augmented',
    'sus2': 'suspended-second',
    'sus4': 'suspended-fourth',
    '6': 'major-sixth',
    'm6': 'minor-sixth',
    '9': 'dominant-ninth',
    'maj9': 'major-ninth',
    'm9': 'minor-ninth',
    '11': 'dominant-11th',
    '13': 'dominant-13th',
}


def _add_text(parent, tag, value, attributes=None):
    element = ET.SubElement(parent, tag, attributes or {})
    element.text = str(value)
    return element


def _accidental_alter(accidental):
    return 1 if accidental == '#' else -1 if accidental == 'b' else 0


def _split_chord_name(chord_name):
    if chord_name == 'N.C.':
        return {'no_chord': True}

    match = re.fullmatch(r'([A-G])([#b]?)(.*)', chord_name)
    if not match:
        return {'no_chord': True}
    root_step, root_accidental, suffix = match.groups()

    bass_step = bass_accidental = None
    bass_match = re.search(r'/([A-G])([#b]?)$', suffix)
    if bass_match:
        bass_step, bass_accidental = bass_match.groups()
        suffix = suffix[:bass_match.start()]

    degree_tokens = []
    parenthetical = re.search(r'\(([^)]+)\)$', suffix)
    if parenthetical:
        degree_tokens.extend(parenthetical.group(1).split(','))
        suffix = suffix[:parenthetical.start()]
    if suffix.endswith('6/9'):
        suffix = suffix[:-3] + '6'
        degree_tokens.append('9')
    add_match = re.search(r'add(\d+)$', suffix)
    if add_match:
        degree_tokens.append(add_match.group(1))
        suffix = suffix[:add_match.start()]

    return {
        'no_chord': False,
        'root_step': root_step,
        'root_alter': _accidental_alter(root_accidental),
        'kind': _CHORD_KINDS.get(suffix, 'other'),
        'kind_text': suffix if suffix and suffix not in _CHORD_KINDS else None,
        'bass_step': bass_step,
        'bass_alter': _accidental_alter(bass_accidental),
        'degrees': degree_tokens,
    }


def _append_harmony(measure_element, chord):
    chord_name = str(chord.get('chord_name') or 'N.C.')
    parsed = _split_chord_name(chord_name)
    harmony = ET.SubElement(measure_element, 'harmony', {'placement': 'above'})
    if parsed['no_chord']:
        root = ET.SubElement(harmony, 'root')
        _add_text(root, 'root-step', 'C')
        _add_text(harmony, 'kind', 'none', {'text': 'N.C.'})
        return

    root = ET.SubElement(harmony, 'root')
    _add_text(root, 'root-step', parsed['root_step'])
    if parsed['root_alter']:
        _add_text(root, 'root-alter', parsed['root_alter'])

    kind_attributes = {'text': parsed['kind_text']} if parsed['kind_text'] else None
    _add_text(harmony, 'kind', parsed['kind'], kind_attributes)

    if parsed['bass_step']:
        bass = ET.SubElement(harmony, 'bass')
        _add_text(bass, 'bass-step', parsed['bass_step'])
        if parsed['bass_alter']:
            _add_text(bass, 'bass-alter', parsed['bass_alter'])

    for token in parsed['degrees']:
        degree_match = re.fullmatch(r'([#b]?)(\d+)', token.strip())
        if not degree_match:
            continue
        accidental, value = degree_match.groups()
        degree = ET.SubElement(harmony, 'degree')
        _add_text(degree, 'degree-value', value)
        _add_text(degree, 'degree-alter', _accidental_alter(accidental))
        _add_text(degree, 'degree-type', 'add' if not accidental else 'alter')


def _beat_position(chord, beats_per_measure):
    position_ratio = chord.get('_manual_position_ratio')
    if isinstance(position_ratio, (int, float)) and not isinstance(position_ratio, bool):
        position_ratio = float(position_ratio)
        if math.isfinite(position_ratio):
            return max(0.0, min(1.0, position_ratio)) * beats_per_measure
    try:
        beat_start = float(chord.get('beat_start') or 0)
    except (TypeError, ValueError):
        beat_start = 0.0
    return max(0.0, min(float(beats_per_measure), beat_start))


def _note_type_for_beat_type(beat_type):
    return {
        1: 'whole',
        2: 'half',
        4: 'quarter',
        8: 'eighth',
        16: '16th',
        32: '32nd',
    }.get(beat_type, 'quarter')


def _append_rhythmic_slashes(measure_element, beats, beat_type):
    """Write one slash-note per beat instead of a visible measure rest."""
    beat_ticks = int(round((4 / beat_type) * MUSICXML_DIVISIONS))
    note_type = _note_type_for_beat_type(beat_type)
    for _ in range(beats):
        note = ET.SubElement(measure_element, 'note')
        unpitched = ET.SubElement(note, 'unpitched')
        _add_text(unpitched, 'display-step', 'B')
        _add_text(unpitched, 'display-octave', 4)
        _add_text(note, 'duration', beat_ticks)
        _add_text(note, 'voice', 1)
        _add_text(note, 'type', note_type)
        _add_text(note, 'stem', 'none')
        _add_text(note, 'notehead', 'slash')


def _append_measure(part, measure, measure_index, beats, beat_type, key_name, bpm):
    measure_element = ET.SubElement(part, 'measure', {'number': str(measure.get('number') or measure_index)})
    quarter_beats_per_measure = beats * (4 / beat_type)
    measure_ticks = int(round(quarter_beats_per_measure * MUSICXML_DIVISIONS))

    if measure_index == 1:
        initial_print = ET.SubElement(measure_element, 'print')
        _add_text(initial_print, 'measure-numbering', 'none')

        attributes = ET.SubElement(measure_element, 'attributes')
        _add_text(attributes, 'divisions', MUSICXML_DIVISIONS)
        key = ET.SubElement(attributes, 'key')
        _add_text(key, 'fifths', _KEY_FIFTHS.get(key_name, 0))
        _add_text(key, 'mode', 'minor' if key_name.endswith('m') else 'major')
        time = ET.SubElement(attributes, 'time')
        _add_text(time, 'beats', beats)
        _add_text(time, 'beat-type', beat_type)
        clef = ET.SubElement(attributes, 'clef')
        _add_text(clef, 'sign', 'G')
        _add_text(clef, 'line', 2)

        direction = ET.SubElement(measure_element, 'direction', {'placement': 'above'})
        direction_type = ET.SubElement(direction, 'direction-type')
        metronome = ET.SubElement(direction_type, 'metronome')
        _add_text(metronome, 'beat-unit', 'quarter')
        _add_text(metronome, 'per-minute', bpm)
        ET.SubElement(direction, 'sound', {'tempo': str(bpm)})

    if measure_index > 1 and (measure_index - 1) % MEASURES_PER_SYSTEM == 0:
        print_element = ET.SubElement(measure_element, 'print', {'new-system': 'yes'})
        system_layout = ET.SubElement(print_element, 'system-layout')
        _add_text(system_layout, 'system-distance', 140)
        _add_text(print_element, 'measure-numbering', 'none')

    cursor_ticks = 0
    chords = sorted(
        measure.get('chords') or [],
        key=lambda chord: _beat_position(chord, beats),
    )
    for chord in chords:
        beat_position = _beat_position(chord, beats)
        quarter_position = beat_position * (4 / beat_type)
        chord_ticks = max(0, min(measure_ticks, int(round(quarter_position * MUSICXML_DIVISIONS))))
        if chord_ticks > cursor_ticks:
            forward = ET.SubElement(measure_element, 'forward')
            _add_text(forward, 'duration', chord_ticks - cursor_ticks)
            cursor_ticks = chord_ticks
        _append_harmony(measure_element, chord)

    if cursor_ticks:
        backup = ET.SubElement(measure_element, 'backup')
        _add_text(backup, 'duration', cursor_ticks)

    _append_rhythmic_slashes(measure_element, beats, beat_type)


def render_musicxml(score_data):
    """Build an editable MusicXML 3.1 document from overlaid score data."""
    title = str(score_data.get('title') or 'Harmix Score')
    artist = str(score_data.get('artist') or 'Harmix')
    key_name = str(score_data.get('key') or 'C')
    bpm = score_data.get('bpm') or 120
    bpm = int(bpm) if isinstance(bpm, float) and bpm.is_integer() else bpm
    time_signature = score_data.get('structure', {}).get('time_signature', {}) or {}
    beats = int(time_signature.get('beats_per_measure') or 4)
    beat_type = int(time_signature.get('beat_unit') or 4)

    root = ET.Element('score-partwise', {'version': '3.1'})
    work = ET.SubElement(root, 'work')
    _add_text(work, 'work-title', title)
    _add_text(root, 'movement-title', title)
    identification = ET.SubElement(root, 'identification')
    _add_text(identification, 'creator', artist, {'type': 'composer'})
    encoding = ET.SubElement(identification, 'encoding')
    _add_text(encoding, 'software', 'Harmix')
    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', {'id': 'P1'})
    _add_text(score_part, 'part-name', 'Harmix Lead Sheet')
    part = ET.SubElement(root, 'part', {'id': 'P1'})

    measures = score_data.get('structure', {}).get('measures') or []
    if not measures:
        measures = [
            {
                'number': index,
                'chords': [{'chord_name': chord.get('chord_name'), 'beat_start': 0}],
            }
            for index, chord in enumerate(score_data.get('chords') or [], start=1)
        ]
    for measure_index, measure in enumerate(measures, start=1):
        _append_measure(part, measure, measure_index, beats, beat_type, key_name, bpm)

    body = ET.tostring(root, encoding='utf-8').decode('utf-8')
    doctype = (
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">'
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{doctype}\n{body}\n'
