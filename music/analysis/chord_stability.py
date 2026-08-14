import re


_CHORD_ROOT_RE = re.compile(r'^([A-G](?:#|b)?)(.*)$')


def harmonic_family(chord_name):
    """
    Return the stable harmonic identity used to compare nearby detections.

    Extensions, suspensions and inversions keep the same family as their
    underlying triad; minor, diminished and augmented qualities remain
    distinct because they change the harmonic function.
    """
    base = str(chord_name or '').strip().split('/', 1)[0]
    match = _CHORD_ROOT_RE.match(base)
    if not match:
        return base, ''

    root, suffix = match.groups()
    quality = suffix.lower()
    if quality.startswith('maj'):
        family = 'major'
    elif quality.startswith('m') and not quality.startswith('maj'):
        family = 'minor'
    elif quality.startswith('dim') or quality.startswith('hdim'):
        family = 'diminished'
    elif quality.startswith('aug'):
        family = 'augmented'
    else:
        # major, dominant, 6, sus2/sus4 and added tensions
        family = 'major'
    return root, family


def stabilize_chord_detections(detections, bpm=None):
    """
    Suppress brief harmonic flicker while preserving sustained complex chords.

    A short label is replaced only when its surrounding context supports the
    same harmony: either it is a temporary variant of an adjacent chord family
    or it interrupts the same family on both sides. Long-lived extensions and
    inversions are therefore retained as real harmonic events.
    """
    ordered = sorted(
        (
            {
                'start': float(item['start']),
                'end': float(item['end']),
                'chord': str(item['chord']).strip(),
                'confidence': float(item.get('confidence', 0.0)),
            }
            for item in detections
            if item.get('chord') and float(item['end']) > float(item['start'])
        ),
        key=lambda item: (item['start'], item['end']),
    )
    if not ordered:
        return []

    beat_seconds = 60.0 / float(bpm) if bpm and bpm > 0 else 0.5
    # Shorter than a beat: enough to remove spectral flicker without deleting
    # a stable one-beat harmonic change.
    stability_seconds = max(0.25, beat_seconds * 0.75)
    labels = [item['chord'] for item in ordered]

    for index, item in enumerate(ordered):
        duration = item['end'] - item['start']
        if duration > stability_seconds:
            continue

        previous = labels[index - 1] if index > 0 else None
        following = labels[index + 1] if index + 1 < len(labels) else None
        current_family = harmonic_family(labels[index])
        previous_family = harmonic_family(previous) if previous else None
        following_family = harmonic_family(following) if following else None

        if previous and following and previous_family == following_family:
            labels[index] = _stronger_neighbor_label(
                ordered[index - 1],
                previous,
                ordered[index + 1],
                following,
            )
        elif previous and current_family == previous_family:
            labels[index] = previous
        elif following and current_family == following_family:
            labels[index] = following

    stabilized = []
    for item, label in zip(ordered, labels):
        candidate = {**item, 'chord': label}
        if (
            stabilized
            and stabilized[-1]['chord'] == label
            and candidate['start'] <= stabilized[-1]['end'] + 0.001
        ):
            stabilized[-1]['end'] = max(stabilized[-1]['end'], candidate['end'])
            stabilized[-1]['confidence'] = max(
                stabilized[-1]['confidence'],
                candidate['confidence'],
            )
        else:
            stabilized.append(candidate)

    return stabilized


def _stronger_neighbor_label(previous_item, previous_label, next_item, next_label):
    previous_duration = previous_item['end'] - previous_item['start']
    next_duration = next_item['end'] - next_item['start']
    if next_duration > previous_duration:
        return next_label
    return previous_label
