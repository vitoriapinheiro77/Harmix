"""Compare an independent Beat This! rhythm grid with the score grid."""

from bisect import bisect_left
from collections import Counter
from statistics import median


def _alignment(reference, predicted):
    if not reference or not predicted:
        return None
    distances = []
    for value in reference:
        index = bisect_left(predicted, value)
        neighbors = predicted[max(0, index - 1):index + 1]
        distances.append(min(abs(value - neighbor) for neighbor in neighbors))
    return round(median(distances), 4)


def compare_beat_this(audio_path, rhythm):
    from beat_this.inference import File2Beats

    tracker = File2Beats(checkpoint_path='small0', device='cpu', dbn=False)
    beats, downbeats = tracker(str(audio_path))
    beats = sorted(set(map(float, beats)))
    downbeats = sorted(set(map(float, downbeats)))
    boundaries = [bisect_left(beats, time - 1e-6) for time in downbeats]
    pulses_per_bar = Counter(
        right - left for left, right in zip(boundaries, boundaries[1:]) if right > left
    )
    intervals = [right - left for left, right in zip(beats, beats[1:])]
    return {
        'model': 'small0',
        'reference_source': rhythm.source,
        'pulse_bpm': round(60 / median(intervals), 2) if intervals else None,
        'beat_count': len(beats),
        'downbeat_count': len(downbeats),
        'beats_sample': beats[:16],
        'downbeats_sample': downbeats[:16],
        'pulses_per_bar': {str(count): bars for count, bars in sorted(pulses_per_bar.items())},
        'beat_median_difference_seconds': _alignment(rhythm.beats, beats),
        'downbeat_median_difference_seconds': _alignment(rhythm.downbeats, downbeats),
    }
