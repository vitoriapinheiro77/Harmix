import numpy as np
import gc
from statistics import median

from .meter_detection import detect_meter
from .types import RhythmGrid


def detect_beats(audio_path, rhythm: RhythmGrid | None = None, bpm=None):
    rhythm = rhythm or RhythmGrid()
    if bpm:
        rhythm.bpm = bpm
    try:
        return _detect_beats_madmom(audio_path, rhythm)
    except Exception as exc:
        rhythm.errors.append(f'madmom_beats: {exc}')
    return _detect_beats_fallback(audio_path, rhythm)


def _detect_beats_madmom(audio_path, rhythm: RhythmGrid):
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    downbeat_activations = RNNDownBeatProcessor()(str(audio_path))
    # The selected DBN path already contains every metric subdivision. Running
    # RNNBeatProcessor as well consumed substantial memory and then its output
    # was overwritten below. Evaluate each meter sequentially to bound the
    # peak for long recordings.
    paths = {}
    for meter in (4, 6):
        tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[meter], fps=100)
        paths[meter] = tracker(downbeat_activations)
        del tracker
        gc.collect()
    rhythm.meter = detect_meter(paths, downbeat_activations, fps=100)
    rhythm.beats_per_measure = rhythm.meter['beats_per_measure']
    rhythm.beat_unit = rhythm.meter['beat_unit']
    selected_path = np.atleast_2d(paths[rhythm.beats_per_measure])
    metric_beats = [float(row[0]) for row in selected_path if len(row) >= 2]
    downbeat_times = [
        float(row[0])
        for row in selected_path
        if len(row) >= 2 and int(row[1]) == 1
    ]
    rhythm.beats = metric_beats or rhythm.beats
    rhythm.downbeats = downbeat_times or _infer_downbeats(rhythm.beats, rhythm.beats_per_measure)
    rhythm.source = 'madmom'
    return rhythm


def _detect_beats_fallback(audio_path, rhythm: RhythmGrid):
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    rhythm.beats = [float(value) for value in librosa.frames_to_time(beat_frames, sr=sr)]
    rhythm.downbeats = _infer_downbeats(rhythm.beats, rhythm.beats_per_measure)
    rhythm.meter = {
        'beats_per_measure': 4,
        'beat_unit': 4,
        'label': '4/4',
        'confidence': 0.0,
        'low_confidence': True,
        'decision': 'fallback_without_metric_activations',
        'hypotheses': {},
    }
    rhythm.source = 'librosa' if rhythm.source == 'librosa' else rhythm.source
    if rhythm.source not in ('madmom', 'librosa'):
        rhythm.source = 'librosa'
    return rhythm


def _infer_downbeats(beats, beats_per_measure):
    if not beats:
        return []
    return [beats[index] for index in range(0, len(beats), beats_per_measure)]


def snap_time_to_grid(time_value, grid_times):
    if not grid_times:
        return time_value
    grid = np.asarray(grid_times, dtype=float)
    index = int(np.argmin(np.abs(grid - float(time_value))))
    return float(grid[index])


def quantize_detections(detections, rhythm: RhythmGrid, merge_fn):
    if not detections:
        return []

    beat_grid = rhythm.beats or []
    downbeat_grid = rhythm.downbeats or []
    if not beat_grid and rhythm.bpm and rhythm.bpm > 0:
        beat_sec = 60.0 / rhythm.bpm
        max_end = max(det['end'] for det in detections)
        beat_grid = [round(index * beat_sec, 3) for index in range(int(max_end / beat_sec) + 2)]

    beat_intervals = [
        right - left
        for left, right in zip(beat_grid, beat_grid[1:])
        if right > left
    ]
    beat_sec = median(beat_intervals) if beat_intervals else (
        60.0 / rhythm.bpm if rhythm.bpm and rhythm.bpm > 0 else None
    )
    # A downbeat is preferred only when the change is already close to it.
    # The old 35%-of-a-measure window could move a change by more than one beat.
    downbeat_snap_threshold = beat_sec * 0.45 if beat_sec else 0.25

    def snap_start(t):
        value = float(t)
        if downbeat_grid:
            nearest_downbeat = snap_time_to_grid(value, downbeat_grid)
            if abs(nearest_downbeat - value) <= downbeat_snap_threshold:
                return round(nearest_downbeat, 3)
        if beat_grid:
            return round(snap_time_to_grid(value, beat_grid), 3)
        return round(value, 3)

    def snap_end(t):
        if not beat_grid:
            return round(float(t), 3)
        return round(snap_time_to_grid(float(t), beat_grid), 3)

    quantized = []
    for det in detections:
        q_start = snap_start(det['start'])
        q_end = snap_end(det['end'])
        if q_end <= q_start and beat_grid:
            next_index = min(
                range(len(beat_grid)),
                key=lambda idx: abs(beat_grid[idx] - (q_start + 0.01)),
            )
            q_end = beat_grid[min(next_index + 1, len(beat_grid) - 1)]
        if q_end <= q_start:
            q_end = q_start + (60.0 / rhythm.bpm if rhythm.bpm else 0.5)
        quantized.append({
            'start': q_start,
            'end': q_end,
            'chord': det['chord'],
            'confidence': det.get('confidence', 0.0),
        })

    merged = merge_fn(quantized)
    if downbeat_grid:
        for item in merged:
            item['aligned_to_downbeats'] = True
    return merged
