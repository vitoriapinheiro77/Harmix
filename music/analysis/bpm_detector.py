from .types import RhythmGrid


def detect_bpm(audio_path, rhythm: RhythmGrid | None = None):
    rhythm = rhythm or RhythmGrid()
    try:
        return _detect_bpm_madmom(audio_path, rhythm)
    except Exception as exc:
        rhythm.errors.append(f'madmom_bpm: {exc}')
    try:
        return _detect_bpm_librosa(audio_path, rhythm)
    except Exception as exc:
        rhythm.errors.append(f'librosa_bpm: {exc}')
        rhythm.source = 'none'
        return None, rhythm


def _madmom_primary_bpm(tempo_candidates):
    """Extrai o BPM principal do retorno do TempoEstimationProcessor."""
    import numpy as np

    if tempo_candidates is None or len(tempo_candidates) == 0:
        raise RuntimeError('Madmom nao retornou BPM valido.')

    # API madmom-modern: ndarray (n, 2) com [bpm, forca_relativa] por linha.
    candidates = np.atleast_2d(np.asarray(tempo_candidates, dtype=float))
    bpm = float(candidates[0, 0])
    if bpm <= 0:
        raise RuntimeError('Madmom nao retornou BPM valido.')
    return bpm


def _detect_bpm_madmom(audio_path, rhythm: RhythmGrid):
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.tempo import TempoEstimationProcessor

    activations = RNNBeatProcessor()(str(audio_path))
    tempo = TempoEstimationProcessor(fps=100)(activations)
    bpm = _madmom_primary_bpm(tempo)

    rhythm.bpm = bpm
    rhythm.source = 'madmom'
    return bpm, rhythm


def _detect_bpm_librosa(audio_path, rhythm: RhythmGrid):
    import librosa
    import numpy as np

    y, _sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=120)
    tempo, _beats = librosa.beat.beat_track(y=y, sr=22050)
    bpm = float(np.atleast_1d(tempo)[0])
    if not bpm or bpm <= 0:
        raise RuntimeError('Librosa nao retornou BPM valido.')

    rhythm.bpm = bpm
    rhythm.source = 'librosa'
    return bpm, rhythm
