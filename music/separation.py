import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEMUCS_STEMS = ('vocals', 'drums', 'bass', 'other')
DEFAULT_DEMUCS_MODEL = 'htdemucs'
DEFAULT_DEMUCS_SEGMENT_SECONDS = 6.0
DEFAULT_DEMUCS_JOBS = 1
DEFAULT_DEMUCS_SHIFTS = 0


class SeparationError(RuntimeError):
    """Raised when the external four-stem separator cannot complete."""


def run_demucs(audio_path, output_dir):
    """Run Demucs in CPU mode and return the generated four stem paths."""
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    model = os.environ.get('HARMIX_DEMUCS_MODEL', DEFAULT_DEMUCS_MODEL)
    segment = os.environ.get(
        'HARMIX_DEMUCS_SEGMENT',
        str(DEFAULT_DEMUCS_SEGMENT_SECONDS),
    )
    jobs = os.environ.get('HARMIX_DEMUCS_JOBS', str(DEFAULT_DEMUCS_JOBS))
    shifts = os.environ.get('HARMIX_DEMUCS_SHIFTS', str(DEFAULT_DEMUCS_SHIFTS))

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        '-m',
        'demucs',
        '-n',
        model,
        '--device',
        'cpu',
        '--segment',
        str(segment),
        '--shifts',
        str(shifts),
        '--jobs',
        str(jobs),
        '-o',
        str(output_dir),
        str(audio_path),
    ]
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')

    logger.info(
        'Iniciando Demucs (%s, segment=%ss, shifts=%s, jobs=%s) para %s',
        model,
        segment,
        shifts,
        jobs,
        audio_path.name,
    )
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3300,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        detail = _subprocess_detail(exc.stderr, exc.stdout)
        raise SeparationError(
            f'Demucs excedeu o tempo limite ({detail})'
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = _subprocess_detail(exc.stderr, exc.stdout)
        raise SeparationError(f'Demucs falhou: {detail}') from exc
    except OSError as exc:
        raise SeparationError(f'Demucs falhou ao iniciar: {exc}') from exc

    stem_dir = _resolve_stem_dir(output_dir, model, audio_path)
    stems = {name: stem_dir / f'{name}.wav' for name in DEMUCS_STEMS}
    missing = [name for name, path in stems.items() if not path.exists()]
    if missing:
        detail = _subprocess_detail(completed.stderr, completed.stdout)
        raise SeparationError(
            f'Demucs nao gerou os stems esperados ({", ".join(missing)}). {detail}'
        )

    logger.info('Demucs concluiu com sucesso: %s', ', '.join(stems))
    return stems


def _resolve_stem_dir(output_dir, model, audio_path):
    """Locate Demucs output even when the track folder name differs slightly."""
    expected = output_dir / model / audio_path.stem
    if expected.is_dir():
        return expected

    model_dir = output_dir / model
    if model_dir.is_dir():
        candidates = sorted(
            (path for path in model_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if all((candidate / f'{stem}.wav').exists() for stem in DEMUCS_STEMS):
                return candidate
        if len(candidates) == 1:
            return candidates[0]

    raise SeparationError(
        f'Demucs nao criou a pasta de stems esperada em {expected}. '
        f'Conteudo em {model_dir}: '
        f'{", ".join(path.name for path in model_dir.iterdir()) if model_dir.is_dir() else "inexistente"}'
    )


def _subprocess_detail(stderr, stdout):
    chunks = []
    for stream in (stderr, stdout):
        text = (stream or '').strip()
        if text:
            chunks.append(text[-1500:])
    return ' '.join(chunks) if chunks else 'sem detalhes de stderr/stdout'


def select_chord_source(other_audio, bass_audio):
    """
    Build a chord-oriented mix from accompaniment and bass.

    The accompaniment stem carries chord quality while a restrained bass layer
    reinforces roots. If the accompaniment is too weak, it is safer to use it
    alone (or bass only as a last resort) than to let bass dominate Chordino.
    """
    other = np.asarray(other_audio, dtype=np.float32)
    bass = np.asarray(bass_audio, dtype=np.float32)
    length = min(len(other), len(bass))
    other = other[:length]
    bass = bass[:length]

    other_rms = _rms(other)
    bass_rms = _rms(bass)
    silence_threshold = 0.003
    max_bass_ratio = 2.5

    if other_rms < silence_threshold and bass_rms >= silence_threshold:
        selected = bass
        source = 'baixo'
        bass_gain = 1.0
    elif other_rms >= silence_threshold and bass_rms > other_rms * max_bass_ratio:
        selected = other
        source = 'acompanhamento'
        bass_gain = 0.0
    else:
        bass_gain = 0.45
        selected = other + (bass * bass_gain)
        peak = float(np.max(np.abs(selected))) if len(selected) else 0.0
        if peak > 0.98:
            selected = selected * (0.98 / peak)
        source = 'acompanhamento_com_baixo'

    return selected.astype(np.float32), {
        'stem_escolhido_chordino': source,
        'ganho_baixo': bass_gain,
        'rms_acompanhamento': round(other_rms, 6),
        'rms_baixo': round(bass_rms, 6),
        'cobertura_acompanhamento': round(_coverage(other), 4),
        'cobertura_baixo': round(_coverage(bass), 4),
    }


def stem_metrics(audio):
    values = np.asarray(audio, dtype=np.float32)
    return {
        'rms': round(_rms(values), 6),
        'cobertura': round(_coverage(values), 4),
        'pico': round(float(np.max(np.abs(values))) if len(values) else 0.0, 6),
    }


def _rms(audio):
    if not len(audio):
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def _coverage(audio):
    if not len(audio):
        return 0.0
    window_size = min(22050, len(audio))
    windows = np.array_split(audio, max(1, len(audio) // window_size))
    active = sum(_rms(window) >= 0.003 for window in windows)
    return active / len(windows)
