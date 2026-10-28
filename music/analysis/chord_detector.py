import logging
from contextlib import nullcontext

from .lv_chordia_detector import detect_lv_chordia

logger = logging.getLogger('harmix.analysis')


def _stage(diagnostics, name, details=None):
    return diagnostics.stage(name, details=details) if diagnostics else nullcontext({})


def detect_chords(audio_path, duration, diagnostics=None):
    """Detect chords exclusively with LV-Chordia."""
    with _stage(diagnostics, 'lv_chordia_detection') as metrics:
        result = detect_lv_chordia(audio_path, duration)
        metrics['total_chords'] = len(result.chords)

    if not result.chords:
        raise RuntimeError('LV-Chordia nao retornou nenhum acorde valido.')

    logger.info('LV-Chordia detectou %s segmentos.', len(result.chords))
    return result
