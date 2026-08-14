import logging

logger = logging.getLogger('harmix.analysis')


def log_analysis_start(audio_path):
    logger.info('Iniciando analise musical: %s', audio_path)


def log_bpm_result(bpm, source, errors=None):
    if errors:
        logger.warning('BPM (%s): %s — erros: %s', source, bpm, '; '.join(errors))
    else:
        logger.info('BPM detectado (%s): %s', source, bpm)


def log_beat_result(rhythm):
    logger.info(
        'Beats (%s): %s | Downbeats: %s',
        rhythm.source,
        len(rhythm.beats),
        len(rhythm.downbeats),
    )
    for error in rhythm.errors:
        logger.warning('Beat analysis: %s', error)


def log_chord_result(result):
    logger.info('Acordes detectados (%s): %s segmentos', result.source, len(result.chords))
    for error in result.errors:
        logger.warning('Chord analysis: %s', error)


def log_analysis_complete(processing_seconds, errors=None):
    logger.info('Analise musical concluida em %.2fs', processing_seconds)
    if errors:
        logger.warning('Analise concluida com avisos: %s', '; '.join(errors))
