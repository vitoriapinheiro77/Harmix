import logging
import math
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

from django.core.exceptions import ObjectDoesNotExist
from django.core.files.base import ContentFile

from .models import MusicProject

logger = logging.getLogger(__name__)


def delete_project_files(project):
    """Delete stored files associated with a music project."""
    errors = []
    file_fields = []

    if project.audio_file:
        file_fields.append(project.audio_file)

    for track in project.tracks.all():
        if track.audio_file:
            file_fields.append(track.audio_file)

    try:
        score = project.score
    except ObjectDoesNotExist:
        score = None

    if score:
        if score.musicxml_file:
            file_fields.append(score.musicxml_file)
        if score.pdf_file:
            file_fields.append(score.pdf_file)

    for field_file in file_fields:
        try:
            name = field_file.name
            if name and field_file.storage.exists(name):
                field_file.storage.delete(name)
        except Exception as exc:
            errors.append(f'Nao foi possivel apagar {field_file.name}: {exc}')

    temp_root = Path(tempfile.gettempdir())
    for temp_dir in temp_root.glob(f'harmix_{project.pk}_*'):
        try:
            if temp_dir.is_dir():
                shutil.rmtree(temp_dir)
        except Exception as exc:
            errors.append(f'Nao foi possivel apagar arquivos temporarios de {temp_dir.name}: {exc}')

    return errors


PDF_DEFAULT_ARTIST = 'nome do cantor'


def _normalized_manual_beat_start(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 4:
        return None
    return normalized


def _normalized_manual_position_ratio(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1:
        return None
    return normalized


def _manual_added_chord_id(measure_number, beat_start):
    identifier_beat = int(beat_start) if float(beat_start).is_integer() else beat_start
    return f'{measure_number}:{identifier_beat}'


def enqueue_project_reprocessing(project):
    """Reenfileira o processamento assíncrono reutilizando o áudio já armazenado."""
    if not project.audio_file:
        raise ValueError('Projeto nao possui arquivo de audio.')

    if project.task_id:
        try:
            from celery import current_app

            current_app.control.revoke(project.task_id, terminate=True, signal='SIGTERM')
        except Exception:
            logger.exception('Falha ao revogar task anterior do projeto %s', project.pk)

    project.status = MusicProject.Status.QUEUED
    project.processing_step = 'Reprocessamento solicitado. Aguardando worker.'
    project.error_message = ''
    project.completed_at = None
    project.save(update_fields=[
        'status',
        'processing_step',
        'error_message',
        'completed_at',
        'updated_at',
    ])

    from .tasks import process_music_project

    task = process_music_project.delay(project.pk)
    project.task_id = task.id or ''
    project.save(update_fields=['task_id', 'updated_at'])
    return task


def score_data_with_manual_chord_edits(score, *, include_editor_metadata=False):
    """Apply manual chord labels, insertions, and deletions to a detected-score copy."""
    score_data = deepcopy(score.data or {})
    manual_edits = score.manual_chord_edits or {}
    chord_edits = manual_edits.get('chords', {})
    added_chords = manual_edits.get('added_chords', {})
    deleted_chords = manual_edits.get('deleted_chords', {})
    chord_edits = chord_edits if isinstance(chord_edits, dict) else {}
    added_chords = added_chords if isinstance(added_chords, dict) else {}
    deleted_chords = deleted_chords if isinstance(deleted_chords, dict) else {}

    for measure in score_data.get('structure', {}).get('measures', []):
        measure_number = measure.get('number')
        chords = []
        for chord_index, chord in enumerate(measure.get('chords') or []):
            if deleted_chords.get(f'{measure_number}:{chord_index}'):
                continue
            detected_name = chord.get('chord_name') or 'N.C.'
            detected_beat_start = chord.get('beat_start', 0)
            if include_editor_metadata:
                chord['_detected_chord_name'] = detected_name
                chord['_detected_chord_index'] = chord_index
                chord['_detected_beat_start'] = detected_beat_start
            edit = chord_edits.get(f'{measure_number}:{chord_index}')
            if isinstance(edit, dict) and isinstance(edit.get('chord_name'), str):
                chord['chord_name'] = edit['chord_name']
                if include_editor_metadata:
                    chord['_is_manually_edited'] = True
            manual_beat_start = _normalized_manual_beat_start(
                edit.get('beat_start') if isinstance(edit, dict) else None
            )
            position_ratio = _normalized_manual_position_ratio(
                edit.get('position_ratio') if isinstance(edit, dict) else None
            )
            if position_ratio is not None:
                manual_beat_start = position_ratio * 4
                chord['_manual_position_ratio'] = position_ratio
            if manual_beat_start is not None:
                chord['beat_start'] = manual_beat_start
                if include_editor_metadata:
                    chord['_is_manually_edited'] = True
            chords.append(chord)

        for manual_chord_id, added_chord in added_chords.items():
            if not isinstance(added_chord, dict):
                continue
            manual_beat_start = _normalized_manual_beat_start(added_chord.get('beat_start'))
            if manual_beat_start is None:
                continue
            if not isinstance(added_chord.get('chord_name'), str):
                continue
            try:
                manual_measure_number = int(manual_chord_id.split(':', 1)[0])
            except (AttributeError, TypeError, ValueError):
                continue
            if manual_measure_number != measure_number:
                continue
            position_ratio = _normalized_manual_position_ratio(added_chord.get('position_ratio'))
            if position_ratio is not None:
                manual_beat_start = position_ratio * 4
            chord = {
                'chord_name': added_chord['chord_name'],
                'beat_start': manual_beat_start,
                '_manual_chord_id': manual_chord_id,
            }
            if position_ratio is not None:
                chord['_manual_position_ratio'] = position_ratio
            if include_editor_metadata:
                chord['_is_manually_edited'] = True
                chord['_is_manually_added'] = True
                chord['_persisted_manual_chord_id'] = manual_chord_id
            chords.append(chord)

        chords.sort(key=lambda chord: (float(chord.get('beat_start') or 0), chord.get('_manual_chord_id', '')))
        for chord_index, chord in enumerate(chords):
            start = float(chord.get('beat_start') or 0)
            next_start = (
                float(chords[chord_index + 1].get('beat_start') or 4)
                if chord_index + 1 < len(chords)
                else 4.0
            )
            chord['beat_duration'] = max(0.1, next_start - start)
        measure['chords'] = chords

    return score_data


def pdf_score_data(project, score):
    score_data = score_data_with_manual_chord_edits(score)
    # The PDF renderer centers labels over beat durations. A negligible duration
    # keeps a manually dragged label at its exact proportional position.
    for measure in score_data.get('structure', {}).get('measures', []):
        chords = measure.get('chords') or []
        if not chords:
            # The PDF renderer supplies a visible fallback for an empty list.
            # A whitespace marker preserves an intentionally blank measure.
            measure['chords'] = [{'chord_name': ' ', 'beat_start': 0, 'beat_duration': 4}]
            continue
        for chord in chords:
            position_ratio = _normalized_manual_position_ratio(
                chord.get('_manual_position_ratio')
            )
            if position_ratio is None:
                continue
            target_beat = position_ratio * 4
            epsilon = 1e-9
            chord['beat_start'] = max(0.0, target_beat - (epsilon / 2))
            chord['beat_duration'] = epsilon
    artist = (score_data.get('artist') or '').strip()
    score_data['artist'] = artist or PDF_DEFAULT_ARTIST
    return score_data


def score_has_pdf_data(score):
    score_data = score.data or {}
    return bool(score_data.get('chords') or score_data.get('structure', {}).get('measures'))


def regenerate_score_pdf(project, score=None, *, save=True):
    """Rebuild the score PDF from stored score data and optionally save it to media."""
    from .pipeline import HarmixPipeline

    if score is None:
        try:
            score = project.score
        except ObjectDoesNotExist:
            return None

    if not score_has_pdf_data(score):
        return None

    pipeline = HarmixPipeline(project)
    try:
        pdf_bytes = pipeline._make_pdf(pdf_score_data(project, score))
        if save:
            score.pdf_file.save(
                f'project_{project.pk}.pdf',
                ContentFile(pdf_bytes),
                save=True,
            )
        return pdf_bytes
    finally:
        pipeline._cleanup_workdir()
