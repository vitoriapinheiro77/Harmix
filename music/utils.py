import logging
import shutil
import tempfile
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


def pdf_score_data(project, score):
    score_data = dict(score.data or {})
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
