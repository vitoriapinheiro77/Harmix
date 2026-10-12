import logging

from celery import shared_task
from celery.signals import task_failure

from .models import MusicProject
from .pipeline import HarmixPipeline


logger = logging.getLogger(__name__)


@shared_task(bind=True)
def process_music_project(self, project_id):
    logger.info(
        'Iniciando task de processamento project_id=%s task_id=%s.',
        project_id,
        self.request.id,
    )
    project = MusicProject.objects.get(pk=project_id)
    project.task_id = self.request.id or ''
    project.save(update_fields=['task_id', 'updated_at'])
    try:
        HarmixPipeline(project).run()
    except Exception as exc:
        logger.exception(
            'Falha na task de processamento project_id=%s task_id=%s.',
            project_id,
            self.request.id,
        )
        project.refresh_from_db()
        if project.status != MusicProject.Status.DONE:
            project.status = MusicProject.Status.ERROR
            project.error_message = str(exc)
            project.processing_step = 'Erro durante o processamento.'
            project.save(update_fields=['status', 'error_message', 'processing_step', 'updated_at'])
        raise
    logger.info(
        'Task concluida project_id=%s task_id=%s status=%s.',
        project_id,
        self.request.id,
        project.status,
    )
    return {'project_id': project_id, 'status': project.status}


@task_failure.connect
def on_process_music_failure(sender=None, task_id=None, exception=None, args=None, **kwargs):
    if not sender or sender.name != 'music.tasks.process_music_project':
        return

    project_id = (args or [None])[0]
    if not project_id:
        return

    try:
        project = MusicProject.objects.get(pk=project_id)
    except MusicProject.DoesNotExist:
        return

    if project.status in (MusicProject.Status.DONE, MusicProject.Status.ERROR):
        return

    project.status = MusicProject.Status.ERROR
    project.error_message = f'Processamento interrompido: {exception}'
    project.processing_step = 'Erro durante o processamento.'
    project.save(update_fields=['status', 'error_message', 'processing_step', 'updated_at'])
