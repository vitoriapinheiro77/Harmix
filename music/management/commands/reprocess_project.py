from django.core.management.base import BaseCommand, CommandError

from music.models import MusicProject
from music.utils import enqueue_project_reprocessing


class Command(BaseCommand):
    help = 'Reenfileira o processamento de um projeto musical travado ou com erro.'

    def add_arguments(self, parser):
        parser.add_argument('project_id', type=int, help='ID do projeto')

    def handle(self, *args, **options):
        project_id = options['project_id']
        project = MusicProject.objects.filter(pk=project_id).first()
        if not project:
            raise CommandError(f'Projeto {project_id} nao encontrado.')

        try:
            enqueue_project_reprocessing(project)
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS(
            f'Projeto {project_id} reenfileirado. Task: {project.task_id}'
        ))
