"""
Exporta o PDF de um MusicProject para testes locais.

Exemplos:
    python manage.py export_project_pdf 6
    python manage.py export_project_pdf 6 --output exports/project_6_test.pdf
    python manage.py export_project_pdf 6 --save
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from music.models import MusicProject, Score
from music.utils import regenerate_score_pdf, score_has_pdf_data


class Command(BaseCommand):
    help = 'Gera o PDF de um projeto musical e salva em disco para testes.'

    def add_arguments(self, parser):
        parser.add_argument(
            'project_id',
            nargs='?',
            type=int,
            default=6,
            help='ID do projeto (padrao: 6)',
        )
        parser.add_argument(
            '--output',
            '-o',
            type=str,
            default='',
            help='Caminho do PDF de saida. Padrao: exports/project_<id>_<timestamp>.pdf',
        )
        parser.add_argument(
            '--save',
            action='store_true',
            help='Tambem atualiza score.pdf_file no banco/media do projeto',
        )

    def handle(self, *args, **options):
        project_id = options['project_id']
        project = MusicProject.objects.filter(pk=project_id).select_related('score').first()
        if not project:
            raise CommandError(f'Projeto {project_id} nao encontrado.')

        try:
            score = project.score
        except Score.DoesNotExist:
            raise CommandError(
                f'Projeto {project_id} nao possui score gerado. '
                'Processe a musica antes de exportar o PDF.'
            )

        score_data = score.data or {}
        if not score_has_pdf_data(score):
            raise CommandError(
                f'Score do projeto {project_id} esta vazio (sem acordes/compassos).'
            )

        pdf_bytes = regenerate_score_pdf(project, score, save=options['save'])

        output_path = self._resolve_output_path(project_id, options['output'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(pdf_bytes)

        self.stdout.write(self.style.SUCCESS(f'PDF gerado: {output_path}'))
        self.stdout.write(f'  Projeto: {project.title}')
        self.stdout.write(f'  Tom: {score_data.get("key") or project.key or "-"}')
        self.stdout.write(f'  BPM: {score_data.get("bpm") or project.bpm or "-"}')
        self.stdout.write(f'  Acordes: {len(score_data.get("chords", []))}')
        self.stdout.write(
            f'  Compassos: {score_data.get("structure", {}).get("measure_count", "-")}'
        )
        self.stdout.write(f'  Tamanho: {len(pdf_bytes)} bytes')

        if options['save']:
            self.stdout.write(self.style.SUCCESS(f'  Salvo em media: {score.pdf_file.name}'))

    def _resolve_output_path(self, project_id, output_arg):
        if output_arg:
            return Path(output_arg)

        from django.utils import timezone

        stamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        return Path(settings.BASE_DIR) / 'exports' / f'project_{project_id}_{stamp}.pdf'
