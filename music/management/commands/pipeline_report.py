import json

from django.core.management.base import BaseCommand, CommandError

from music.models import MusicProject


class Command(BaseCommand):
    help = 'Exibe o diagnostico da ultima execucao da pipeline de um projeto.'

    def add_arguments(self, parser):
        parser.add_argument('project_id', type=int)
        parser.add_argument(
            '--json',
            action='store_true',
            dest='as_json',
            help='Imprime o diagnostico completo em JSON.',
        )
        parser.add_argument(
            '--events',
            action='store_true',
            help='Inclui a linha do tempo de eventos no relatorio textual.',
        )

    def handle(self, *args, **options):
        try:
            project = MusicProject.objects.select_related('score').get(
                pk=options['project_id'],
            )
        except MusicProject.DoesNotExist as exc:
            raise CommandError('Projeto nao encontrado.') from exc

        score = getattr(project, 'score', None)
        diagnostics = (score.data or {}).get('pipeline_diagnostics') if score else None
        if not diagnostics:
            raise CommandError(
                'Este projeto nao possui diagnosticos. Reprocesse-o com a instrumentacao ativa.',
            )

        if options['as_json']:
            self.stdout.write(json.dumps(diagnostics, ensure_ascii=False, indent=2))
            return

        self.stdout.write(f'Projeto: {project.pk} - {project.title}')
        self.stdout.write(f"Execucao: {diagnostics.get('run_id', '-')}")
        self.stdout.write(f"Status: {diagnostics.get('status', '-')}")
        self.stdout.write(
            f"Duracao total: {diagnostics.get('total_duration_seconds', 0):.4f}s",
        )
        peak_memory = diagnostics.get('peak_memory_mb')
        self.stdout.write(
            f"Pico de memoria do processo: {peak_memory if peak_memory is not None else 'indisponivel'} MB",
        )
        self.stdout.write('')
        self.stdout.write('Etapas:')

        stages = diagnostics.get('stages') or []
        total = float(diagnostics.get('total_duration_seconds') or 0)
        for stage in stages:
            duration = float(stage.get('duration_seconds') or 0)
            percent = (duration / total * 100) if total else 0
            self.stdout.write(
                f"- {stage.get('name')}: {duration:.4f}s "
                f"({percent:.1f}%) [{stage.get('status')}]",
            )

        parent_stages = {
            stage.get('parent_stage')
            for stage in stages
            if stage.get('parent_stage')
        }
        timed_stages = [
            stage for stage in stages
            if stage.get('status') == 'completed'
            and stage.get('duration_seconds') is not None
            and stage.get('name') not in parent_stages
        ]
        if timed_stages:
            bottleneck = max(
                timed_stages,
                key=lambda stage: float(stage.get('duration_seconds') or 0),
            )
            self.stdout.write('')
            self.stdout.write(
                self.style.WARNING(
                    'Gargalo provavel: '
                    f"{bottleneck.get('name')} "
                    f"({float(bottleneck.get('duration_seconds') or 0):.4f}s).",
                ),
            )

        if diagnostics.get('error'):
            error = diagnostics['error']
            self.stdout.write(
                self.style.ERROR(
                    f"Erro final: {error.get('type')}: {error.get('message')}",
                ),
            )

        if options['events']:
            self.stdout.write('')
            self.stdout.write('Eventos:')
            for event in diagnostics.get('events') or []:
                self.stdout.write(
                    f"- +{float(event.get('elapsed_seconds') or 0):.4f}s "
                    f"{event.get('event')} [{event.get('stage') or '-'}]",
                )
