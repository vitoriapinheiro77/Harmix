import logging
import mimetypes
import json
import re
from copy import deepcopy
from io import BytesIO

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import FileResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from .forms import MusicProjectUploadForm
from .models import MusicProject, Score
from .utils import delete_project_files, enqueue_project_reprocessing, regenerate_score_pdf

logger = logging.getLogger(__name__)

CHORD_NAME_PATTERN = re.compile(r'^(?:[A-G](?:#|b)?[A-Za-z0-9+#()/.-]*|N\.C\.)$')


@login_required
def project_list_view(request):
    projects = MusicProject.objects.filter(user=request.user)
    return render(request, 'music/project_list.html', {'projects': projects})


def _try_regenerate_pdf(project, score):
    if project.status != MusicProject.Status.DONE or not score:
        return
    try:
        regenerate_score_pdf(project, score)
    except Exception:
        logger.exception('Falha ao regenerar PDF do projeto %s', project.pk)


@login_required
def project_detail_view(request, pk):
    project = get_object_or_404(
        MusicProject.objects.select_related('score').prefetch_related('tracks__chord_segments'),
        pk=pk,
        user=request.user,
    )
    try:
        score = project.score
    except Score.DoesNotExist:
        score = None

    if score:
        _try_regenerate_pdf(project, score)
        score.refresh_from_db(fields=['pdf_file'])

    pipeline_steps = [
        'Arquivo de audio recebido',
        'Analise de tom e BPM',
        'Separacao harmonica / percussiva',
        'Extracao de acordes',
        'Geracao do MusicXML e PDF',
    ]
    return render(
        request,
        'music/project_detail.html',
        {
            'project': project,
            'score': score,
            'tracks': project.tracks.all(),
            'pipeline_steps': pipeline_steps,
        },
    )


@login_required
def project_original_audio_view(request, pk):
    project = get_object_or_404(MusicProject, pk=pk, user=request.user)
    if not project.audio_file:
        messages.error(request, 'O audio original nao esta disponivel neste projeto.')
        return redirect('music:project_detail', pk=project.pk)
    return _stream_project_audio(project.audio_file)


@login_required
def project_track_audio_view(request, pk, track_id):
    project = get_object_or_404(MusicProject, pk=pk, user=request.user)
    track = get_object_or_404(project.tracks.all(), pk=track_id)
    return _stream_project_audio(track.audio_file)


def _stream_project_audio(audio_file):
    content_type, _ = mimetypes.guess_type(audio_file.name)
    return FileResponse(
        audio_file.open('rb'),
        content_type=content_type or 'application/octet-stream',
        as_attachment=False,
        filename=audio_file.name.rsplit('/', 1)[-1],
    )


@login_required
def export_pdf_view(request, pk):
    project = get_object_or_404(
        MusicProject.objects.select_related('score'),
        pk=pk,
        user=request.user,
    )
    try:
        score = project.score
    except Score.DoesNotExist:
        messages.error(request, 'A partitura ainda nao foi gerada para este projeto.')
        return redirect('music:project_detail', pk=project.pk)

    if project.status != MusicProject.Status.DONE:
        messages.error(request, 'O projeto ainda nao foi processado completamente.')
        return redirect('music:project_detail', pk=project.pk)

    try:
        pdf_bytes = regenerate_score_pdf(project, score)
    except Exception:
        logger.exception('Falha ao exportar PDF do projeto %s', project.pk)
        messages.error(request, 'Nao foi possivel gerar o PDF agora. Tente novamente.')
        return redirect('music:project_detail', pk=project.pk)

    if not pdf_bytes:
        messages.error(request, 'Nao ha dados suficientes para gerar o PDF deste projeto.')
        return redirect('music:project_detail', pk=project.pk)

    filename = project.title or 'partitura'
    safe_filename = ''.join(c for c in filename if c.isalnum() or c in ' ._-').strip() or 'partitura'
    return FileResponse(
        BytesIO(pdf_bytes),
        content_type='application/pdf',
        filename=f'{safe_filename}.pdf',
    )


@login_required
@require_GET
def score_edit_view(request, pk):
    project = get_object_or_404(
        MusicProject.objects.select_related('score'),
        pk=pk,
        user=request.user,
    )
    try:
        score = project.score
    except Score.DoesNotExist:
        messages.error(request, 'A partitura ainda nao foi gerada para este projeto.')
        return redirect('music:project_detail', pk=project.pk)

    score_data = _score_data_with_manual_chord_edits(score)
    return render(
        request,
        'music/score_edit.html',
        {
            'project': project,
            'score': score,
            'score_data': score_data,
            'chords': score_data.get('chords', []),
        },
    )


def _score_data_with_manual_chord_edits(score):
    """Overlay manual chord labels without changing the detected score data."""
    score_data = deepcopy(score.data or {})
    edits = (score.manual_chord_edits or {}).get('chords', {})
    if not isinstance(edits, dict):
        edits = {}

    for measure in score_data.get('structure', {}).get('measures', []):
        measure_number = measure.get('number')
        for chord_index, chord in enumerate(measure.get('chords') or []):
            detected_name = chord.get('chord_name') or 'N.C.'
            chord['_detected_chord_name'] = detected_name
            edit = edits.get(f'{measure_number}:{chord_index}')
            if isinstance(edit, dict) and isinstance(edit.get('chord_name'), str):
                chord['chord_name'] = edit['chord_name']
                chord['_is_manually_edited'] = True
    return score_data


def _detected_chord_lookup(score):
    lookup = {}
    for measure in (score.data or {}).get('structure', {}).get('measures', []):
        measure_number = measure.get('number')
        if not isinstance(measure_number, int):
            continue
        for chord_index, chord in enumerate(measure.get('chords') or []):
            lookup[(measure_number, chord_index)] = chord.get('chord_name') or 'N.C.'
    return lookup


@login_required
@require_POST
def score_chord_edits_view(request, pk):
    project = get_object_or_404(
        MusicProject.objects.select_related('score'),
        pk=pk,
        user=request.user,
    )
    try:
        score = project.score
    except Score.DoesNotExist:
        return JsonResponse({'error': 'Partitura nao encontrada.'}, status=404)

    try:
        payload = json.loads(request.body)
    except (TypeError, ValueError):
        return JsonResponse({'error': 'Corpo JSON invalido.'}, status=400)
    changes = payload.get('changes') if isinstance(payload, dict) else None
    if not isinstance(changes, list) or len(changes) > 100:
        return JsonResponse({'error': 'Lista de alteracoes invalida.'}, status=400)

    detected_chords = _detected_chord_lookup(score)
    validated_changes = []
    for change in changes:
        if not isinstance(change, dict):
            return JsonResponse({'error': 'Alteracao de acorde invalida.'}, status=400)
        measure_number = change.get('measure_number')
        chord_index = change.get('chord_index')
        chord_name = change.get('chord_name')
        if (
            isinstance(measure_number, bool)
            or not isinstance(measure_number, int)
            or isinstance(chord_index, bool)
            or not isinstance(chord_index, int)
            or (measure_number, chord_index) not in detected_chords
            or not isinstance(chord_name, str)
        ):
            return JsonResponse({'error': 'Compasso ou acorde invalido.'}, status=400)
        chord_name = chord_name.strip()
        if not chord_name or len(chord_name) > 20 or not CHORD_NAME_PATTERN.fullmatch(chord_name):
            return JsonResponse({'error': 'Nome de acorde invalido.'}, status=400)
        validated_changes.append((measure_number, chord_index, chord_name))

    manual_edits = dict(score.manual_chord_edits or {})
    chord_edits = dict(manual_edits.get('chords') or {})
    for measure_number, chord_index, chord_name in validated_changes:
        key = f'{measure_number}:{chord_index}'
        if chord_name == detected_chords[(measure_number, chord_index)]:
            chord_edits.pop(key, None)
        else:
            chord_edits[key] = {'chord_name': chord_name}
    manual_edits['chords'] = chord_edits
    score.manual_chord_edits = manual_edits
    score.save(update_fields=['manual_chord_edits'])
    return JsonResponse({'saved_changes': len(validated_changes)})


@login_required
@require_POST
def project_create_view(request):
    form = MusicProjectUploadForm(request.POST, request.FILES)
    if form.is_valid():
        project = form.save(user=request.user)
        try:
            from .tasks import process_music_project

            task = process_music_project.delay(project.pk)
            project.task_id = task.id or ''
            project.save(update_fields=['task_id', 'updated_at'])
            messages.success(request, 'Arquivo enviado. O processamento foi iniciado.')
        except Exception as exc:
            project.status = MusicProject.Status.ERROR
            project.error_message = f'Nao foi possivel iniciar a fila de processamento: {exc}'
            project.processing_step = 'Erro ao enviar tarefa para o worker.'
            project.save(update_fields=['status', 'error_message', 'processing_step', 'updated_at'])
            messages.error(request, 'Arquivo salvo, mas o worker de processamento nao foi iniciado.')
        return redirect('music:project_detail', pk=project.pk)

    for field, errors in form.errors.items():
        for error in errors:
            messages.error(request, error)
    return redirect('home')


@login_required
def project_status_view(request, pk):
    project = get_object_or_404(MusicProject, pk=pk, user=request.user)
    return JsonResponse({
        'status': project.status,
        'processing_step': project.processing_step,
        'error_message': project.error_message,
    })


@login_required
@require_POST
def project_reprocess_view(request, pk):
    project = get_object_or_404(MusicProject, pk=pk, user=request.user)

    if not project.audio_file:
        messages.error(request, 'Este projeto nao possui audio armazenado para reprocessar.')
        return redirect('music:project_detail', pk=project.pk)

    try:
        enqueue_project_reprocessing(project)
        messages.success(
            request,
            'Reprocessamento iniciado. A pagina sera atualizada automaticamente ao concluir.',
        )
    except Exception as exc:
        logger.exception('Falha ao reprocessar projeto %s', project.pk)
        messages.error(request, f'Nao foi possivel iniciar o reprocessamento: {exc}')

    return redirect('music:project_detail', pk=project.pk)


@login_required
@require_POST
def project_delete_view(request, pk):
    project = get_object_or_404(
        MusicProject.objects.select_related('score').prefetch_related('tracks'),
        pk=pk,
        user=request.user,
    )

    try:
        if project.task_id:
            try:
                from celery import current_app

                current_app.control.revoke(project.task_id, terminate=True, signal='SIGTERM')
            except Exception:
                pass

        with transaction.atomic():
            cleanup_errors = delete_project_files(project)
            project.delete()
    except Exception:
        messages.error(
            request,
            'Nao foi possivel excluir este projeto agora. Tente novamente em alguns instantes.',
        )
        return redirect('music:project_list')

    if cleanup_errors:
        messages.error(
            request,
            'O projeto foi removido, mas alguns arquivos associados nao puderam ser apagados automaticamente.',
        )
    else:
        messages.success(request, 'Projeto removido com sucesso.')

    return redirect('music:project_list')
