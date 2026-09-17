import logging
import math
import mimetypes
import json
import re
from io import BytesIO

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import FileResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.text import slugify
from django.views.decorators.http import require_GET, require_POST

from .forms import MusicProjectUploadForm
from .models import MusicProject, Score
from .utils import (
    delete_project_files,
    enqueue_project_reprocessing,
    generate_score_musicxml,
    musicxml_score_data,
    regenerate_score_pdf,
    score_data_with_manual_chord_edits,
)

logger = logging.getLogger(__name__)

CHORD_NAME_PATTERN = re.compile(
    r'^(?:N\.C\.|[A-G](?:#|b)?'
    r'(?:(?:(?:maj|min|m|dim|aug|add)?(?:[0-9]{1,2}|6/9)?|(?:[0-9]{1,2})?sus(?:2|4)?))?'
    r'(?:\((?:[#b]?(?:5|9|11|13))(?:,[#b]?(?:5|9|11|13))*\))?'
    r'(?:/[A-G](?:#|b)?)?)$'
)
KEY_NAME_PATTERN = re.compile(r'^[A-G](?:#|b)?m?$')


def _score_header_values(project, score_data):
    """Return the score header, falling back to project metadata when needed."""
    score_data = score_data or {}
    title = score_data.get('title') or project.title
    artist = score_data.get('artist') or project.artist or 'Harmix'
    key = score_data.get('key') or project.key or '-'
    bpm = score_data.get('bpm') or project.bpm or '-'
    return {
        'title': str(title),
        'artist': str(artist),
        'key': str(key),
        'bpm': bpm,
    }


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
def export_musicxml_view(request, pk):
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
        xml = generate_score_musicxml(project, score)
        score_data = musicxml_score_data(project, score)
    except Exception:
        logger.exception('Falha ao exportar MusicXML do projeto %s', project.pk)
        messages.error(request, 'Nao foi possivel gerar o MusicXML agora. Tente novamente.')
        return redirect('music:score_edit', pk=project.pk)

    filename = slugify(score_data.get('title') or project.title) or 'partitura'
    return FileResponse(
        BytesIO(xml.encode('utf-8')),
        content_type='application/vnd.recordare.musicxml+xml',
        as_attachment=True,
        filename=f'{filename}.musicxml',
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

    score_data = score_data_with_manual_chord_edits(score, include_editor_metadata=True)
    header_values = _score_header_values(project, score_data)
    header_original_values = _score_header_values(project, score.data)
    return render(
        request,
        'music/score_edit.html',
        {
            'project': project,
            'score': score,
            'score_data': score_data,
            'header_values': header_values,
            'header_original_values': header_original_values,
            'chords': score_data.get('chords', []),
        },
    )


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
    added_chords = payload.get('added_chords', []) if isinstance(payload, dict) else None
    deleted_chords = payload.get('deleted_chords', []) if isinstance(payload, dict) else None
    deleted_added_chords = payload.get('deleted_added_chords', []) if isinstance(payload, dict) else None
    header_edits = payload.get('header_edits', {}) if isinstance(payload, dict) else None
    if (
        not isinstance(changes, list)
        or not isinstance(added_chords, list)
        or not isinstance(deleted_chords, list)
        or not isinstance(deleted_added_chords, list)
        or not isinstance(header_edits, dict)
        or len(changes) + len(added_chords) + len(deleted_chords) + len(deleted_added_chords) + len(header_edits) > 104
    ):
        return JsonResponse({'error': 'Lista de alteracoes invalida.'}, status=400)

    if set(header_edits) - {'title', 'artist', 'key', 'bpm'}:
        return JsonResponse({'error': 'Alteracao de cabecalho invalida.'}, status=400)
    validated_header_edits = {}
    for field_name, value in header_edits.items():
        if field_name in {'title', 'artist'}:
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > 255:
                return JsonResponse({'error': 'Titulo ou artista invalido.'}, status=400)
            validated_header_edits[field_name] = value.strip()
        elif field_name == 'key':
            if not isinstance(value, str) or not KEY_NAME_PATTERN.fullmatch(value.strip()):
                return JsonResponse({'error': 'Tom musical invalido.'}, status=400)
            validated_header_edits[field_name] = value.strip()
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                return JsonResponse({'error': 'BPM invalido.'}, status=400)
            try:
                bpm = float(value)
            except (TypeError, ValueError):
                return JsonResponse({'error': 'BPM invalido.'}, status=400)
            if not math.isfinite(bpm) or not 0 < bpm <= 999:
                return JsonResponse({'error': 'BPM invalido.'}, status=400)
            validated_header_edits['bpm'] = int(bpm) if bpm.is_integer() else bpm

    detected_chords = _detected_chord_lookup(score)
    detected_beat_starts = {}
    for measure in (score.data or {}).get('structure', {}).get('measures', []):
        measure_number = measure.get('number')
        if not isinstance(measure_number, int):
            continue
        for chord_index, chord in enumerate(measure.get('chords') or []):
            detected_beat_starts[(measure_number, chord_index)] = chord.get('beat_start', 0)

    validated_changes = []
    for change in changes:
        if not isinstance(change, dict):
            return JsonResponse({'error': 'Alteracao de acorde invalida.'}, status=400)
        measure_number = change.get('measure_number')
        chord_index = change.get('chord_index')
        chord_name = change.get('chord_name')
        beat_start = change.get('beat_start')
        position_ratio_value = change.get('position_ratio')
        position_ratio = (
            _normalized_manual_position_ratio(position_ratio_value)
            if position_ratio_value is not None
            else None
        )
        normalized_beat_start = (
            _normalized_manual_beat_start(beat_start)
            if beat_start is not None
            else None
        )
        if position_ratio is not None:
            normalized_beat_start = position_ratio * 4
        if (
            isinstance(measure_number, bool)
            or not isinstance(measure_number, int)
            or isinstance(chord_index, bool)
            or not isinstance(chord_index, int)
            or (measure_number, chord_index) not in detected_chords
            or not isinstance(chord_name, str)
            or (beat_start is not None and normalized_beat_start is None)
            or (position_ratio_value is not None and position_ratio is None)
        ):
            return JsonResponse({'error': 'Compasso ou acorde invalido.'}, status=400)
        chord_name = chord_name.strip()
        if not chord_name or len(chord_name) > 20 or not CHORD_NAME_PATTERN.fullmatch(chord_name):
            return JsonResponse({'error': 'Nome de acorde invalido.'}, status=400)
        validated_changes.append(
            (measure_number, chord_index, chord_name, normalized_beat_start, position_ratio)
        )

    validated_deleted_chords = []
    for deleted_chord in deleted_chords:
        if not isinstance(deleted_chord, dict):
            return JsonResponse({'error': 'Exclusao de acorde invalida.'}, status=400)
        measure_number = deleted_chord.get('measure_number')
        chord_index = deleted_chord.get('chord_index')
        if (
            isinstance(measure_number, bool)
            or not isinstance(measure_number, int)
            or isinstance(chord_index, bool)
            or not isinstance(chord_index, int)
            or (measure_number, chord_index) not in detected_chords
        ):
            return JsonResponse({'error': 'Compasso ou acorde excluido invalido.'}, status=400)
        validated_deleted_chords.append((measure_number, chord_index))

    deleted_chord_keys = {
        f'{measure_number}:{chord_index}'
        for measure_number, chord_index in validated_deleted_chords
    }
    if len(deleted_chord_keys) != len(validated_deleted_chords) or any(
        f'{measure_number}:{chord_index}' in deleted_chord_keys
        for measure_number, chord_index, _, _, _ in validated_changes
    ):
        return JsonResponse({'error': 'Alteracoes de exclusao de acorde invalidas.'}, status=400)

    measure_numbers = {measure_number for measure_number, _ in detected_chords}

    validated_added_chords = []
    for added_chord in added_chords:
        if not isinstance(added_chord, dict):
            return JsonResponse({'error': 'Acorde adicionado invalido.'}, status=400)
        measure_number = added_chord.get('measure_number')
        beat_start = _normalized_manual_beat_start(added_chord.get('beat_start'))
        position_ratio_value = added_chord.get('position_ratio')
        position_ratio = (
            _normalized_manual_position_ratio(position_ratio_value)
            if position_ratio_value is not None
            else None
        )
        if position_ratio is not None:
            beat_start = position_ratio * 4
        chord_name = added_chord.get('chord_name')
        previous_manual_chord_id = added_chord.get('previous_manual_chord_id')
        if (
            isinstance(measure_number, bool)
            or not isinstance(measure_number, int)
            or beat_start is None
            or measure_number not in measure_numbers
            or not isinstance(chord_name, str)
            or (previous_manual_chord_id is not None and not isinstance(previous_manual_chord_id, str))
            or (position_ratio_value is not None and position_ratio is None)
        ):
            return JsonResponse({'error': 'Compasso, beat ou acorde adicionado invalido.'}, status=400)
        chord_name = chord_name.strip()
        if not chord_name or len(chord_name) > 20 or not CHORD_NAME_PATTERN.fullmatch(chord_name):
            return JsonResponse({'error': 'Nome de acorde invalido.'}, status=400)
        validated_added_chords.append(
            (measure_number, beat_start, chord_name, previous_manual_chord_id, position_ratio)
        )

    manual_edits = dict(score.manual_chord_edits or {})
    stored_header_edits = dict(manual_edits.get('manual_header_edits') or {})
    original_header_values = _score_header_values(project, score.data)
    for field_name, value in validated_header_edits.items():
        original_value = original_header_values[field_name]
        if field_name == 'bpm':
            try:
                matches_original = float(value) == float(original_value)
            except (TypeError, ValueError):
                matches_original = False
        else:
            matches_original = value == original_value
        if matches_original:
            stored_header_edits.pop(field_name, None)
        else:
            stored_header_edits[field_name] = value
    chord_edits = dict(manual_edits.get('chords') or {})
    stored_deleted_chords = dict(manual_edits.get('deleted_chords') or {})
    stored_deleted_chords.update({key: True for key in deleted_chord_keys})
    for chord_key in deleted_chord_keys:
        chord_edits.pop(chord_key, None)
    effective_beat_starts = {
        chord_key: beat_start
        for chord_key, beat_start in detected_beat_starts.items()
        if not stored_deleted_chords.get(f'{chord_key[0]}:{chord_key[1]}')
    }
    freely_positioned_detected = set()
    for key, edit in chord_edits.items():
        if not isinstance(edit, dict):
            continue
        manual_beat_start = _normalized_manual_beat_start(edit.get('beat_start'))
        if manual_beat_start is None:
            continue
        try:
            measure_number, chord_index = (int(value) for value in key.split(':', 1))
        except (TypeError, ValueError):
            continue
        if (measure_number, chord_index) in effective_beat_starts:
            effective_beat_starts[(measure_number, chord_index)] = manual_beat_start
            if _normalized_manual_position_ratio(edit.get('position_ratio')) is not None:
                freely_positioned_detected.add((measure_number, chord_index))
    for measure_number, chord_index, chord_name, beat_start, position_ratio in validated_changes:
        key = f'{measure_number}:{chord_index}'
        edit = dict(chord_edits.get(key) or {})
        if chord_name == detected_chords[(measure_number, chord_index)]:
            edit.pop('chord_name', None)
        else:
            edit['chord_name'] = chord_name
        if beat_start is not None:
            effective_beat_starts[(measure_number, chord_index)] = beat_start
            if position_ratio is not None:
                edit['beat_start'] = beat_start
                edit['position_ratio'] = position_ratio
                freely_positioned_detected.add((measure_number, chord_index))
            elif beat_start == detected_beat_starts[(measure_number, chord_index)]:
                edit.pop('beat_start', None)
                edit.pop('position_ratio', None)
            else:
                edit['beat_start'] = beat_start
        if edit:
            chord_edits[key] = edit
        else:
            chord_edits.pop(key, None)

    occupied_detected_beats = {}
    detected_at_position = {}
    for chord_key, beat_start in effective_beat_starts.items():
        measure_number, _ = chord_key
        location = (measure_number, beat_start)
        other_key = detected_at_position.get(location)
        if (
            other_key is not None
            and chord_key not in freely_positioned_detected
            and other_key not in freely_positioned_detected
        ):
            return JsonResponse({'error': 'Dois acordes nao podem iniciar no mesmo beat.'}, status=400)
        detected_at_position[location] = chord_key
        occupied_detected_beats.setdefault(measure_number, set()).add(beat_start)

    stored_added_chords = dict(manual_edits.get('added_chords') or {})
    if len(set(deleted_added_chords)) != len(deleted_added_chords) or not all(
        isinstance(manual_chord_id, str) and manual_chord_id in stored_added_chords
        for manual_chord_id in deleted_added_chords
    ):
        return JsonResponse({'error': 'Acorde adicionado excluido invalido.'}, status=400)
    deleted_added_chord_ids = set(deleted_added_chords)
    if any(
        previous_manual_chord_id in deleted_added_chord_ids
        for _, _, _, previous_manual_chord_id, _ in validated_added_chords
        if previous_manual_chord_id is not None
    ):
        return JsonResponse({'error': 'Alteracoes de exclusao de acorde invalidas.'}, status=400)
    for manual_chord_id in deleted_added_chord_ids:
        stored_added_chords.pop(manual_chord_id)
    moved_added_ids = set()
    for measure_number, _, _, previous_manual_chord_id, _ in validated_added_chords:
        if previous_manual_chord_id is None:
            continue
        stored_chord = stored_added_chords.get(previous_manual_chord_id)
        if (
            not isinstance(stored_chord, dict)
            or previous_manual_chord_id in moved_added_ids
            or not previous_manual_chord_id.startswith(f'{measure_number}:')
        ):
            return JsonResponse({'error': 'Acorde adicionado invalido.'}, status=400)
        moved_added_ids.add(previous_manual_chord_id)
    occupied_added_beats = set()
    for manual_chord_id, stored_chord in stored_added_chords.items():
        if not isinstance(stored_chord, dict):
            continue
        try:
            measure_number, _ = manual_chord_id.split(':', 1)
            measure_number = int(measure_number)
            beat_start = _normalized_manual_beat_start(stored_chord.get('beat_start'))
        except (TypeError, ValueError):
            continue
        if beat_start is None:
            continue
        location = (measure_number, beat_start)
        if manual_chord_id not in moved_added_ids:
            occupied_added_beats.add(location)
    for measure_number, beat_start, chord_name, previous_manual_chord_id, position_ratio in validated_added_chords:
        location = (measure_number, beat_start)
        if previous_manual_chord_id is None and (
            beat_start in occupied_detected_beats.get(measure_number, set())
            or location in occupied_added_beats
        ):
            return JsonResponse({'error': 'Ja existe um acorde iniciando nesse beat.'}, status=400)
        occupied_added_beats.add(location)
        manual_chord_id = previous_manual_chord_id or _manual_added_chord_id(measure_number, beat_start)
        stored_added_chords[manual_chord_id] = {
            'chord_name': chord_name,
            'beat_start': beat_start,
        }
        if position_ratio is not None:
            stored_added_chords[manual_chord_id]['position_ratio'] = position_ratio
    manual_edits['manual_header_edits'] = stored_header_edits
    manual_edits['chords'] = chord_edits
    manual_edits['added_chords'] = stored_added_chords
    manual_edits['deleted_chords'] = stored_deleted_chords
    score.manual_chord_edits = manual_edits
    score.save(update_fields=['manual_chord_edits'])
    return JsonResponse({
        'saved_changes': len(validated_changes),
        'saved_added_chords': len(validated_added_chords),
        'saved_deleted_chords': len(validated_deleted_chords),
        'saved_deleted_added_chords': len(deleted_added_chord_ids),
        'saved_header_edits': len(validated_header_edits),
    })


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
