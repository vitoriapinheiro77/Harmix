import html
import logging
import math
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

logger = logging.getLogger(__name__)

from django.core.files import File
from django.core.files.base import ContentFile
from django.conf import settings
from django.utils import timezone

from .analysis.analysis_pipeline import run_chord_analysis, run_rhythm_analysis
from .analysis.beat_detector import quantize_detections
from .analysis.chord_stability import stabilize_chord_detections
from .models import ChordSegment, MusicProject, Score, Track
from .pdf_notation import draw_system_signature
from .pipeline_diagnostics import PipelineDiagnostics
from .separation import SeparationError, run_demucs, select_chord_source, stem_metrics
from .structure import build_musical_structure
from .utils import PDF_DEFAULT_ARTIST


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Chord quality templates: (semitone offsets from root, label suffix, forbidden semitones)
# Forbidden semitones: notes whose presence PENALISES this chord's score.
# sus chords are penalised heavily when the 3rd (minor or major) is present.
CHORD_TEMPLATES = [
    ([0, 4, 7],        '',      []),         # major triad
    ([0, 3, 7],        'm',     []),         # minor triad
    ([0, 4, 7, 10],    '7',     []),         # dominant 7th
    ([0, 4, 7, 11],    'maj7',  []),         # major 7th
    ([0, 3, 7, 10],    'm7',    []),         # minor 7th
    ([0, 3, 7, 11],    'mM7',   []),         # minor major 7th
    ([0, 3, 6],        'dim',   []),         # diminished triad
    ([0, 3, 6, 9],     'dim7',  []),         # diminished 7th
    ([0, 3, 6, 10],    'm7b5',  []),         # half diminished
    ([0, 4, 8],        'aug',   []),         # augmented triad
    ([0, 5, 7],        'sus4',  [3, 4]),     # sus4 ÔÇö penalise b3 and 3 heavily
    ([0, 2, 7],        'sus2',  [3, 4]),     # sus2 ÔÇö penalise b3 and 3 heavily
    ([0, 4, 7, 9],     '6',     []),         # major 6th
    ([0, 3, 7, 9],     'm6',    []),         # minor 6th
]

# Weight given to forbidden notes when penalising a chord score.
# 0.6 means 60% of the forbidden note's chroma energy is subtracted from the score.
_FORBIDDEN_PENALTY = 0.6

# Chord extraction limits ÔÇö keeps memory/CPU bounded on long worship/live tracks.
MAX_CHORD_WINDOWS = 360
MIN_CHORD_HOP_SEC = 0.75
MAX_CHORD_WINDOW_SEC = 2.0


class HarmixPipeline:
    def __init__(self, project):
        self.project = project
        self.workdir = Path(tempfile.mkdtemp(prefix=f'harmix_{project.pk}_'))
        self._pipeline_log = []
        self._rhythm_grid = None
        self._diagnostics_active = False
        self._diagnostics = PipelineDiagnostics(
            project.pk,
            on_change=self._persist_pipeline_diagnostics,
        )

    def _log(self, step, details):
        """Append a debug entry that will be stored in score.data['pipeline_log']."""
        self._pipeline_log.append({'step': step, 'details': details})
        if self._diagnostics_active:
            self._diagnostics.event('checkpoint', stage=step, details=details)

    def _stage(self, name, details=None):
        if not self._diagnostics_active:
            return nullcontext({})
        return self._diagnostics.stage(name, details=details)

    def _persist_pipeline_diagnostics(self, diagnostics):
        score, _ = Score.objects.get_or_create(music_project=self.project)
        score_data = dict(score.data or {})
        score_data['pipeline_diagnostics'] = diagnostics
        score_data['pipeline_log'] = list(self._pipeline_log)
        score.data = score_data
        score.save(update_fields=['data'])

    def run(self):
        failure = None
        self._diagnostics_active = True
        self._diagnostics.start({
            'project_id': self.project.pk,
            'title': self.project.title,
            'task_id': self.project.task_id,
        })
        try:
            self._set_status(MusicProject.Status.ANALYZING, 'Copiando audio e detectando BPM e tonalidade.')
            with self._stage('audio_preparation') as metrics:
                audio_path = self._prepare_audio()
                metrics.update({
                    'file': audio_path.name,
                    'size_bytes': audio_path.stat().st_size,
                })
            rhythm_bpm, rhythm = run_rhythm_analysis(
                audio_path,
                diagnostics=self._diagnostics,
            )
            with self._stage('audio_analysis') as metrics:
                y, sr, bpm, key = self._analyze_audio(audio_path, rhythm_bpm=rhythm_bpm)
                metrics.update({
                    'sample_rate': sr,
                    'audio_samples': len(y),
                    'duration_seconds': round(len(y) / sr, 4),
                    'bpm': bpm,
                    'key': key,
                })
            self._rhythm_grid = rhythm
            self.project.bpm = int(round(bpm)) if bpm else None
            self.project.key = key
            self.project.save(update_fields=['bpm', 'key', 'updated_at'])
            self._log('2_analise_ritmo', rhythm.as_dict())

            self._set_status(MusicProject.Status.SEPARATING, 'Preparando audio para extracao de acordes.')
            harmony_track, y_harmonic, harmony_path, sep_debug = self._separate_tracks(
                audio_path,
                y,
                sr,
            )
            del y

            self._set_status(MusicProject.Status.EXTRACTING_CHORDS, 'Extraindo acordes principais.')
            chord_segments, chord_debug = self._extract_chords(
                y_harmonic,
                sr,
                harmony_track,
                bpm=bpm,
                audio_path=harmony_path,
            )
            del y_harmonic

            self._set_status(MusicProject.Status.BUILDING_SCORE, 'Montando MusicXML e PDF.')
            self._build_score(chord_segments)

            self.project.status = MusicProject.Status.DONE
            self.project.processing_step = 'Pipeline concluido. PDF pronto para revisao/exportacao.'
            self.project.completed_at = timezone.now()
            self.project.error_message = ''
            self.project.save(update_fields=['status', 'processing_step', 'completed_at', 'error_message', 'updated_at'])
        except Exception as exc:
            failure = exc
            self.project.status = MusicProject.Status.ERROR
            self.project.error_message = str(exc)
            self.project.processing_step = 'Erro durante o processamento.'
            self.project.save(update_fields=['status', 'error_message', 'processing_step', 'updated_at'])
            raise
        finally:
            try:
                with self._stage('temporary_cleanup') as metrics:
                    metrics['workdir'] = str(self.workdir)
                    self._cleanup_workdir()
                    metrics['removed'] = not self.workdir.exists()
            finally:
                self._diagnostics.finish(
                    'failed' if failure else 'completed',
                    error=failure,
                )
                self._diagnostics_active = False

    def _set_status(self, status, step):
        self.project.status = status
        self.project.processing_step = step
        self.project.save(update_fields=['status', 'processing_step', 'updated_at'])

    def _cleanup_workdir(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _prepare_audio(self):
        """Copy the uploaded audio file to the temp work directory for processing."""
        if not self.project.audio_file:
            raise RuntimeError('Nenhum arquivo de audio encontrado no projeto.')
        src_name = Path(self.project.audio_file.name).name
        dst = self.workdir / src_name
        with self.project.audio_file.open('rb') as src:
            dst.write_bytes(src.read())
        self._log('1_audio_recebido', {
            'arquivo': src_name,
            'tamanho_bytes': dst.stat().st_size,
        })
        return dst

    def _analyze_audio(self, audio_path, rhythm_bpm=None):
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
        if rhythm_bpm and rhythm_bpm > 0:
            bpm = float(rhythm_bpm)
            beats = []
        else:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(np.atleast_1d(tempo)[0])
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key, key_confidence = self._estimate_key_with_confidence(chroma)

        # Top-3 dominant notes by chroma energy
        chroma_mean = chroma.mean(axis=1)
        top_notes = sorted(
            [(NOTE_NAMES[i], round(float(chroma_mean[i]), 4)) for i in range(12)],
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        self._log('2_analise_audio', {
            'duracao_segundos': round(duration, 2),
            'sample_rate': sr,
            'bpm_detectado': round(bpm, 2),
            'tom_detectado': key,
            'confianca_tom': round(key_confidence, 4),
            'notas_dominantes': [{'nota': n, 'energia': e} for n, e in top_notes],
            'total_beats': int(len(beats)),
        })
        return y, sr, bpm, key

    def _estimate_key(self, chroma):
        key, _ = self._estimate_key_with_confidence(chroma)
        return key

    def _estimate_key_with_confidence(self, chroma):
        import numpy as np

        chroma_mean = chroma.mean(axis=1)
        if chroma_mean.sum() == 0:
            return '', 0.0
        chroma_mean = chroma_mean / chroma_mean.sum()

        best_score = -math.inf
        best_key = ''
        major = np.array(MAJOR_PROFILE) / np.sum(MAJOR_PROFILE)
        minor = np.array(MINOR_PROFILE) / np.sum(MINOR_PROFILE)

        for idx, note in enumerate(NOTE_NAMES):
            major_score = float(np.corrcoef(chroma_mean, np.roll(major, idx))[0, 1])
            minor_score = float(np.corrcoef(chroma_mean, np.roll(minor, idx))[0, 1])
            if major_score > best_score:
                best_score = major_score
                best_key = note
            if minor_score > best_score:
                best_score = minor_score
                best_key = f'{note}m'
        return best_key, best_score

    def _separate_tracks(self, audio_path, y, sr):
        """Choose the harmonic audio used by Chordino."""
        if settings.HARMIX_HARMONY_SOURCE == 'original':
            return self._use_original_audio_for_chords(audio_path, y)
        if settings.HARMIX_HARMONY_SOURCE == 'demucs':
            try:
                return self._separate_tracks_demucs(audio_path, sr)
            except SeparationError as exc:
                logger.warning('Demucs falhou; usando HPSS: %s', exc)
                return self._separate_tracks_hpss_fallback(y, sr, str(exc))
        if settings.HARMIX_HARMONY_SOURCE != 'hpss':
            raise ValueError('HARMIX_HARMONY_SOURCE deve ser original, hpss ou demucs.')
        try:
            return self._separate_tracks_hpss_fallback(y, sr, 'fonte_principal')
        except SeparationError as exc:
            logger.warning('HPSS sem energia harmonica suficiente; usando Demucs: %s', exc)
            return self._separate_tracks_demucs(audio_path, sr)

    def _use_original_audio_for_chords(self, audio_path, y):
        """Keep the uploaded mix intact when it is the Chordino source."""
        audio_path = Path(audio_path)
        with self._stage('original_audio_source', {'audio_file': audio_path.name}) as stage_metrics:
            if not audio_path.exists():
                raise SeparationError('Arquivo de audio original nao encontrado para extracao de acordes.')
            self._delete_existing_tracks()
            track = self._persist_track(
                'Audio original (para cifra)',
                audio_path,
                audio_path.name,
            )
            stage_metrics.update({
                'size_bytes': audio_path.stat().st_size,
                'audio_preserved_without_separation': True,
            })

        debug = {
            'metodo': 'audio_original',
            'stems_gerados': ['Audio original (para cifra)'],
            'stem_escolhido_chordino': 'audio_original',
            'audio_preservado_sem_separacao': True,
        }
        self._log('3_separacao_harmonica', debug)
        return track, y, audio_path, debug

    def _separate_tracks_demucs(self, audio_path, sr):
        import librosa
        import soundfile as sf

        self._delete_existing_tracks()
        with self._stage('demucs_separation', {'audio_file': audio_path.name}) as stage_metrics:
            stems = run_demucs(audio_path, self.workdir / 'demucs')
            stage_metrics['stems'] = {
                name: {
                    'path': str(path),
                    'size_bytes': Path(path).stat().st_size,
                }
                for name, path in stems.items()
            }
        stem_labels = {
            'vocals': ('Vozes', 'vocals.wav'),
            'drums': ('Bateria', 'drums.wav'),
            'bass': ('Baixo', 'bass.wav'),
            'other': ('Acompanhamento', 'other.wav'),
        }

        generated_tracks = {}
        metrics = {}
        with self._stage('stem_analysis_persistence') as stage_metrics:
            for stem_name, (label, filename) in stem_labels.items():
                stem_path = stems[stem_name]
                audio, _ = librosa.load(stem_path, sr=sr, mono=True)
                metrics[label] = stem_metrics(audio)
                generated_tracks[label] = self._persist_track(label, stem_path, filename)
            stage_metrics.update({
                'total_stems': len(generated_tracks),
                'stem_metrics': metrics,
            })

        with self._stage('harmonic_mix_selection') as stage_metrics:
            other_audio, _ = librosa.load(stems['other'], sr=sr, mono=True)
            bass_audio, _ = librosa.load(stems['bass'], sr=sr, mono=True)
            chord_audio, choice_debug = select_chord_source(other_audio, bass_audio)
            chord_path = self.workdir / 'harmony_for_chords.wav'
            sf.write(chord_path, chord_audio, sr)
            harmony_track = self._persist_track(
                'Harmonia para cifra',
                chord_path,
                'harmony_for_chords.wav',
            )
            metrics['Harmonia para cifra'] = stem_metrics(chord_audio)
            stage_metrics.update({
                **choice_debug,
                'output_size_bytes': chord_path.stat().st_size,
                'output_metrics': metrics['Harmonia para cifra'],
            })

        debug = {
            'metodo': 'demucs_htdemucs_cpu',
            'stems_gerados': list(generated_tracks) + ['Harmonia para cifra'],
            'metricas_stems': metrics,
            **choice_debug,
            'fallback_hpss': False,
        }
        self._log('3_separacao_harmonica', debug)
        return harmony_track, chord_audio, chord_path, debug

    def _separate_tracks_hpss_fallback(self, y, sr, failure_reason):
        import librosa
        import numpy as np
        import soundfile as sf

        with self._stage('hpss_fallback', {'reason': failure_reason}) as stage_metrics:
            try:
                harmonic, percussive = librosa.effects.hpss(y)
            except Exception as exc:
                raise SeparationError(f'HPSS falhou: {exc}') from exc
            harmonic_metrics = stem_metrics(harmonic)
            original_energy = float(np.mean(y ** 2)) if len(y) else 0.0
            harmonic_energy = float(np.mean(harmonic ** 2)) if len(harmonic) else 0.0
            energy_ratio = harmonic_energy / original_energy if original_energy > 0 else 0.0
            stage_metrics.update({
                'harmonic_metrics': harmonic_metrics,
                'harmonic_energy_ratio': round(energy_ratio, 4),
            })
            if harmonic_metrics['rms'] < 0.003 or energy_ratio < 0.08:
                raise SeparationError('HPSS gerou uma fonte harmonica fraca.')
            self._delete_existing_tracks()
            chord_path = self.workdir / 'harmony_for_chords.wav'
            percussion_path = self.workdir / 'percussion_hpss.wav'
            sf.write(chord_path, harmonic, sr)
            sf.write(percussion_path, percussive, sr)
            harmony_track = self._persist_track(
                'Harmonia para cifra',
                chord_path,
                'harmony_for_chords.wav',
            )
            self._persist_track(
                'Percussao (HPSS)',
                percussion_path,
                'percussion_hpss.wav',
            )
            stage_metrics.update({
                'audio_samples': len(y),
                'harmonic_size_bytes': chord_path.stat().st_size,
                'percussive_size_bytes': percussion_path.stat().st_size,
            })

        debug = {
            'metodo': 'hpss_harmonico',
            'fallback_hpss': failure_reason != 'fonte_principal',
            'motivo_fallback': failure_reason,
            'stems_gerados': ['Harmonia para cifra', 'Percussao (HPSS)'],
            'energia_harmonica_ratio': round(energy_ratio, 4),
            'energia_percussiva_ratio': round(
                float(np.mean(percussive ** 2)) / original_energy if original_energy else 0.0,
                4,
            ),
            'metricas_stems': {'Harmonia para cifra': harmonic_metrics},
            'stem_escolhido_chordino': 'harmonia_hpss',
        }
        self._log('3_separacao_harmonica', debug)
        return harmony_track, harmonic, chord_path, debug

    def _delete_existing_tracks(self):
        for track in self.project.tracks.all():
            if track.audio_file:
                try:
                    track.audio_file.delete(save=False)
                except Exception:
                    pass
        self.project.tracks.all().delete()

    def _persist_track(self, label, source_path, filename):
        track = Track(music_project=self.project, instrument_name=label)
        with Path(source_path).open('rb') as file_obj:
            track.audio_file.save(
                f'project_{self.project.pk}_{filename}',
                File(file_obj),
                save=True,
            )
        return track

    def _chord_analysis_params(self, duration, bpm):
        beat_sec = (60.0 / bpm) if bpm and bpm > 0 else 1.0
        window = max(0.5, min(beat_sec, MAX_CHORD_WINDOW_SEC))
        hop = max(window / 2.0, MIN_CHORD_HOP_SEC)
        if duration > 0 and hop > 0:
            estimated_windows = math.ceil(duration / hop)
            if estimated_windows > MAX_CHORD_WINDOWS:
                hop = duration / MAX_CHORD_WINDOWS
        return window, hop

    def _extract_chords(self, y, sr, track, bpm=None, audio_path=None):
        import librosa

        if track is None:
            raise RuntimeError('Faixa harmonica nao encontrada para extracao de acordes.')

        duration = float(librosa.get_duration(y=y, sr=sr))
        chord_source_path = audio_path if audio_path and Path(audio_path).exists() else None

        def librosa_fallback():
            raw = self._collect_librosa_chord_detections(y, sr, track)
            merged = self._merge_chord_detections(raw)
            return [
                {
                    'start_time': item['start_time'],
                    'end_time': item['end_time'],
                    'chord_name': item['chord_name'],
                    'confidence': item.get('confidence', 0.0),
                }
                for item in merged
            ]

        if chord_source_path:
            chord_analysis = run_chord_analysis(
                Path(chord_source_path),
                duration,
                librosa_fallback,
                diagnostics=self._diagnostics if self._diagnostics_active else None,
            )
            raw_detections = [
                {
                    'start': event.start_time,
                    'end': event.end_time,
                    'chord': event.chord_name,
                    'confidence': event.confidence,
                }
                for event in chord_analysis.chords.chords
            ]
            chord_source = chord_analysis.chords.source
            chord_errors = chord_analysis.chords.errors
        else:
            raw = self._collect_librosa_chord_detections(y, sr, track)
            raw_detections = raw
            chord_source = 'librosa'
            chord_errors = ['harmony_path_unavailable']

        detected_count = len(raw_detections)
        raw_snapshot = [dict(item) for item in raw_detections]
        if chord_source == 'chordino':
            with self._stage(
                'chord_stabilization',
                {'source': chord_source, 'input_count': detected_count},
            ) as metrics:
                raw_detections = stabilize_chord_detections(raw_detections, bpm=bpm)
                metrics.update({
                    'output_count': len(raw_detections),
                    'removed_or_merged': detected_count - len(raw_detections),
                    'before_sample': raw_snapshot[:10],
                    'after_sample': raw_detections[:10],
                })

        stabilized_snapshot = [dict(item) for item in raw_detections]

        with self._stage(
            'chord_quantization',
            {
                'input_count': len(raw_detections),
                'has_rhythm_grid': bool(self._rhythm_grid),
                'bpm': bpm,
            },
        ) as metrics:
            if self._rhythm_grid and bpm:
                quantization_method = 'rhythm_grid'
                merged = quantize_detections(
                    raw_detections,
                    self._rhythm_grid,
                    self._merge_chord_detections,
                )
            elif bpm:
                quantization_method = 'bpm_uniform_grid'
                merged = self._beat_quantize(raw_detections, bpm)
            else:
                quantization_method = 'merge_only'
                merged = self._merge_chord_detections(raw_detections)
            metrics.update({
                'method': quantization_method,
                'output_count': len(merged),
                'before_sample': stabilized_snapshot[:10],
                'after_sample': merged[:10],
            })

        with self._stage('chord_segment_persistence') as metrics:
            deleted_count, _ = ChordSegment.objects.filter(track=track).delete()
            segments = ChordSegment.objects.bulk_create([
                ChordSegment(
                    track=track,
                    start_time=det['start_time'],
                    end_time=det['end_time'],
                    chord_name=det['chord_name'],
                )
                for det in merged
            ])
            metrics.update({
                'deleted_count': deleted_count,
                'created_count': len(segments),
            })

        unique_chords = sorted({d['chord'] for d in raw_detections})
        inversions_found = [d['chord'] for d in raw_detections if '/' in d['chord']]
        debug = {
            'fonte_acordes': chord_source,
            'erros_fonte_acordes': chord_errors,
            'duracao_segundos': round(duration, 2),
            'bpm_usado': bpm,
            'total_deteccoes_brutas': detected_count,
            'total_apos_estabilizacao_harmonica': len(raw_detections),
            'total_apos_quantizacao_beat': len(merged),
            'acordes_unicos': unique_chords,
            'inversoes_detectadas': inversions_found[:10],
            'amostra_deteccoes_brutas': raw_snapshot[:10],
            'amostra_apos_estabilizacao': stabilized_snapshot[:10],
            'amostra_apos_quantizacao': merged[:10],
        }
        self._log('4_extracao_acordes', debug)
        return segments, debug

    def _collect_librosa_chord_detections(self, y, sr, track):
        import librosa
        import numpy as np

        duration = float(librosa.get_duration(y=y, sr=sr))
        window, hop = self._chord_analysis_params(duration, self.project.bpm)
        starts = list(np.arange(0, max(duration, window), hop))
        detect_bass = len(starts) <= 180
        raw_detections = []

        for index, start in enumerate(starts):
            end = min(float(start + window), duration)
            if end - start < 0.25:
                continue
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chunk = y[start_sample:end_sample]
            if len(chunk) < int(sr * 0.25):
                continue

            chroma = librosa.feature.chroma_cqt(y=chunk, sr=sr)
            chord_name, confidence = self._estimate_chord_with_confidence(chroma)

            if detect_bass:
                bass_note = self._detect_bass_note(chunk, sr)
                chord_name = self._maybe_add_inversion(chord_name, bass_note)

            raw_detections.append({
                'start': round(float(start), 2),
                'end': round(float(end), 2),
                'chord': chord_name,
                'confidence': round(confidence, 4),
            })

            if index > 0 and index % 40 == 0:
                pct = min(99, int(100 * index / len(starts)))
                if self._diagnostics_active:
                    self._diagnostics.event(
                        'fallback_chord_progress',
                        stage='librosa_chord_fallback',
                        details={
                            'completed_windows': index,
                            'total_windows': len(starts),
                            'percent': pct,
                        },
                    )
                self._set_status(
                    MusicProject.Status.EXTRACTING_CHORDS,
                    f'Extraindo acordes principais ({pct}%).',
                )

        return raw_detections

    def _detect_bass_note(self, chunk, sr):
        """
        Detect the dominant bass note by looking at the spectral peak
        in the bass frequency range (40ÔÇô300 Hz) using FFT.
        Returns a note name string (e.g. 'E') or None on failure.
        """
        import librosa
        import numpy as np

        if len(chunk) < int(sr * 0.1):
            return None
        try:
            magnitudes = np.abs(librosa.stft(chunk))
            freqs = librosa.fft_frequencies(sr=sr)
            bass_mask = (freqs >= 40) & (freqs <= 300)
            if not bass_mask.any():
                return None
            bass_energy = magnitudes[bass_mask, :].mean(axis=1)
            peak_freq = float(freqs[bass_mask][np.argmax(bass_energy)])
            if peak_freq < 40:
                return None
            midi = int(round(librosa.hz_to_midi(peak_freq)))
            return NOTE_NAMES[midi % 12]
        except Exception:
            return None

    def _maybe_add_inversion(self, chord_name, bass_note):
        """
        If the detected bass note differs from the chord root AND is a valid
        chord tone, append '/bass' to create an inversion label (e.g. C/E).
        """
        if not bass_note or chord_name in ('N.C.', ''):
            return chord_name
        # Determine root (e.g. 'C' from 'Cm7', 'A#' from 'A#maj7')
        if len(chord_name) >= 2 and chord_name[1] == '#':
            root = chord_name[:2]
            suffix = chord_name[2:]
        else:
            root = chord_name[0]
            suffix = chord_name[1:]
        if root not in NOTE_NAMES or bass_note == root:
            return chord_name
        root_idx = NOTE_NAMES.index(root)
        bass_idx = NOTE_NAMES.index(bass_note) if bass_note in NOTE_NAMES else -1
        if bass_idx < 0:
            return chord_name
        # Find chord template for this suffix
        for intervals, tmpl_suffix, _ in CHORD_TEMPLATES:
            if tmpl_suffix == suffix:
                chord_tones = {(root_idx + i) % 12 for i in intervals}
                if bass_idx in chord_tones:
                    return f'{chord_name}/{bass_note}'
                break
        return chord_name

    def _beat_quantize(self, detections, bpm):
        """
        Snap every chord-change boundary to the nearest beat boundary.
        This eliminates spurious half-beat chord oscillations and aligns
        the output to the actual rhythmic grid of the song.
        """
        if not bpm or bpm <= 0 or not detections:
            return detections
        beat_sec = 60.0 / bpm

        def snap(t):
            return round(round(t / beat_sec) * beat_sec, 3)

        quantized = []
        for det in detections:
            q_start = snap(det['start'])
            q_end = snap(det['end'])
            if q_end <= q_start:
                q_end = q_start + beat_sec
            quantized.append({
                'start': q_start,
                'end': q_end,
                'chord': det['chord'],
                'confidence': det['confidence'],
            })
        # Re-merge after snapping (nearby windows may now have identical boundaries)
        return self._merge_chord_detections(quantized)

    def _merge_chord_detections(self, detections):
        if not detections:
            return []
        merged = []
        current = {
            'start_time': detections[0]['start'],
            'end_time': detections[0]['end'],
            'chord_name': detections[0]['chord'],
            'best_confidence': detections[0]['confidence'],
        }
        for det in detections[1:]:
            if det['chord'] == current['chord_name']:
                current['end_time'] = det['end']
                if det['confidence'] > current['best_confidence']:
                    current['best_confidence'] = det['confidence']
            else:
                merged.append({
                    'start_time': round(current['start_time'], 2),
                    'end_time': round(current['end_time'], 2),
                    'chord_name': current['chord_name'],
                })
                current = {
                    'start_time': det['start'],
                    'end_time': det['end'],
                    'chord_name': det['chord'],
                    'best_confidence': det['confidence'],
                }
        merged.append({
            'start_time': round(current['start_time'], 2),
            'end_time': round(current['end_time'], 2),
            'chord_name': current['chord_name'],
        })
        return merged

    def _estimate_chord(self, chroma):
        name, _ = self._estimate_chord_with_confidence(chroma)
        return name

    def _estimate_chord_with_confidence(self, chroma):
        import numpy as np

        chroma_mean = chroma.mean(axis=1)
        if chroma_mean.sum() == 0:
            return 'N.C.', 0.0
        chroma_mean = chroma_mean / chroma_mean.sum()

        best_score = -math.inf
        best_chord = 'N.C.'

        for intervals, suffix, forbidden in CHORD_TEMPLATES:
            template = np.zeros(12)
            for interval in intervals:
                template[interval % 12] = 1.0
            template = template / template.sum()

            for idx, note in enumerate(NOTE_NAMES):
                rolled = np.roll(template, idx)
                score = float(np.dot(chroma_mean, rolled))

                # Penalise forbidden notes: if the 3rd is present in the chroma,
                # sus chords lose score proportionally.
                for f_interval in forbidden:
                    forbidden_idx = (idx + f_interval) % 12
                    score -= chroma_mean[forbidden_idx] * _FORBIDDEN_PENALTY

                if score > best_score:
                    best_score = score
                    best_chord = f'{note}{suffix}'

        return best_chord, best_score

    def _build_score(self, chord_segments):
        chords = [
            {
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'chord_name': segment.chord_name,
            }
            for segment in chord_segments
        ]
        with self._stage(
            'musical_structure',
            {'input_chords': len(chords), 'bpm': self.project.bpm},
        ) as metrics:
            structure = build_musical_structure(
                chords,
                self.project.bpm,
                rhythm_grid=self._rhythm_grid.as_dict() if self._rhythm_grid else None,
            ) if self.project.bpm else {}
            metrics.update({
                'measure_count': structure.get('measure_count', 0),
                'output_chords': sum(
                    len(measure.get('chords') or [])
                    for measure in structure.get('measures') or []
                ),
                'rhythm_grid_source': (
                    self._rhythm_grid.source if self._rhythm_grid else None
                ),
            })
        self._log('5_geracao_score', {
            'total_acordes': len(chords),
            'total_compassos': structure.get('measure_count', 0),
            'bpm_final': self.project.bpm,
            'tom_final': self.project.key,
            'amostra_acordes': chords[:8],
        })
        score_data = {
            'title': self.project.title,
            'artist': (self.project.artist or PDF_DEFAULT_ARTIST).strip()[:80],
            'key': self.project.key,
            'bpm': self.project.bpm,
            'chords': chords,
            'structure': structure,
            'pipeline_log': self._pipeline_log,
            'pipeline_diagnostics': self._diagnostics.as_dict(),
        }
        score, _ = Score.objects.get_or_create(music_project=self.project)
        score.data = score_data

        with self._stage('musicxml_generation') as metrics:
            musicxml = self._make_musicxml(score_data)
            musicxml_bytes = musicxml.encode('utf-8')
            score.musicxml_file.save(
                f'project_{self.project.pk}.musicxml',
                ContentFile(musicxml_bytes),
                save=False,
            )
            metrics.update({
                'size_bytes': len(musicxml_bytes),
                'measure_count': structure.get('measure_count', 0),
            })

        with self._stage('pdf_generation') as metrics:
            pdf_bytes = self._make_pdf(score_data)
            score.pdf_file.save(
                f'project_{self.project.pk}.pdf',
                ContentFile(pdf_bytes),
                save=False,
            )
            metrics['size_bytes'] = len(pdf_bytes)

        with self._stage('score_persistence') as metrics:
            score.data['pipeline_diagnostics'] = self._diagnostics.as_dict()
            score.save()
            metrics.update({
                'score_id': score.pk,
                'musicxml_file': score.musicxml_file.name,
                'pdf_file': score.pdf_file.name,
            })

    def _make_musicxml(self, score_data):
        measures = []
        structured_measures = score_data.get('structure', {}).get('measures') or []
        if structured_measures:
            for measure in structured_measures:
                measures.append(self._make_musicxml_measure(measure))
        else:
            for index, chord in enumerate(score_data['chords'], start=1):
                measures.append(
                    self._make_musicxml_measure(
                        {
                            'number': index,
                            'chords': [
                                {
                                    'chord_name': chord['chord_name'],
                                    'beat_duration': 4,
                                }
                            ],
                        }
                    )
                )

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <work><work-title>{html.escape(score_data['title'])}</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Harmix Lead Sheet</part-name></score-part>
  </part-list>
  <part id="P1">
    {''.join(measures)}
  </part>
</score-partwise>
"""

    def _make_musicxml_measure(self, measure):
        chord_blocks = []
        beats_per_measure = 4
        chords = sorted(
            measure.get('chords') or [],
            key=lambda chord: float(chord.get('beat_start') or 0),
        )
        cursor = 0
        for index, chord in enumerate(chords):
            beat_start = max(0, min(
                beats_per_measure,
                float(chord.get('beat_start') or 0),
            ))
            if beat_start > cursor:
                chord_blocks.append(self._musicxml_rest(beat_start - cursor))

            next_start = (
                float(chords[index + 1].get('beat_start') or beats_per_measure)
                if index + 1 < len(chords)
                else beats_per_measure
            )
            available = max(0.0, min(next_start, beats_per_measure) - beat_start)
            duration = min(
                max(0.0, float(chord.get('beat_duration') or 1)),
                available,
            )
            if duration <= 0:
                continue

            root, alter, kind = self._split_chord(chord['chord_name'])
            alter_xml = f'<root-alter>{alter}</root-alter>' if alter else ''
            chord_blocks.append(
                f"""
          <harmony>
            <root><root-step>{html.escape(root)}</root-step>{alter_xml}</root>
            <kind>{html.escape(kind)}</kind>
          </harmony>
          {self._musicxml_rest(duration)}"""
            )
            cursor = beat_start + duration

        if cursor < beats_per_measure:
            chord_blocks.append(self._musicxml_rest(beats_per_measure - cursor))

        attributes = ''
        if int(measure.get('number', 1)) == 1:
            attributes = """
          <attributes>
            <divisions>1</divisions>
            <time><beats>4</beats><beat-type>4</beat-type></time>
          </attributes>"""

        return f"""
        <measure number="{measure.get('number', 1)}">{attributes}{''.join(chord_blocks)}
        </measure>"""

    def _musicxml_rest(self, beat_duration):
        duration = max(1, int(round(float(beat_duration))))
        return f'<note><rest/><duration>{duration}</duration><type>quarter</type></note>'

    _MUSICXML_KIND = {
        '':      'major',
        'm':     'minor',
        '7':     'dominant',
        'maj7':  'major-seventh',
        'm7':    'minor-seventh',
        'mM7':   'major-minor',
        'dim':   'diminished',
        'dim7':  'diminished-seventh',
        'm7b5':  'half-diminished',
        'aug':   'augmented',
        'sus4':  'suspended-fourth',
        'sus2':  'suspended-second',
        '6':     'major-sixth',
        'm6':    'minor-sixth',
    }

    def _split_chord(self, chord):
        """Parse a chord label into (root_step, alter, musicxml_kind).
        Handles inversions (C/E) by using only the part before '/'.
        """
        if chord == 'N.C.':
            return 'C', 0, 'none'
        base = chord.split('/')[0]  # ignore bass note for MusicXML root
        # Determine root (1 or 2 chars)
        if len(base) >= 2 and base[1] == '#':
            root = base[:2]
            suffix = base[2:]
        else:
            root = base[0]
            suffix = base[1:]
        alter = 1 if '#' in root else 0
        root_step = root.replace('#', '')
        kind = self._MUSICXML_KIND.get(suffix, 'major')
        return root_step, alter, kind

    def _make_pdf(self, score_data):
        from io import BytesIO

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        measures = score_data.get('structure', {}).get('measures') or self._fallback_pdf_measures(score_data)
        beats_per_measure = (
            score_data.get('structure', {})
            .get('time_signature', {})
            .get('beats_per_measure', 4)
        )
        left = 1.7 * cm
        right = width - 1.4 * cm
        top = height - 1.8 * cm
        staff_width = right - left
        measures_per_system = 4
        system_gap = 2.55 * cm
        staff_line_gap = 0.14 * cm

        pdf.setTitle(score_data.get('title') or 'Harmix Score')
        self._draw_pdf_header(pdf, score_data, width, top)

        y = top - 2.25 * cm
        for index in range(0, len(measures), measures_per_system):
            if y < 3.2 * cm:
                pdf.showPage()
                self._draw_pdf_header(pdf, score_data, width, top)
                y = top - 2.25 * cm

            system_measures = measures[index:index + measures_per_system]
            self._draw_pdf_system(
                pdf=pdf,
                measures=system_measures,
                x=left,
                y=y,
                width=staff_width,
                line_gap=staff_line_gap,
                beats_per_measure=beats_per_measure,
                is_first_system=index == 0,
            )
            y -= system_gap

        pdf.save()
        buffer.seek(0)
        return buffer.getvalue()

    def _fallback_pdf_measures(self, score_data):
        return [
            {
                'number': index,
                'chords': [
                    {
                        'chord_name': chord['chord_name'],
                        'beat_start': 0,
                        'beat_duration': 4,
                    }
                ],
            }
            for index, chord in enumerate(score_data.get('chords', []), start=1)
        ]

    def _draw_pdf_header(self, pdf, score_data, width, y):
        from reportlab.lib.units import cm

        title = (score_data.get('title') or 'Harmix Score')[:80]
        key = score_data.get('key') or 'N/A'
        bpm = score_data.get('bpm') or 'N/A'
        artist = (score_data.get('artist') or PDF_DEFAULT_ARTIST).strip()[:80]
        side_margin = 1.4 * cm

        pdf.setFont('Helvetica', 10)
        pdf.drawString(side_margin, y, f'Tom: {key}   BPM: {bpm}')

        pdf.setFont('Helvetica-Bold', 24)
        pdf.drawCentredString(width / 2, y, title)

        pdf.setFont('Helvetica', 10)
        pdf.drawRightString(width - side_margin, y, artist)

    def _draw_pdf_system(self, pdf, measures, x, y, width, line_gap, beats_per_measure=4, is_first_system=False):
        from reportlab.lib.units import cm

        staff_height = line_gap * 4
        measure_width = width / max(len(measures), 1)
        staff_top = y
        staff_bottom = y - staff_height

        pdf.setStrokeColorRGB(0.08, 0.08, 0.08)
        pdf.setLineWidth(0.7)
        for line in range(5):
            line_y = staff_top - (line * line_gap)
            pdf.line(x, line_y, x + width, line_y)

        for idx in range(len(measures) + 1):
            bar_x = x + (idx * measure_width)
            pdf.line(bar_x, staff_top, bar_x, staff_bottom)

        signature_end = draw_system_signature(
            pdf,
            x,
            staff_top,
            line_gap,
            include_time_signature=is_first_system,
        )
        first_measure_padding = max(signature_end - x, 0.55 * cm if is_first_system else 0.42 * cm)

        for idx, measure in enumerate(measures):
            measure_x = x + (idx * measure_width)
            self._draw_pdf_measure_content(
                pdf=pdf,
                measure=measure,
                x=measure_x,
                y=staff_top,
                width=measure_width,
                staff_height=staff_height,
                beats_per_measure=beats_per_measure,
                left_padding=(first_measure_padding if idx == 0 else 0),
            )

    def _pdf_measure_play_area(self, x, width, left_padding):
        from reportlab.lib.units import cm

        bar_margin = 0.14 * cm
        play_left = x + left_padding + bar_margin
        play_right = x + width - bar_margin
        play_width = max(play_right - play_left, 0.6 * cm)
        return play_left, play_width

    def _pdf_chord_center_x(self, play_left, play_width, beat_start, beat_duration, beats_per_measure):
        beat_start = max(0.0, min(float(beat_start or 0), beats_per_measure))
        duration = max(0.0, float(beat_duration or beats_per_measure))
        beat_end = min(beat_start + duration, beats_per_measure)
        if beat_end <= beat_start:
            beat_end = min(beat_start + 1.0, beats_per_measure)
        center_beat = beat_start + ((beat_end - beat_start) / 2.0)
        return play_left + (center_beat / beats_per_measure) * play_width

    def _pdf_clamp_chord_x(self, pdf, chord_x, chord_name, play_left, play_width):
        from reportlab.pdfbase import pdfmetrics

        text_width = pdfmetrics.stringWidth(chord_name, 'Helvetica-Bold', 10)
        half_width = text_width / 2
        min_x = play_left + half_width
        max_x = play_left + play_width - half_width
        if min_x > max_x:
            return play_left + (play_width / 2)
        return max(min(chord_x, max_x), min_x)

    def _draw_pdf_measure_content(self, pdf, measure, x, y, width, staff_height, beats_per_measure=4, left_padding=0):
        from reportlab.lib.units import cm

        pdf.setFont('Helvetica', 7)
        pdf.drawString(x + 0.08 * cm, y + 0.12 * cm, str(measure.get('number', '')))

        chords = measure.get('chords') or [{'chord_name': '%', 'beat_start': 0, 'beat_duration': 4}]
        play_left, play_width = self._pdf_measure_play_area(x, width, left_padding)

        for chord in chords:
            chord_name = self._pdf_chord_name(chord.get('chord_name') or 'N.C.')
            chord_x = self._pdf_chord_center_x(
                play_left,
                play_width,
                chord.get('beat_start'),
                chord.get('beat_duration'),
                beats_per_measure,
            )
            chord_x = self._pdf_clamp_chord_x(pdf, chord_x, chord_name, play_left, play_width)
            pdf.setFont('Helvetica-Bold', 10)
            pdf.drawCentredString(chord_x, y + 0.13 * cm, chord_name)

        slash_center_y = y - (staff_height / 2)
        slash_count = max(1, int(beats_per_measure or 4))
        for slash_x in self._pdf_slash_positions(play_left, play_width, slash_count):
            self._draw_pdf_rhythm_slash(pdf, slash_x, slash_center_y, staff_height)

    def _pdf_slash_positions(self, play_left, play_width, slash_count):
        return [
            play_left + ((index + 0.5) * play_width / slash_count)
            for index in range(slash_count)
        ]

    def _draw_pdf_rhythm_slash(self, pdf, x, center_y, staff_height):
        half_height = staff_height * 0.22
        half_length = half_height * 0.82

        pdf.setLineWidth(1.85)
        pdf.setLineCap(1)
        pdf.setStrokeColorRGB(0.02, 0.02, 0.02)
        pdf.line(x - half_length, center_y - half_height, x + half_length, center_y + half_height)

    def _pdf_chord_name(self, chord_name):
        return str(chord_name).replace('#', '#').replace('m', 'm')
