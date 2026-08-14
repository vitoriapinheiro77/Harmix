from unittest.mock import MagicMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from .analysis.chordino_labels import chord_changes_to_events, normalize_chordino_label
from .analysis.bpm_detector import _madmom_primary_bpm
from .analysis.chord_stability import harmonic_family, stabilize_chord_detections
from .models import MusicProject, Track
from .pipeline import HarmixPipeline
from .separation import SeparationError, run_demucs, select_chord_source
from .structure import MusicalStructureBuilder, build_musical_structure
from .utils import enqueue_project_reprocessing


class _ChordChangeStub:
    """Espelha chord_extractor.outputs.ChordChange (NamedTuple: chord, timestamp)."""

    __slots__ = ('chord', 'timestamp')

    def __init__(self, chord, timestamp):
        self.chord = chord
        self.timestamp = timestamp


class MusicalStructureBuilderTests(TestCase):
    def test_seconds_to_beats_uses_bpm(self):
        builder = MusicalStructureBuilder(bpm=120)

        self.assertEqual(builder.seconds_to_beats(1), 2)
        self.assertEqual(builder.seconds_to_beats(2), 4)

    def test_chords_are_grouped_inside_four_four_measures(self):
        structure = build_musical_structure(
            [
                {'start_time': 0, 'end_time': 2, 'chord_name': 'Dm'},
                {'start_time': 2, 'end_time': 4, 'chord_name': 'C'},
            ],
            bpm=120,
        )

        self.assertEqual(structure['time_signature']['label'], '4/4')
        self.assertEqual(structure['measure_count'], 2)
        self.assertEqual(structure['measures'][0]['chords'][0]['chord_name'], 'Dm')
        self.assertEqual(structure['measures'][1]['chords'][0]['chord_name'], 'C')

    def test_sustained_chord_is_represented_in_each_measure_it_dominates(self):
        structure = build_musical_structure(
            [
                {'start_time': 1, 'end_time': 3, 'chord_name': 'F'},
            ],
            bpm=120,
        )

        first_measure_chord = structure['measures'][0]['chords'][0]
        second_measure_chord = structure['measures'][1]['chords'][0]
        self.assertEqual(first_measure_chord['chord_name'], 'F')
        self.assertEqual(first_measure_chord['beat_start'], 0)
        self.assertEqual(first_measure_chord['beat_duration'], 4)
        self.assertEqual(second_measure_chord['chord_name'], 'F')
        self.assertEqual(second_measure_chord['beat_start'], 0)
        self.assertEqual(second_measure_chord['beat_duration'], 4)

    def test_real_changes_are_placed_in_the_measure_where_they_start(self):
        structure = build_musical_structure(
            [
                {'start_time': 0.0, 'end_time': 2.3, 'chord_name': 'C'},
                {'start_time': 2.3, 'end_time': 4.5, 'chord_name': 'G'},
                {'start_time': 4.5, 'end_time': 6.5, 'chord_name': 'Am'},
                {'start_time': 6.5, 'end_time': 8.5, 'chord_name': 'F'},
            ],
            bpm=120,
        )

        measure_chords = [
            measure['chords'][0]['chord_name']
            for measure in structure['measures'][:4]
            if measure['chords']
        ]
        self.assertEqual(measure_chords, ['C', 'G', 'Am', 'F'])
        for measure in structure['measures'][:4]:
            self.assertEqual(len(measure['chords']), 1)

    def test_multiple_real_changes_inside_measure_are_preserved(self):
        structure = build_musical_structure(
            [
                {'start_time': 0.0, 'end_time': 1.0, 'chord_name': 'C'},
                {'start_time': 1.0, 'end_time': 2.0, 'chord_name': 'G'},
            ],
            bpm=120,
        )

        chords = structure['measures'][0]['chords']
        self.assertEqual([chord['chord_name'] for chord in chords], ['C', 'G'])
        self.assertEqual([chord['beat_start'] for chord in chords], [0, 2])

    def test_mixed_tensions_use_stable_base_triad(self):
        structure = build_musical_structure(
            [
                {'start_time': 0.0, 'end_time': 0.5, 'chord_name': 'Cmaj7'},
                {'start_time': 0.5, 'end_time': 1.5, 'chord_name': 'C'},
                {'start_time': 1.5, 'end_time': 2.0, 'chord_name': 'C7'},
            ],
            bpm=120,
        )

        chords = structure['measures'][0]['chords']
        self.assertEqual(len(chords), 1)
        self.assertEqual(chords[0]['chord_name'], 'C')

    def test_consistently_complex_harmony_is_preserved(self):
        structure = build_musical_structure(
            [
                {'start_time': 0.0, 'end_time': 2.0, 'chord_name': 'Cmaj7'},
            ],
            bpm=120,
        )

        self.assertEqual(
            structure['measures'][0]['chords'][0]['chord_name'],
            'Cmaj7',
        )

    def test_structure_honors_first_downbeat_offset(self):
        structure = build_musical_structure(
            [
                {'start_time': 1.0, 'end_time': 3.0, 'chord_name': 'G'},
            ],
            bpm=120,
            rhythm_grid={'downbeats': [1.0], 'source': 'madmom'},
        )

        first_chord = structure['measures'][0]['chords'][0]
        self.assertEqual(first_chord['beat_start'], 0)
        self.assertEqual(structure['rhythm_grid']['beat_offset_seconds'], 1.0)

    def test_structure_uses_real_beat_grid_instead_of_constant_bpm(self):
        builder = MusicalStructureBuilder(
            bpm=120,
            rhythm_grid={
                'beats': [1.0, 1.5, 2.1, 2.7, 3.3],
                'downbeats': [1.0],
                'source': 'madmom',
            },
        )

        self.assertEqual(builder.seconds_to_beats(2.1), 2.0)
        self.assertEqual(builder.seconds_to_beats(2.7), 3.0)

    def test_chordino_label_normalization(self):
        self.assertEqual(normalize_chordino_label('A:min'), 'Am')
        self.assertEqual(normalize_chordino_label('G:maj7'), 'Gmaj7')
        self.assertEqual(normalize_chordino_label('F:min7/C'), 'Fm7/C')
        self.assertIsNone(normalize_chordino_label('N'))

    def test_madmom_primary_bpm_reads_first_candidate(self):
        import numpy as np

        tempo = np.array([
            [176.47059, 0.47469],
            [117.64706, 0.17667],
        ])
        self.assertEqual(_madmom_primary_bpm(tempo), 176.47059)

    def test_chord_changes_to_events_reads_chordchange_objects(self):
        changes = [
            _ChordChangeStub('A:min', 0.0),
            _ChordChangeStub('D:maj', 2.0),
            _ChordChangeStub('N', 4.0),
        ]
        events = chord_changes_to_events(changes, duration=6.0, source='chordino')

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].chord_name, 'Am')
        self.assertEqual(events[0].start_time, 0.0)
        self.assertEqual(events[0].end_time, 2.0)
        self.assertEqual(events[0].source, 'chordino')
        self.assertEqual(events[1].chord_name, 'D')
        self.assertEqual(events[1].end_time, 4.0)

    def test_no_chord_marker_terminates_previous_chord(self):
        changes = [
            _ChordChangeStub('C:maj', 0.0),
            _ChordChangeStub('N', 2.0),
            _ChordChangeStub('G:maj', 4.0),
        ]

        events = chord_changes_to_events(changes, duration=6.0, source='chordino')

        self.assertEqual(events[0].chord_name, 'C')
        self.assertEqual(events[0].end_time, 2.0)
        self.assertEqual(events[1].chord_name, 'G')
        self.assertEqual(events[1].start_time, 4.0)

    def test_transient_extension_does_not_fragment_stable_harmony(self):
        detections = [
            {'start': 0.0, 'end': 2.0, 'chord': 'C', 'confidence': 0.9},
            {'start': 2.0, 'end': 2.2, 'chord': 'C7', 'confidence': 0.7},
            {'start': 2.2, 'end': 4.0, 'chord': 'C', 'confidence': 0.9},
        ]

        stabilized = stabilize_chord_detections(detections, bpm=120)

        self.assertEqual(len(stabilized), 1)
        self.assertEqual(stabilized[0]['chord'], 'C')
        self.assertEqual(stabilized[0]['end'], 4.0)

    def test_brief_different_chord_between_same_context_is_suppressed(self):
        detections = [
            {'start': 0.0, 'end': 2.0, 'chord': 'Am', 'confidence': 0.9},
            {'start': 2.0, 'end': 2.2, 'chord': 'F', 'confidence': 0.5},
            {'start': 2.2, 'end': 4.0, 'chord': 'Am', 'confidence': 0.9},
        ]

        stabilized = stabilize_chord_detections(detections, bpm=120)

        self.assertEqual([item['chord'] for item in stabilized], ['Am'])

    def test_stable_complex_chord_is_preserved(self):
        detections = [
            {'start': 0.0, 'end': 1.0, 'chord': 'C', 'confidence': 0.9},
            {'start': 1.0, 'end': 4.0, 'chord': 'Cmaj7', 'confidence': 0.9},
        ]

        stabilized = stabilize_chord_detections(detections, bpm=120)

        self.assertEqual([item['chord'] for item in stabilized], ['C', 'Cmaj7'])
        self.assertEqual(harmonic_family('Cmaj9/E'), harmonic_family('Csus4'))


class ProjectReprocessTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            email='dev@harmix.test',
            password='test-pass-123',
        )
        self.project = MusicProject.objects.create(
            user=self.user,
            title='Test Song',
            audio_file='audio/projects/sample.wav',
            status=MusicProject.Status.DONE,
        )

    @patch('music.tasks.process_music_project')
    def test_enqueue_project_reprocessing_resets_status_and_queues_task(self, mock_task):
        mock_result = MagicMock()
        mock_result.id = 'task-123'
        mock_task.delay.return_value = mock_result

        enqueue_project_reprocessing(self.project)
        self.project.refresh_from_db()

        self.assertEqual(self.project.status, MusicProject.Status.QUEUED)
        self.assertEqual(self.project.task_id, 'task-123')
        self.assertEqual(self.project.processing_step, 'Reprocessamento solicitado. Aguardando worker.')
        mock_task.delay.assert_called_once_with(self.project.pk)

    @patch('music.views.enqueue_project_reprocessing')
    def test_project_reprocess_view_requires_post(self, mock_enqueue):
        self.client.force_login(self.user)
        response = self.client.get(reverse('music:project_reprocess', args=[self.project.pk]))
        self.assertEqual(response.status_code, 405)

    @patch('music.views.enqueue_project_reprocessing')
    def test_project_reprocess_view_redirects_to_project_detail(self, mock_enqueue):
        self.client.force_login(self.user)
        response = self.client.post(reverse('music:project_reprocess', args=[self.project.pk]))
        self.assertRedirects(response, reverse('music:project_detail', args=[self.project.pk]))
        mock_enqueue.assert_called_once()

    def test_project_status_view_returns_json(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse('music:project_status', args=[self.project.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], MusicProject.Status.DONE)

    def test_owner_can_stream_original_and_separated_audio(self):
        self.project.audio_file.save(
            'source.wav',
            SimpleUploadedFile('source.wav', b'original-audio', content_type='audio/wav'),
        )
        track = Track.objects.create(
            music_project=self.project,
            instrument_name='Harmonia',
            audio_file=SimpleUploadedFile('harmony.wav', b'harmony-audio', content_type='audio/wav'),
        )
        self.client.force_login(self.user)

        original_response = self.client.get(
            reverse('music:project_original_audio', args=[self.project.pk])
        )
        track_response = self.client.get(
            reverse('music:project_track_audio', args=[self.project.pk, track.pk])
        )

        self.assertEqual(original_response.status_code, 200)
        self.assertEqual(original_response['Content-Type'], 'audio/x-wav')
        self.assertEqual(track_response.status_code, 200)
        self.assertEqual(track_response['Content-Type'], 'audio/x-wav')

    def test_other_user_cannot_stream_project_audio(self):
        other_user = get_user_model().objects.create_user(
            email='other@harmix.test',
            password='test-pass-123',
        )
        self.client.force_login(other_user)

        response = self.client.get(
            reverse('music:project_original_audio', args=[self.project.pk])
        )

        self.assertEqual(response.status_code, 404)

    def test_reprocessing_track_cleanup_removes_previous_stems(self):
        Track.objects.create(
            music_project=self.project,
            instrument_name='Harmonia antiga',
            audio_file=SimpleUploadedFile('old.wav', b'old-audio', content_type='audio/wav'),
        )
        pipeline = HarmixPipeline(self.project)
        try:
            pipeline._delete_existing_tracks()
        finally:
            pipeline._cleanup_workdir()

        self.assertFalse(self.project.tracks.exists())

    @patch.object(HarmixPipeline, '_separate_tracks_hpss_fallback')
    @patch.object(HarmixPipeline, '_separate_tracks_demucs')
    def test_pipeline_uses_hpss_when_demucs_fails(self, mock_demucs, mock_fallback):
        mock_demucs.side_effect = SeparationError('sem memoria')
        mock_fallback.return_value = ('track', 'audio', 'path', {'fallback_hpss': True})
        pipeline = HarmixPipeline(self.project)
        try:
            result = pipeline._separate_tracks('source.wav', [0.0], 22050)
        finally:
            pipeline._cleanup_workdir()

        self.assertEqual(result[3]['fallback_hpss'], True)
        mock_fallback.assert_called_once_with([0.0], 22050, 'sem memoria')


class SeparationTests(TestCase):
    def test_chord_source_mixes_balanced_accompaniment_and_bass(self):
        import numpy as np

        selected, details = select_chord_source(
            np.full(22050, 0.10, dtype=np.float32),
            np.full(22050, 0.08, dtype=np.float32),
        )

        self.assertEqual(details['stem_escolhido_chordino'], 'acompanhamento_com_baixo')
        self.assertEqual(details['ganho_baixo'], 0.45)
        self.assertGreater(float(selected.mean()), 0.10)

    def test_chord_source_avoids_overpowering_bass(self):
        import numpy as np

        selected, details = select_chord_source(
            np.full(22050, 0.01, dtype=np.float32),
            np.full(22050, 0.10, dtype=np.float32),
        )

        self.assertEqual(details['stem_escolhido_chordino'], 'acompanhamento')
        self.assertEqual(details['ganho_baixo'], 0.0)
        self.assertAlmostEqual(float(selected.mean()), 0.01, places=4)

    @patch('music.separation.subprocess.run')
    def test_run_demucs_returns_expected_four_stems(self, mock_run):
        mock_run.return_value = MagicMock(stdout='', stderr='')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / 'song.wav'
            audio_path.write_bytes(b'audio')
            stem_dir = root / 'output' / 'htdemucs' / 'song'
            stem_dir.mkdir(parents=True)
            for stem in ('vocals', 'drums', 'bass', 'other'):
                (stem_dir / f'{stem}.wav').write_bytes(stem.encode())

            stems = run_demucs(audio_path, root / 'output')

        self.assertEqual(set(stems), {'vocals', 'drums', 'bass', 'other'})
        command = mock_run.call_args.args[0]
        self.assertIn('--device', command)
        self.assertIn('cpu', command)
