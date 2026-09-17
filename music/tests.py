from unittest.mock import MagicMock, patch
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from .analysis.chordino_labels import chord_changes_to_events, normalize_chordino_label
from .analysis.bpm_detector import _madmom_primary_bpm
from .analysis.chord_stability import harmonic_family, stabilize_chord_detections
from .models import MusicProject, Score, Track
from .pipeline import HarmixPipeline
from .separation import SeparationError, run_demucs, select_chord_source
from .structure import MusicalStructureBuilder, build_musical_structure
from .musicxml import MUSICXML_DIVISIONS
from .utils import (
    enqueue_project_reprocessing,
    pdf_score_data,
    score_data_with_manual_chord_edits,
)
from .views import CHORD_NAME_PATTERN


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

    def test_next_measure_anticipation_is_not_promoted_to_second_chord(self):
        structure = build_musical_structure(
            [
                {'start_time': 0.0, 'end_time': 4.0, 'chord_name': 'F'},
                {'start_time': 4.0, 'end_time': 8.0, 'chord_name': 'F'},
                {'start_time': 8.0, 'end_time': 10.5, 'chord_name': 'C'},
                {'start_time': 10.5, 'end_time': 16.0, 'chord_name': 'Em7'},
            ],
            bpm=60,
        )

        chord_names = [
            [chord['chord_name'] for chord in measure['chords']]
            for measure in structure['measures']
        ]
        self.assertEqual(chord_names, [['F'], ['F'], ['C'], ['Em7']])

    def test_two_half_measure_changes_are_preserved(self):
        structure = build_musical_structure(
            [
                {'start_time': 0.0, 'end_time': 2.0, 'chord_name': 'C'},
                {'start_time': 2.0, 'end_time': 4.0, 'chord_name': 'G'},
                {'start_time': 4.0, 'end_time': 6.0, 'chord_name': 'Am'},
                {'start_time': 6.0, 'end_time': 8.0, 'chord_name': 'F'},
            ],
            bpm=60,
        )

        chord_names = [
            [chord['chord_name'] for chord in measure['chords']]
            for measure in structure['measures']
        ]
        self.assertEqual(chord_names, [['C', 'G'], ['Am', 'F']])

    def test_near_tie_uses_entry_chord_as_provisional_primary(self):
        with self.assertLogs('music.structure', level='INFO') as captured_logs:
            structure = build_musical_structure(
                [
                    {'start_time': 0.0, 'end_time': 1.989, 'chord_name': 'F'},
                    {'start_time': 1.989, 'end_time': 4.0, 'chord_name': 'C'},
                ],
                bpm=60,
            )

        chords = structure['measures'][0]['chords']
        self.assertEqual(
            [chord['chord_name'] for chord in chords],
            ['F', 'C'],
        )
        log_output = '\n'.join(captured_logs.output)
        self.assertIn('ambiguous_balanced_split: entry=F duration_winner=C', log_output)
        self.assertIn('reliable_measures=0 total_measures=1', log_output)

    def test_half_measure_change_remains_real_when_it_continues_next(self):
        structure = build_musical_structure(
            [
                {'start_time': 0.0, 'end_time': 2.0, 'chord_name': 'C'},
                {'start_time': 2.0, 'end_time': 8.0, 'chord_name': 'G'},
            ],
            bpm=60,
        )

        chord_names = [
            [chord['chord_name'] for chord in measure['chords']]
            for measure in structure['measures']
        ]
        self.assertEqual(chord_names, [['C', 'G'], ['G']])

    def test_late_change_without_next_measure_confirmation_is_preserved(self):
        structure = build_musical_structure(
            [
                {'start_time': 0.0, 'end_time': 2.5, 'chord_name': 'C'},
                {'start_time': 2.5, 'end_time': 4.0, 'chord_name': 'G'},
                {'start_time': 4.0, 'end_time': 8.0, 'chord_name': 'Am'},
            ],
            bpm=60,
        )

        first_measure = structure['measures'][0]['chords']
        self.assertEqual(
            [chord['chord_name'] for chord in first_measure],
            ['C', 'G'],
        )

    def test_repeating_four_measure_pattern_corrects_late_spillovers(self):
        chord_events = []
        progression = ['F', 'F', 'C', 'Em7']
        ambiguous_measure_indexes = {6, 14}
        for measure_index in range(16):
            measure_start = measure_index * 4.0
            chord_name = progression[measure_index % len(progression)]
            if measure_index in ambiguous_measure_indexes:
                chord_events.extend(
                    [
                        {
                            'start_time': measure_start,
                            'end_time': measure_start + 1.989,
                            'chord_name': chord_name,
                        },
                        {
                            'start_time': measure_start + 1.989,
                            'end_time': measure_start + 4.0,
                            'chord_name': 'Em7',
                        },
                    ]
                )
            else:
                chord_events.append(
                    {
                        'start_time': measure_start,
                        'end_time': measure_start + 4.0,
                        'chord_name': chord_name,
                    }
                )

        with self.assertLogs('music.structure', level='INFO') as captured_logs:
            structure = build_musical_structure(
                chord_events,
                bpm=60,
            )

        chord_names = [
            [chord['chord_name'] for chord in measure['chords']]
            for measure in structure['measures']
        ]
        self.assertEqual(
            chord_names,
            [
                ['F'], ['F'], ['C'], ['Em7'],
                ['F'], ['F'], ['C'], ['Em7'],
                ['F'], ['F'], ['C'], ['Em7'],
                ['F'], ['F'], ['C'], ['Em7'],
            ],
        )
        log_output = '\n'.join(captured_logs.output)
        self.assertIn('Detected harmonic cycle: length=4', log_output)
        self.assertIn('Measure 7 classified as ambiguous_balanced_split', log_output)
        self.assertIn('Measure 15 classified as ambiguous_balanced_split', log_output)
        self.assertIn(
            'Pattern correction on measure 7 (ambiguous_balanced_split)',
            log_output,
        )
        self.assertIn(
            'Pattern correction on measure 15 (ambiguous_balanced_split)',
            log_output,
        )

    def test_repeating_real_two_chord_measures_are_not_forced_into_single_chords(self):
        structure = build_musical_structure(
            [
                {'start_time': 0.0, 'end_time': 2.0, 'chord_name': 'C'},
                {'start_time': 2.0, 'end_time': 4.0, 'chord_name': 'G'},
                {'start_time': 4.0, 'end_time': 6.0, 'chord_name': 'Am'},
                {'start_time': 6.0, 'end_time': 8.0, 'chord_name': 'F'},
                {'start_time': 8.0, 'end_time': 10.0, 'chord_name': 'C'},
                {'start_time': 10.0, 'end_time': 12.0, 'chord_name': 'G'},
                {'start_time': 12.0, 'end_time': 14.0, 'chord_name': 'Am'},
                {'start_time': 14.0, 'end_time': 16.0, 'chord_name': 'F'},
            ],
            bpm=60,
        )

        chord_names = [
            [chord['chord_name'] for chord in measure['chords']]
            for measure in structure['measures']
        ]
        self.assertEqual(chord_names, [['C', 'G'], ['Am', 'F'], ['C', 'G'], ['Am', 'F']])

    def test_cycle_phase_blocks_single_with_strong_local_contradiction(self):
        chord_events = []
        progression = ['F', 'F', 'C', 'Em7']
        for measure_index in range(16):
            measure_start = measure_index * 4.0
            chord_name = progression[measure_index % len(progression)]
            if measure_index == 7:
                chord_events.extend(
                    [
                        {
                            'start_time': measure_start,
                            'end_time': measure_start + 3.0,
                            'chord_name': 'C',
                        },
                        {
                            'start_time': measure_start + 3.0,
                            'end_time': measure_start + 4.0,
                            'chord_name': 'Em7',
                        },
                    ]
                )
            else:
                chord_events.append(
                    {
                        'start_time': measure_start,
                        'end_time': measure_start + 4.0,
                        'chord_name': chord_name,
                    }
                )

        with self.assertLogs('music.structure', level='INFO') as captured_logs:
            structure = build_musical_structure(chord_events, bpm=60)

        self.assertEqual(
            [chord['chord_name'] for chord in structure['measures'][7]['chords']],
            ['C'],
        )
        log_output = '\n'.join(captured_logs.output)
        self.assertIn('Detected cycle phase: phase=0 expected_measure_one=F', log_output)
        self.assertIn(
            'Phase-based single correction blocked on measure 8: strong_local_evidence detected=C expected=Em7.',
            log_output,
        )

    def test_pattern_is_not_applied_after_section_break(self):
        chord_events = []
        for measure_index, chord_name in enumerate(
            ['F', 'F', 'C', 'Em7'] * 3
        ):
            measure_start = measure_index * 4.0
            chord_events.append(
                {
                    'start_time': measure_start,
                    'end_time': measure_start + 4.0,
                    'chord_name': chord_name,
                }
            )
        chord_events.extend(
            [
                {'start_time': 48.0, 'end_time': 50.0, 'chord_name': 'G'},
                {'start_time': 50.0, 'end_time': 52.0, 'chord_name': 'D'},
                {'start_time': 52.0, 'end_time': 54.0, 'chord_name': 'Em'},
                {'start_time': 54.0, 'end_time': 56.0, 'chord_name': 'C'},
                {'start_time': 56.0, 'end_time': 58.0, 'chord_name': 'G'},
                {'start_time': 58.0, 'end_time': 60.0, 'chord_name': 'D'},
                {'start_time': 60.0, 'end_time': 62.0, 'chord_name': 'Em'},
                {'start_time': 62.0, 'end_time': 64.0, 'chord_name': 'C'},
            ]
        )

        with self.assertLogs('music.structure', level='INFO') as captured_logs:
            structure = build_musical_structure(chord_events, bpm=60)

        section_chords = [
            [chord['chord_name'] for chord in measure['chords']]
            for measure in structure['measures'][12:]
        ]
        self.assertEqual(
            section_chords,
            [['G', 'D'], ['Em7', 'C'], ['G', 'D'], ['Em7', 'C']],
        )
        self.assertIn(
            'Detected harmonic cycle: length=4',
            '\n'.join(captured_logs.output),
        )

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

    def test_score_edit_page_displays_existing_measures_and_chords(self):
        Score.objects.create(
            music_project=self.project,
            data={
                'chords': [{'start_time': 0, 'end_time': 2, 'chord_name': 'F'}],
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [
                                {'chord_name': 'F', 'beat_start': 0, 'beat_duration': 4},
                            ],
                        },
                    ],
                },
            },
        )
        self.client.force_login(self.user)

        response = self.client.get(reverse('music:score_edit', args=[self.project.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Editar partitura')
        self.assertContains(response, 'scoreVisual')
        self.assertContains(response, 'scoreMeasuresData')
        self.assertContains(response, 'data-measure-number')
        self.assertContains(response, 'score-rhythm-slash')
        self.assertContains(response, 'dblclick')
        self.assertContains(response, 'pointerdown')
        self.assertContains(response, 'position_ratio')
        self.assertContains(response, 'deleted_chords')
        self.assertContains(response, 'deleted_added_chords')
        self.assertContains(response, "event.key === 'Delete'")
        self.assertContains(response, 'scoreHeaderTitle')
        self.assertContains(response, 'scoreHeaderArtist')
        self.assertContains(response, 'scoreHeaderKey')
        self.assertContains(response, 'scoreHeaderBpm')
        self.assertContains(response, 'header_edits')
        self.assertNotContains(response, 'toFixed(3)')
        self.assertContains(response, 'Adicionar acorde')
        self.assertContains(response, 'Exportar PDF')
        self.assertContains(response, reverse('music:export_pdf', args=[self.project.pk]))
        self.assertContains(response, 'Exportar para MuseScore (.musicxml)')
        self.assertContains(response, reverse('music:export_musicxml', args=[self.project.pk]))
        self.assertContains(response, 'exportButtons')

    def test_pdf_score_data_applies_manual_chords_without_mutating_detected_data(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [{'chord_name': 'F', 'beat_start': 0, 'beat_duration': 4}],
                        },
                    ],
                },
            },
            manual_chord_edits={'chords': {'1:0': {'chord_name': 'Am7'}}},
        )

        rendered_data = pdf_score_data(self.project, score)

        self.assertEqual(
            rendered_data['structure']['measures'][0]['chords'][0]['chord_name'],
            'Am7',
        )
        self.assertEqual(
            score.data['structure']['measures'][0]['chords'][0]['chord_name'],
            'F',
        )

    def test_manual_added_chord_is_overlaid_and_splits_the_measure_for_pdf(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [{'chord_name': 'F', 'beat_start': 0, 'beat_duration': 4}],
                        },
                    ],
                },
            },
            manual_chord_edits={
                'added_chords': {'1:2': {'chord_name': 'C', 'beat_start': 2}},
            },
        )

        rendered_data = score_data_with_manual_chord_edits(score)
        rendered_chords = rendered_data['structure']['measures'][0]['chords']

        self.assertEqual([chord['chord_name'] for chord in rendered_chords], ['F', 'C'])
        self.assertEqual([chord['beat_start'] for chord in rendered_chords], [0, 2])
        self.assertEqual([chord['beat_duration'] for chord in rendered_chords], [2.0, 2.0])
        self.assertEqual(score.data['structure']['measures'][0]['chords'][0]['chord_name'], 'F')
        self.assertEqual(score.data['structure']['measures'][0]['chords'][0]['beat_duration'], 4)

    def test_manual_deleted_detected_chord_is_removed_from_editor_and_pdf_data(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [{
                        'number': 1,
                        'chords': [
                            {'chord_name': 'F', 'beat_start': 0, 'beat_duration': 2},
                            {'chord_name': 'C', 'beat_start': 2, 'beat_duration': 2},
                        ],
                    }],
                },
            },
            manual_chord_edits={'deleted_chords': {'1:0': True}},
        )

        editor_data = score_data_with_manual_chord_edits(score, include_editor_metadata=True)
        pdf_data = pdf_score_data(self.project, score)

        self.assertEqual(
            [chord['chord_name'] for chord in editor_data['structure']['measures'][0]['chords']],
            ['C'],
        )
        self.assertEqual(
            [chord['chord_name'] for chord in pdf_data['structure']['measures'][0]['chords']],
            ['C'],
        )
        self.assertEqual(score.data['structure']['measures'][0]['chords'][0]['chord_name'], 'F')

    def test_fully_deleted_measure_stays_blank_in_editor_and_pdf_data(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [{
                        'number': 1,
                        'chords': [{'chord_name': 'F', 'beat_start': 0, 'beat_duration': 4}],
                    }],
                },
            },
            manual_chord_edits={'deleted_chords': {'1:0': True}},
        )

        editor_data = score_data_with_manual_chord_edits(score, include_editor_metadata=True)
        pdf_data = pdf_score_data(self.project, score)

        self.assertEqual(editor_data['structure']['measures'][0]['chords'], [])
        self.assertEqual(pdf_data['structure']['measures'][0]['chords'][0]['chord_name'], ' ')
        self.assertEqual(score.data['structure']['measures'][0]['chords'][0]['chord_name'], 'F')

    def test_detected_no_chord_label_remains_visible(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [{
                        'number': 1,
                        'chords': [{'chord_name': 'N.C.', 'beat_start': 0, 'beat_duration': 4}],
                    }],
                },
            },
        )

        editor_data = score_data_with_manual_chord_edits(score)
        pdf_data = pdf_score_data(self.project, score)

        self.assertEqual(editor_data['structure']['measures'][0]['chords'][0]['chord_name'], 'N.C.')
        self.assertEqual(pdf_data['structure']['measures'][0]['chords'][0]['chord_name'], 'N.C.')

    def test_manual_header_edits_overlay_editor_and_pdf_without_mutating_score_data(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'title': 'Titulo detectado',
                'artist': 'Artista detectado',
                'key': 'C',
                'bpm': 65,
                'structure': {'measures': []},
            },
            manual_chord_edits={
                'manual_header_edits': {
                    'title': 'Titulo manual',
                    'artist': 'Artista manual',
                    'key': 'F#m',
                    'bpm': 72.5,
                },
            },
        )

        editor_data = score_data_with_manual_chord_edits(score)
        pdf_data = pdf_score_data(self.project, score)

        self.assertEqual(
            [editor_data[field_name] for field_name in ('title', 'artist', 'key', 'bpm')],
            ['Titulo manual', 'Artista manual', 'F#m', 72.5],
        )
        self.assertEqual(
            [pdf_data[field_name] for field_name in ('title', 'artist', 'key', 'bpm')],
            ['Titulo manual', 'Artista manual', 'F#m', 72.5],
        )
        self.assertEqual(score.data['title'], 'Titulo detectado')
        self.assertEqual(score.data['key'], 'C')

    def test_manual_position_moves_detected_chord_without_mutating_detected_data(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [
                                {'chord_name': 'F', 'beat_start': 0, 'beat_duration': 2},
                                {'chord_name': 'C', 'beat_start': 2, 'beat_duration': 2},
                            ],
                        },
                    ],
                },
            },
            manual_chord_edits={'chords': {'1:0': {'beat_start': 1.375}}},
        )

        rendered_data = pdf_score_data(self.project, score)
        rendered_chords = rendered_data['structure']['measures'][0]['chords']

        self.assertEqual([chord['chord_name'] for chord in rendered_chords], ['F', 'C'])
        self.assertEqual([chord['beat_start'] for chord in rendered_chords], [1.375, 2])
        self.assertEqual([chord['beat_duration'] for chord in rendered_chords], [0.625, 2.0])
        self.assertEqual(score.data['structure']['measures'][0]['chords'][0]['beat_start'], 0)

    def test_manual_added_chord_keeps_stable_id_when_moved_continuously(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [{'chord_name': 'F', 'beat_start': 0, 'beat_duration': 4}],
                        },
                    ],
                },
            },
            manual_chord_edits={
                'added_chords': {'1:2': {'chord_name': 'C', 'beat_start': 2}},
            },
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [],
                'added_chords': [{
                    'measure_number': 1,
                    'beat_start': 2,
                    'position_ratio': 0.586789123456,
                    'chord_name': 'C',
                    'previous_manual_chord_id': '1:2',
                }],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        score.refresh_from_db()
        stored_chord = score.manual_chord_edits['added_chords']['1:2']
        self.assertEqual(stored_chord['chord_name'], 'C')
        self.assertEqual(stored_chord['position_ratio'], 0.586789123456)
        self.assertEqual(stored_chord['beat_start'], 0.586789123456 * 4)
        rendered_data = score_data_with_manual_chord_edits(score, include_editor_metadata=True)
        self.assertEqual(
            [chord['beat_start'] for chord in rendered_data['structure']['measures'][0]['chords']],
            [0, 0.586789123456 * 4],
        )
        self.assertEqual(
            rendered_data['structure']['measures'][0]['chords'][1]['_manual_position_ratio'],
            0.586789123456,
        )

    @patch('music.views._try_regenerate_pdf')
    def test_project_page_keeps_existing_export_pdf_button(self, mock_regenerate):
        Score.objects.create(
            music_project=self.project,
            data={'structure': {'measures': [{'number': 1, 'chords': [{'chord_name': 'F'}]}]}},
        )
        self.client.force_login(self.user)

        response = self.client.get(reverse('music:project_detail', args=[self.project.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Exportar PDF')
        self.assertContains(response, reverse('music:export_pdf', args=[self.project.pk]))

    @patch.object(HarmixPipeline, '_make_pdf', return_value=b'%PDF-1.4 corrected')
    def test_exported_pdf_receives_manually_corrected_chord(self, mock_make_pdf):
        Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [{'chord_name': 'F', 'beat_start': 0, 'beat_duration': 4}],
                        },
                    ],
                },
            },
            manual_chord_edits={'chords': {'1:0': {'chord_name': 'Am7'}}},
        )
        self.client.force_login(self.user)

        response = self.client.get(reverse('music:export_pdf', args=[self.project.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        rendered_data = mock_make_pdf.call_args.args[0]
        self.assertEqual(
            rendered_data['structure']['measures'][0]['chords'][0]['chord_name'],
            'Am7',
        )

    def test_owner_can_export_current_edited_score_as_valid_musicxml(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'title': 'Sou Casa',
                'artist': 'Artista original',
                'key': 'F',
                'bpm': 65,
                'chords': [],
                'structure': {
                    'time_signature': {'beats_per_measure': 4, 'beat_unit': 4},
                    'measures': [{
                        'number': 1,
                        'chords': [
                            {'chord_name': 'F', 'beat_start': 0, 'beat_duration': 2},
                            {'chord_name': 'C', 'beat_start': 2, 'beat_duration': 2},
                        ],
                    }],
                },
            },
            manual_chord_edits={
                'manual_header_edits': {
                    'title': 'Sou Casa Editada',
                    'artist': 'Artista manual',
                    'key': 'Bb',
                    'bpm': 72.5,
                },
                'chords': {
                    '1:0': {
                        'chord_name': 'Am7',
                        'beat_start': 0.437123456789 * 4,
                        'position_ratio': 0.437123456789,
                    },
                },
                'deleted_chords': {'1:1': True},
                'added_chords': {
                    '1:3': {
                        'chord_name': 'D/F#',
                        'beat_start': 0.82 * 4,
                        'position_ratio': 0.82,
                    },
                },
            },
        )
        original_data = deepcopy(score.data)
        self.client.force_login(self.user)

        response = self.client.get(reverse('music:export_musicxml', args=[self.project.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/vnd.recordare.musicxml+xml')
        self.assertIn('attachment;', response['Content-Disposition'])
        self.assertIn('sou-casa-editada.musicxml', response['Content-Disposition'])
        xml = b''.join(response.streaming_content).decode('utf-8')
        root = ET.fromstring(xml)
        self.assertEqual(root.tag, 'score-partwise')
        self.assertEqual(root.findtext('./work/work-title'), 'Sou Casa Editada')
        self.assertEqual(root.findtext('./identification/creator'), 'Artista manual')

        first_measure = root.find('./part/measure')
        self.assertEqual(first_measure.findtext('./attributes/divisions'), str(MUSICXML_DIVISIONS))
        self.assertEqual(first_measure.findtext('./attributes/time/beats'), '4')
        self.assertEqual(first_measure.findtext('./attributes/time/beat-type'), '4')
        self.assertEqual(first_measure.findtext('./attributes/key/fifths'), '-2')
        self.assertEqual(first_measure.find('./direction/sound').get('tempo'), '72.5')

        harmonies = first_measure.findall('./harmony')
        self.assertEqual([harmony.findtext('./root/root-step') for harmony in harmonies], ['A', 'D'])
        self.assertEqual(harmonies[0].findtext('./kind'), 'minor-seventh')
        self.assertEqual(harmonies[1].findtext('./bass/bass-step'), 'F')
        self.assertEqual(harmonies[1].findtext('./bass/bass-alter'), '1')

        cursor_ticks = 0
        harmony_positions = []
        for child in first_measure:
            if child.tag == 'forward':
                cursor_ticks += int(child.findtext('./duration'))
            elif child.tag == 'harmony':
                harmony_positions.append(cursor_ticks)
        self.assertEqual(
            harmony_positions,
            [
                round(0.437123456789 * 4 * MUSICXML_DIVISIONS),
                round(0.82 * 4 * MUSICXML_DIVISIONS),
            ],
        )
        score.refresh_from_db()
        self.assertEqual(score.data, original_data)

    def test_other_user_cannot_export_project_musicxml(self):
        Score.objects.create(
            music_project=self.project,
            data={
                'title': 'Private Score',
                'structure': {'measures': [{'number': 1, 'chords': []}]},
            },
        )
        other_user = get_user_model().objects.create_user(
            email='other-musicxml@harmix.test',
            password='test-pass-123',
        )
        self.client.force_login(other_user)

        response = self.client.get(reverse('music:export_musicxml', args=[self.project.pk]))

        self.assertEqual(response.status_code, 404)

    def test_score_edit_page_rejects_post_requests(self):
        Score.objects.create(music_project=self.project, data={})
        self.client.force_login(self.user)

        response = self.client.post(reverse('music:score_edit', args=[self.project.pk]))

        self.assertEqual(response.status_code, 405)

    def test_owner_can_save_manual_chord_edit_without_changing_detected_score(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [{'chord_name': 'F', 'beat_start': 0, 'beat_duration': 4}],
                        },
                    ],
                },
            },
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({'changes': [{'measure_number': 1, 'chord_index': 0, 'chord_name': 'Am7'}]}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        score.refresh_from_db()
        self.assertEqual(score.data['structure']['measures'][0]['chords'][0]['chord_name'], 'F')
        self.assertEqual(score.manual_chord_edits['chords']['1:0']['chord_name'], 'Am7')
        response = self.client.get(reverse('music:score_edit', args=[self.project.pk]))
        self.assertContains(response, 'Am7')

    def test_owner_can_save_detected_chord_position(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [
                                {'chord_name': 'F', 'beat_start': 0},
                                {'chord_name': 'C', 'beat_start': 2},
                            ],
                        },
                    ],
                },
            },
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [{
                    'measure_number': 1,
                    'chord_index': 0,
                    'chord_name': 'F',
                    'beat_start': 0,
                    'position_ratio': 0.437123456789,
                }],
                'added_chords': [],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        score.refresh_from_db()
        saved_edit = score.manual_chord_edits['chords']['1:0']
        self.assertEqual(saved_edit['position_ratio'], 0.437123456789)
        self.assertEqual(saved_edit['beat_start'], 0.437123456789 * 4)
        self.assertEqual(score.data['structure']['measures'][0]['chords'][0]['beat_start'], 0)

        editor_data = score_data_with_manual_chord_edits(score, include_editor_metadata=True)
        editor_chord = editor_data['structure']['measures'][0]['chords'][0]
        self.assertEqual(editor_chord['_manual_position_ratio'], 0.437123456789)
        self.assertEqual(editor_chord['beat_start'], 0.437123456789 * 4)

        pdf_chord = pdf_score_data(self.project, score)['structure']['measures'][0]['chords'][0]
        pdf_center_beat = pdf_chord['beat_start'] + (pdf_chord['beat_duration'] / 2)
        self.assertAlmostEqual(pdf_center_beat, 0.437123456789 * 4, places=9)

    def test_owner_can_save_manual_header_edits_without_changing_original_data(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'title': 'Titulo detectado',
                'artist': 'Artista detectado',
                'key': 'C',
                'bpm': 65,
                'structure': {'measures': []},
            },
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [],
                'header_edits': {
                    'title': 'Titulo manual',
                    'artist': 'Artista manual',
                    'key': 'Bb',
                    'bpm': 72.5,
                },
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['saved_header_edits'], 4)
        score.refresh_from_db()
        self.assertEqual(
            score.manual_chord_edits['manual_header_edits'],
            {'title': 'Titulo manual', 'artist': 'Artista manual', 'key': 'Bb', 'bpm': 72.5},
        )
        self.assertEqual(score.data['title'], 'Titulo detectado')
        self.assertEqual(score.data['bpm'], 65)
        self.assertEqual(pdf_score_data(self.project, score)['title'], 'Titulo manual')

    def test_manual_header_edits_reject_invalid_key_and_bpm(self):
        Score.objects.create(
            music_project=self.project,
            data={'structure': {'measures': []}},
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [],
                'header_edits': {'key': 'H7'},
            }),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [],
                'header_edits': {'bpm': 0},
            }),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)

    def test_continuous_detected_position_may_overlap_an_existing_beat(self):
        Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [{
                        'number': 1,
                        'chords': [
                            {'chord_name': 'F', 'beat_start': 0},
                            {'chord_name': 'C', 'beat_start': 2},
                        ],
                    }],
                },
            },
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [{
                    'measure_number': 1,
                    'chord_index': 0,
                    'chord_name': 'F',
                    'beat_start': 2,
                    'position_ratio': 0.5,
                }],
                'added_chords': [],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)

    def test_owner_can_save_detected_chord_deletion_without_mutating_score_data(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [{
                        'number': 1,
                        'chords': [
                            {'chord_name': 'F', 'beat_start': 0},
                            {'chord_name': 'C', 'beat_start': 2},
                        ],
                    }],
                },
            },
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [],
                'added_chords': [],
                'deleted_chords': [{'measure_number': 1, 'chord_index': 0}],
                'deleted_added_chords': [],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['saved_deleted_chords'], 1)
        score.refresh_from_db()
        self.assertTrue(score.manual_chord_edits['deleted_chords']['1:0'])
        self.assertEqual(score.data['structure']['measures'][0]['chords'][0]['chord_name'], 'F')
        self.assertEqual(
            [chord['chord_name'] for chord in pdf_score_data(self.project, score)['structure']['measures'][0]['chords']],
            ['C'],
        )

    def test_owner_can_delete_a_persisted_manual_added_chord(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [{
                        'number': 1,
                        'chords': [{'chord_name': 'F', 'beat_start': 0, 'beat_duration': 4}],
                    }],
                },
            },
            manual_chord_edits={
                'added_chords': {'1:2': {'chord_name': 'C', 'beat_start': 2}},
            },
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [],
                'added_chords': [],
                'deleted_chords': [],
                'deleted_added_chords': ['1:2'],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['saved_deleted_added_chords'], 1)
        score.refresh_from_db()
        self.assertNotIn('1:2', score.manual_chord_edits['added_chords'])
        self.assertEqual(score.data['structure']['measures'][0]['chords'][0]['chord_name'], 'F')
        self.assertEqual(
            [chord['chord_name'] for chord in pdf_score_data(self.project, score)['structure']['measures'][0]['chords']],
            ['F'],
        )

    def test_detected_chord_position_rejects_an_occupied_beat(self):
        Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [
                                {'chord_name': 'F', 'beat_start': 0},
                                {'chord_name': 'C', 'beat_start': 2},
                            ],
                        },
                    ],
                },
            },
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [{'measure_number': 1, 'chord_index': 0, 'chord_name': 'F', 'beat_start': 2}],
                'added_chords': [],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 400)

    def test_owner_can_add_manual_chord_without_changing_detected_score(self):
        score = Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [{'chord_name': 'F', 'beat_start': 0, 'beat_duration': 4}],
                        },
                    ],
                },
            },
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [],
                'added_chords': [{'measure_number': 1, 'beat_start': 2, 'chord_name': 'C'}],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        score.refresh_from_db()
        self.assertEqual(score.data['structure']['measures'][0]['chords'][0]['chord_name'], 'F')
        self.assertEqual(score.manual_chord_edits['added_chords']['1:2']['chord_name'], 'C')
        response = self.client.get(reverse('music:score_edit', args=[self.project.pk]))
        self.assertContains(response, 'C')

    def test_manual_added_chord_rejects_an_existing_beat(self):
        Score.objects.create(
            music_project=self.project,
            data={
                'structure': {
                    'measures': [
                        {
                            'number': 1,
                            'chords': [{'chord_name': 'F', 'beat_start': 0, 'beat_duration': 4}],
                        },
                    ],
                },
            },
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({
                'changes': [],
                'added_chords': [{'measure_number': 1, 'beat_start': 0, 'chord_name': 'C'}],
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 400)

    def test_manual_chord_edit_requires_valid_owned_chord(self):
        Score.objects.create(
            music_project=self.project,
            data={'structure': {'measures': [{'number': 1, 'chords': [{'chord_name': 'F'}]}]}},
        )
        other_user = get_user_model().objects.create_user(
            email='other-score@harmix.test',
            password='test-pass-123',
        )
        self.client.force_login(other_user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({'changes': [{'measure_number': 1, 'chord_index': 0, 'chord_name': 'Invalid chord'}]}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 404)

    def test_manual_chord_edit_rejects_invalid_name(self):
        Score.objects.create(
            music_project=self.project,
            data={'structure': {'measures': [{'number': 1, 'chords': [{'chord_name': 'F'}]}]}},
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({'changes': [{'measure_number': 1, 'chord_index': 0, 'chord_name': 'invalid chord'}]}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 400)

    def test_chord_name_validation_accepts_common_chord_structures(self):
        valid_chords = [
            'C', 'Cm', 'C7', 'Cm7', 'Cmaj7', 'Cdim', 'Caug', 'Csus2', 'Csus4',
            'C#', 'C#m', 'Bb', 'Bbm', 'F#m7', 'G/B', 'D/F#', 'C7sus4', 'C6/9',
            'C7(b9,#11)', 'N.C.',
        ]
        invalid_chords = ['Emm', 'H7', 'C/7', 'CmajX', 'C//G', 'not-a-chord']

        for chord_name in valid_chords:
            with self.subTest(chord_name=chord_name):
                self.assertIsNotNone(CHORD_NAME_PATTERN.fullmatch(chord_name))
        for chord_name in invalid_chords:
            with self.subTest(chord_name=chord_name):
                self.assertIsNone(CHORD_NAME_PATTERN.fullmatch(chord_name))

    def test_manual_chord_edit_accepts_em(self):
        score = Score.objects.create(
            music_project=self.project,
            data={'structure': {'measures': [{'number': 1, 'chords': [{'chord_name': 'F'}]}]}},
        )
        self.client.force_login(self.user)

        response = self.client.post(
            reverse('music:score_chord_edits', args=[self.project.pk]),
            data=json.dumps({'changes': [{'measure_number': 1, 'chord_index': 0, 'chord_name': 'Em'}]}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        score.refresh_from_db()
        self.assertEqual(score.manual_chord_edits['chords']['1:0']['chord_name'], 'Em')

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
