from unittest.mock import MagicMock, patch
from xml.etree import ElementTree as ET

import numpy as np
from django.test import SimpleTestCase

from .analysis.meter_detection import detect_meter, select_meter
from .analysis.rhythm_validation import validate_rhythm_grid
from .analysis.types import RhythmGrid
from .musicxml import render_musicxml
from .pipeline import HarmixPipeline
from .structure import build_musical_structure


def _hypothesis(score, bars=8, **evidence):
    return {
        'score': score,
        'bar_count': bars,
        'event_count': bars * 4,
        'status': 'scored',
        **evidence,
    }


def _path(meter, start, interval, bars):
    return np.asarray([
        [start + (index * interval), (index % meter) + 1]
        for index in range(meter * bars)
    ])


def _accent_path(activations, path, meter, fps=100):
    accents = {1, 3} if meter == 4 else {1, 4}
    for time_value, position_value in path:
        index = int(round(time_value * fps))
        position = int(position_value)
        if position == 1:
            activations[index, 1] = 0.95
        else:
            activations[index, 0] = 0.85 if position in accents else 0.30


class MeterDecisionTests(SimpleTestCase):
    def test_real_scoring_recognizes_four_four_accent_pattern(self):
        paths = {
            4: _path(4, 0.0, 0.5, 8),
            6: _path(6, 0.17, 0.41, 6),
        }
        activations = np.full((1600, 2), 0.03)
        _accent_path(activations, paths[4], 4)

        result = detect_meter(paths, activations)

        self.assertEqual(result['label'], '4/4')
        self.assertGreater(
            result['hypotheses']['4']['score'],
            result['hypotheses']['6']['score'],
        )

    def test_real_scoring_recognizes_six_eight_compound_pattern(self):
        paths = {
            4: _path(4, 0.13, 0.71, 6),
            6: _path(6, 0.0, 0.5, 8),
        }
        activations = np.full((2600, 2), 0.03)
        _accent_path(activations, paths[6], 6)

        result = detect_meter(paths, activations)

        self.assertEqual(result['label'], '6/8')
        self.assertGreaterEqual(result['score_margin_6_over_4'], 0.06)

    def test_clear_four_four_evidence_keeps_four_four(self):
        result = select_meter({4: _hypothesis(0.82), 6: _hypothesis(0.61)})

        self.assertEqual(result['label'], '4/4')
        self.assertEqual(result['beat_unit'], 4)
        self.assertFalse(result['low_confidence'])

    def test_compound_accent_evidence_selects_six_eight(self):
        result = select_meter({
            4: _hypothesis(0.58, accent_contrast=0.42),
            6: _hypothesis(0.79, accent_contrast=0.86),
        })

        self.assertEqual(result['label'], '6/8')
        self.assertEqual(result['beats_per_measure'], 6)
        self.assertEqual(result['decision'], 'six_eight_evidence')

    def test_low_absolute_scores_can_select_six_eight_with_clear_relative_evidence(self):
        """Regression: Ousado Amor has variable timing but clear 6/8 accents."""
        result = select_meter({
            4: _hypothesis(
                0.197,
                accent_contrast=0.3475,
                downbeat_strength=0.2363,
            ),
            6: _hypothesis(
                0.285,
                accent_contrast=0.6988,
                downbeat_strength=0.3339,
            ),
        })

        self.assertEqual(result['label'], '6/8')
        self.assertGreaterEqual(result['compound_accent_advantage'], 0.15)
        self.assertGreaterEqual(result['downbeat_advantage'], 0.05)

    def test_ambiguous_rhythm_defaults_to_four_four_with_warning(self):
        result = select_meter({4: _hypothesis(0.64), 6: _hypothesis(0.67)})

        self.assertEqual(result['label'], '4/4')
        self.assertTrue(result['low_confidence'])
        self.assertLess(result['score_margin_6_over_4'], 0.06)


class SixEightRhythmTests(SimpleTestCase):
    def setUp(self):
        self.meter = {
            'beats_per_measure': 6,
            'beat_unit': 8,
            'label': '6/8',
            'confidence': 0.83,
            'low_confidence': False,
            'hypotheses': {
                '6': {
                    'median_subdivision_seconds': 0.5,
                    'score': 0.79,
                    'bar_count': 4,
                    'status': 'scored',
                }
            },
        }

    def test_validation_preserves_six_subdivisions_per_measure(self):
        rhythm = RhythmGrid(
            bpm=80,
            beats=[index * 0.5 for index in range(18)],
            downbeats=[0.0, 3.0, 6.0],
            beats_per_measure=6,
            beat_unit=8,
            meter=self.meter,
        )

        validated = validate_rhythm_grid(rhythm)

        self.assertEqual(validated.downbeats, [0.0, 3.0, 6.0])
        self.assertEqual(validated.validation['time_signature'], '6/8')
        self.assertFalse(validated.validation['meter_low_confidence'])

    def test_structure_builds_six_eighth_measures(self):
        rhythm_grid = {
            'beats': [index * 0.5 for index in range(13)],
            'downbeats': [0.0, 3.0, 6.0],
            'beats_per_measure': 6,
            'beat_unit': 8,
            'meter': self.meter,
        }
        structure = build_musical_structure(
            [
                {'start_time': 0.0, 'end_time': 3.0, 'chord_name': 'C'},
                {'start_time': 3.0, 'end_time': 6.0, 'chord_name': 'G'},
            ],
            bpm=80,
            rhythm_grid=rhythm_grid,
        )

        self.assertEqual(structure['time_signature']['label'], '6/8')
        self.assertEqual(structure['measure_count'], 2)
        self.assertEqual(structure['measures'][0]['end_beat'], 6)
        self.assertEqual(structure['measures'][1]['start_beat'], 6)

    def test_musicxml_declares_six_eight_and_writes_six_eighth_slashes(self):
        xml = render_musicxml({
            'title': 'Teste 6/8',
            'artist': 'Harmix',
            'key': 'C',
            'bpm': 80,
            'structure': {
                'time_signature': {'beats_per_measure': 6, 'beat_unit': 8},
                'measures': [{'number': 1, 'chords': []}],
            },
        })
        root = ET.fromstring(xml)

        self.assertEqual(root.findtext('.//time/beats'), '6')
        self.assertEqual(root.findtext('.//time/beat-type'), '8')
        notes = root.findall('.//measure/note')
        self.assertEqual(len(notes), 6)
        self.assertTrue(all(note.findtext('type') == 'eighth' for note in notes))

    @patch('music.pipeline.draw_system_signature', return_value=30)
    def test_pdf_system_uses_six_eight_and_six_marks(self, signature):
        pipeline = HarmixPipeline.__new__(HarmixPipeline)
        pdf = MagicMock()
        with patch.object(pipeline, '_draw_pdf_rhythm_slash') as draw_slash:
            pipeline._draw_pdf_system(
                pdf=pdf,
                measures=[{'number': 1, 'chords': []}],
                x=0,
                y=100,
                width=200,
                line_gap=4,
                beats_per_measure=6,
                beat_unit=8,
                is_first_system=True,
            )

        signature.assert_called_once_with(
            pdf,
            0,
            100,
            4,
            include_time_signature=True,
            beats=6,
            beat_unit=8,
        )
        self.assertEqual(draw_slash.call_count, 6)

    def test_real_pdf_renderer_supports_six_eight_glyphs(self):
        pipeline = HarmixPipeline.__new__(HarmixPipeline)
        pdf_bytes = pipeline._make_pdf({
            'title': 'Teste 6/8',
            'artist': 'Harmix',
            'key': 'C',
            'bpm': 80,
            'structure': {
                'time_signature': {'beats_per_measure': 6, 'beat_unit': 8},
                'measures': [{'number': 1, 'chords': []}],
            },
            'chords': [],
        })

        self.assertTrue(pdf_bytes.startswith(b'%PDF'))
        self.assertGreater(len(pdf_bytes), 1000)


class FourFourCompatibilityTests(SimpleTestCase):
    def test_structure_without_meter_metadata_remains_four_four(self):
        structure = build_musical_structure(
            [{'start_time': 0, 'end_time': 2, 'chord_name': 'C'}],
            bpm=120,
        )

        self.assertEqual(structure['time_signature']['label'], '4/4')
        self.assertEqual(structure['time_signature']['beats_per_measure'], 4)

