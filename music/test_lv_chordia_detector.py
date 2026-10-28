import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from .analysis.chord_detector import detect_chords
from .analysis.lv_chordia_detector import detect_lv_chordia
from .analysis.lv_chordia_labels import harte_to_conventional
from .analysis.types import ChordAnalysisResult, ChordEvent


class LVChordiaLabelTests(SimpleTestCase):
    def test_harte_labels_preserve_extensions_and_inversions(self):
        self.assertEqual(harte_to_conventional('G:maj/3'), 'G/B')
        self.assertEqual(harte_to_conventional('E:min7'), 'Em7')
        self.assertEqual(harte_to_conventional('C#:hdim7'), 'C#m7b5')
        self.assertIsNone(harte_to_conventional('N'))

    def test_detector_converts_segments_and_discards_no_chord(self):
        recognize = Mock(return_value=[
            {'start_time': 0.0, 'end_time': 1.0, 'chord': 'N'},
            {'start_time': 1.0, 'end_time': 3.0, 'chord': 'G:maj/3'},
            {'start_time': 3.0, 'end_time': 5.0, 'chord': 'E:min7'},
        ])
        with patch.dict(sys.modules, {'lv_chordia': SimpleNamespace(chord_recognition=recognize)}):
            result = detect_lv_chordia('song.mp3', 4.0)

        self.assertEqual(
            [(event.start_time, event.end_time, event.chord_name) for event in result.chords],
            [(1.0, 3.0, 'G/B'), (3.0, 4.0, 'Em7')],
        )
        recognize.assert_called_once_with('song.mp3', device='cpu')


class LVChordiaOnlyTests(SimpleTestCase):
    @patch('music.analysis.chord_detector.detect_lv_chordia')
    def test_detector_returns_lv_chordia_result(self, mock_lv):
        mock_lv.return_value = ChordAnalysisResult(chords=[ChordEvent(
            start_time=0.0,
            end_time=1.0,
            chord_name='F',
            source='lv_chordia',
        )], source='lv_chordia')
        result = detect_chords('song.mp3', 1.0)

        self.assertEqual(result.source, 'lv_chordia')
        mock_lv.assert_called_once_with('song.mp3', 1.0)

    @patch('music.analysis.chord_detector.detect_lv_chordia')
    def test_empty_lv_result_fails_without_fallback(self, mock_lv):
        mock_lv.return_value = ChordAnalysisResult(source='lv_chordia')
        with self.assertRaisesMessage(RuntimeError, 'LV-Chordia nao retornou'):
            detect_chords('song.mp3', 1.0)
