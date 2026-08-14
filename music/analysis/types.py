from dataclasses import dataclass, field


@dataclass
class RhythmGrid:
    bpm: float | None = None
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    beats_per_measure: int = 4
    source: str = 'librosa'
    errors: list[str] = field(default_factory=list)

    def as_dict(self):
        return {
            'bpm': round(self.bpm, 2) if self.bpm else None,
            'beats': [round(value, 3) for value in self.beats[:500]],
            'downbeats': [round(value, 3) for value in self.downbeats[:200]],
            'beats_per_measure': self.beats_per_measure,
            'source': self.source,
            'errors': self.errors,
            'total_beats': len(self.beats),
            'total_downbeats': len(self.downbeats),
        }


@dataclass
class ChordEvent:
    start_time: float
    end_time: float
    chord_name: str
    confidence: float = 0.0
    source: str = 'librosa'

    def as_detection(self):
        return {
            'start': round(self.start_time, 2),
            'end': round(self.end_time, 2),
            'chord': self.chord_name,
            'confidence': round(self.confidence, 4),
        }

    def as_segment(self):
        return {
            'start_time': round(self.start_time, 2),
            'end_time': round(self.end_time, 2),
            'chord_name': self.chord_name,
        }


@dataclass
class ChordAnalysisResult:
    chords: list[ChordEvent] = field(default_factory=list)
    source: str = 'librosa'
    errors: list[str] = field(default_factory=list)

    def as_dict(self):
        return {
            'source': self.source,
            'errors': self.errors,
            'total_chords': len(self.chords),
            'sample': [chord.as_segment() for chord in self.chords[:10]],
        }


@dataclass
class MusicalAnalysisResult:
    bpm: float | None = None
    key: str = ''
    key_confidence: float = 0.0
    duration: float = 0.0
    sample_rate: int = 22050
    rhythm: RhythmGrid = field(default_factory=RhythmGrid)
    chords: ChordAnalysisResult = field(default_factory=ChordAnalysisResult)
    processing_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_log_entry(self):
        return {
            'bpm_detectado': round(self.bpm, 2) if self.bpm else None,
            'tom_detectado': self.key,
            'confianca_tom': round(self.key_confidence, 4),
            'duracao_segundos': round(self.duration, 2),
            'sample_rate': self.sample_rate,
            'tempo_processamento_segundos': round(self.processing_seconds, 2),
            'ritmo': self.rhythm.as_dict(),
            'acordes': self.chords.as_dict(),
            'erros': self.errors,
        }
