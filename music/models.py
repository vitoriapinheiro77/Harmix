from django.conf import settings
from django.db import models


class MusicProject(models.Model):
    class Status(models.TextChoices):
        QUEUED = 'queued', 'Queued'
        PROCESSING = 'processing', 'Processing'
        DOWNLOADING = 'downloading', 'Downloading audio'
        ANALYZING = 'analyzing', 'Analyzing audio'
        SEPARATING = 'separating', 'Separating tracks'
        EXTRACTING_CHORDS = 'extracting_chords', 'Extracting chords'
        BUILDING_SCORE = 'building_score', 'Building score'
        DONE = 'done', 'Done'
        ERROR = 'error', 'Error'

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='music_projects')
    title = models.CharField(max_length=255)
    artist = models.CharField(max_length=255, blank=True)
    youtube_url = models.URLField(blank=True)
    audio_file = models.FileField(upload_to='audio/projects/', blank=True)
    key = models.CharField(max_length=10, blank=True)
    bpm = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.QUEUED)
    processing_step = models.CharField(max_length=255, blank=True)
    task_id = models.CharField(max_length=255, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        display = (self.user.first_name or '').strip() or self.user.email
        return f'{self.title} ({display})'


class Track(models.Model):
    music_project = models.ForeignKey(MusicProject, on_delete=models.CASCADE, related_name='tracks')
    instrument_name = models.CharField(max_length=100)
    audio_file = models.FileField(upload_to='audio/tracks/')

    def __str__(self):
        return f'{self.instrument_name} — {self.music_project.title}'


class ChordSegment(models.Model):
    track = models.ForeignKey(Track, on_delete=models.CASCADE, related_name='chord_segments')
    start_time = models.FloatField()
    end_time = models.FloatField()
    chord_name = models.CharField(max_length=20)

    class Meta:
        ordering = ['start_time']

    def __str__(self):
        return f'{self.chord_name} ({self.start_time}s – {self.end_time}s)'


class Score(models.Model):
    music_project = models.OneToOneField(MusicProject, on_delete=models.CASCADE, related_name='score')
    data = models.JSONField(default=dict)
    musicxml_file = models.FileField(upload_to='scores/musicxml/', blank=True)
    pdf_file = models.FileField(upload_to='scores/pdf/', blank=True)

    def __str__(self):
        return f'Score for {self.music_project.title}'

