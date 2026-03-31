from django.db import models
from django.contrib.auth.models import User


class MusicProject(models.Model):
    class Status(models.TextChoices):
        PROCESSING = 'processing', 'Processing'
        DONE = 'done', 'Done'
        ERROR = 'error', 'Error'

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='music_projects')
    title = models.CharField(max_length=255)
    youtube_url = models.URLField(blank=True)
    audio_file = models.FileField(upload_to='audio/projects/', blank=True)
    key = models.CharField(max_length=10, blank=True)
    bpm = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PROCESSING)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.title} ({self.user.username})'


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

    def __str__(self):
        return f'Score for {self.music_project.title}'

