import os

from django import forms

from .metadata import parse_artist_title_from_filename
from .models import MusicProject

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac', 'wma', 'opus', 'aiff', 'aif'}
MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


class MusicProjectUploadForm(forms.Form):
    title = forms.CharField(
        label='Titulo do projeto',
        max_length=255,
        required=False,
        widget=forms.TextInput(
            attrs={'placeholder': 'Ex: Solo de violao — opcional (usa o nome do arquivo se vazio)'}
        ),
    )
    artist = forms.CharField(
        label='Artista',
        max_length=255,
        required=False,
        widget=forms.TextInput(
            attrs={'placeholder': 'Ex: Fernandinho — opcional (detectado automaticamente se vazio)'}
        ),
    )
    audio_file = forms.FileField(
        label='Arquivo de audio',
        widget=forms.FileInput(attrs={'accept': 'audio/*'}),
    )

    def clean_audio_file(self):
        f = self.cleaned_data['audio_file']
        ext = os.path.splitext(f.name)[1].lstrip('.').lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f'Formato nao suportado (.{ext}). Use: MP3, WAV, FLAC, OGG, M4A, AAC, OPUS.'
            )
        if f.size > MAX_SIZE_BYTES:
            mb = f.size // (1024 * 1024)
            raise forms.ValidationError(
                f'Arquivo muito grande ({mb} MB). Limite: 100 MB.'
            )
        return f

    def save(self, user):
        audio_file = self.cleaned_data['audio_file']
        raw_title = (self.cleaned_data.get('title') or '').strip()
        raw_artist = (self.cleaned_data.get('artist') or '').strip()
        parsed_artist, parsed_title = parse_artist_title_from_filename(audio_file.name)
        title = raw_title or parsed_title or os.path.splitext(audio_file.name)[0][:255]
        artist = raw_artist

        project = MusicProject(
            user=user,
            title=title,
            artist=artist[:255],
            status=MusicProject.Status.QUEUED,
            processing_step='Arquivo recebido. Aguardando processamento.',
        )
        project.audio_file.save(audio_file.name, audio_file, save=False)
        project.save()
        return project
