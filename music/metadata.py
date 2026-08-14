import os
import tempfile
from pathlib import Path


def parse_artist_title_from_filename(filename):
    stem = Path(filename).stem
    for separator in (' - ', ' – ', ' — ', ' _ '):
        if separator in stem:
            artist, title = (part.strip() for part in stem.split(separator, 1))
            if artist and title:
                return artist, title
    return '', ''


def _artist_from_mutagen_file(path):
    try:
        from mutagen import File as MutagenFile
        from mutagen.easyid3 import EasyID3
        from mutagen.id3 import ID3NoHeaderError
    except ImportError:
        return ''

    try:
        easy = EasyID3(path)
        artist = (easy.get('artist') or [''])[0].strip()
        if artist:
            return artist[:255]
    except ID3NoHeaderError:
        pass
    except Exception:
        pass

    try:
        audio = MutagenFile(path)
        if audio is None:
            return ''

        tag_sources = []
        if getattr(audio, 'tags', None):
            tag_sources.append(audio.tags)
        tag_sources.append(audio)

        for tags in tag_sources:
            for key in ('artist', 'TPE1', '\xa9ART', 'ARTIST'):
                value = tags.get(key) if hasattr(tags, 'get') else None
                if not value:
                    continue
                text = value[0] if isinstance(value, list) else str(value)
                text = str(text).strip()
                if text:
                    return text[:255]
    except Exception:
        return ''

    return ''


def extract_audio_artist(file_obj):
    if not file_obj:
        return ''

    suffix = os.path.splitext(getattr(file_obj, 'name', '') or '.mp3')[1] or '.mp3'
    try:
        file_obj.seek(0)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            if hasattr(file_obj, 'chunks'):
                for chunk in file_obj.chunks():
                    tmp.write(chunk)
            else:
                tmp.write(file_obj.read())
            temp_path = tmp.name

        artist = _artist_from_mutagen_file(temp_path)
        file_obj.seek(0)
        return artist
    except Exception:
        try:
            file_obj.seek(0)
        except Exception:
            pass
        return ''
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)


def extract_audio_artist_from_storage(storage_file):
    if not storage_file or not storage_file.name:
        return ''

    suffix = os.path.splitext(storage_file.name)[1] or '.mp3'
    temp_path = ''
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            with storage_file.open('rb') as src:
                tmp.write(src.read())
            temp_path = tmp.name
        return _artist_from_mutagen_file(temp_path)
    except Exception:
        return ''
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
