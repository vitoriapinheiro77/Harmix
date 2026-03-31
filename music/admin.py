from django.contrib import admin

from .models import ChordSegment, MusicProject, Score, Track


class TrackInline(admin.TabularInline):
    model = Track
    extra = 0
    fields = ('instrument_name', 'audio_file')


class ScoreInline(admin.StackedInline):
    model = Score
    extra = 0
    fields = ('data',)


@admin.register(MusicProject)
class MusicProjectAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'status', 'key', 'bpm', 'created_at')
    list_filter = ('status',)
    search_fields = ('title', 'user__username')
    readonly_fields = ('created_at',)
    inlines = [TrackInline, ScoreInline]


class ChordSegmentInline(admin.TabularInline):
    model = ChordSegment
    extra = 0
    fields = ('start_time', 'end_time', 'chord_name')


@admin.register(Track)
class TrackAdmin(admin.ModelAdmin):
    list_display = ('instrument_name', 'music_project')
    search_fields = ('instrument_name', 'music_project__title')
    inlines = [ChordSegmentInline]


@admin.register(ChordSegment)
class ChordSegmentAdmin(admin.ModelAdmin):
    list_display = ('chord_name', 'track', 'start_time', 'end_time')
    list_filter = ('chord_name',)
    search_fields = ('chord_name', 'track__instrument_name')


@admin.register(Score)
class ScoreAdmin(admin.ModelAdmin):
    list_display = ('music_project',)
    search_fields = ('music_project__title',)

