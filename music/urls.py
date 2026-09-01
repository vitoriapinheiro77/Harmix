from django.urls import path

from . import views

app_name = 'music'

urlpatterns = [
    path('projects/', views.project_list_view, name='project_list'),
    path('projects/create/', views.project_create_view, name='project_create'),
    path('projects/<int:pk>/', views.project_detail_view, name='project_detail'),
    path('projects/<int:pk>/status/', views.project_status_view, name='project_status'),
    path('projects/<int:pk>/reprocess/', views.project_reprocess_view, name='project_reprocess'),
    path('projects/<int:pk>/audio/original/', views.project_original_audio_view, name='project_original_audio'),
    path('projects/<int:pk>/tracks/<int:track_id>/audio/', views.project_track_audio_view, name='project_track_audio'),
    path('projects/<int:pk>/export-pdf/', views.export_pdf_view, name='export_pdf'),
    path('projects/<int:pk>/delete/', views.project_delete_view, name='project_delete'),
    path('projects/<int:pk>/score/edit/', views.score_edit_view, name='score_edit'),
    path('projects/<int:pk>/score/chord-edits/', views.score_chord_edits_view, name='score_chord_edits'),
]
