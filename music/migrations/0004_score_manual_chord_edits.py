from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('music', '0003_musicproject_artist'),
    ]

    operations = [
        migrations.AddField(
            model_name='score',
            name='manual_chord_edits',
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
