from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('music', '0002_musicproject_completed_at_musicproject_error_message_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='musicproject',
            name='artist',
            field=models.CharField(blank=True, max_length=255),
        ),
    ]
