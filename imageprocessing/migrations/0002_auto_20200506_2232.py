# Generated by Django 3.0.5 on 2020-05-06 17:02

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('imageprocessing', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='image',
            old_name='uploaded_image',
            new_name='image',
        ),
    ]
