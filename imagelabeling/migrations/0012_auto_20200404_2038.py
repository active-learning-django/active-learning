# Generated by Django 3.0.4 on 2020-04-04 20:38

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('imagelabeling', '0011_numofiteration'),
    ]

    operations = [
        migrations.RenameField(
            model_name='numofiteration',
            old_name='numInter',
            new_name='Iteration',
        ),
    ]
