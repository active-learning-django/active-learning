# Generated by Django 3.0.2 on 2020-02-29 22:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imagelabeling', '0003_imagelabel_title'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagelabel',
            name='abnormal_votes',
            field=models.IntegerField(default='1'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='imagelabel',
            name='normal_votes',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='imagelabel',
            name='unknown_votes',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
    ]