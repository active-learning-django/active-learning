# Generated by Django 3.0.4 on 2020-04-29 04:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imagelabeling', '0030_alpha'),
    ]

    operations = [
        migrations.AlterField(
            model_name='alpha',
            name='alpha',
            field=models.CharField(max_length=100),
        ),
    ]
