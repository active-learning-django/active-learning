# Generated by Django 3.0.4 on 2020-04-29 05:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imagelabeling', '0034_auto_20200429_0531'),
    ]

    operations = [
        migrations.AlterField(
            model_name='alphainput',
            name='alpha_input',
            field=models.IntegerField(default=0),
        ),
    ]