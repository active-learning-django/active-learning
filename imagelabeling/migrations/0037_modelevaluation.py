# Generated by Django 3.0.4 on 2020-05-03 21:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imagelabeling', '0036_auto_20200429_0537'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelEvaluation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('alpha_value', models.FloatField(default=0)),
                ('r_score', models.FloatField(default=0)),
                ('auc', models.FloatField(default=0)),
            ],
        ),
    ]