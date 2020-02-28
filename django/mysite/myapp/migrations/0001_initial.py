# Generated by Django 3.0.3 on 2020-02-28 02:02

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Label',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('op', models.CharField(choices=[('Abnormal', 'Abnormal'), ('Normal', 'Normal'), ('Not Sure', 'Not Sure')], max_length=1200)),
                ('pic', models.ImageField(upload_to='images/')),
            ],
        ),
    ]
