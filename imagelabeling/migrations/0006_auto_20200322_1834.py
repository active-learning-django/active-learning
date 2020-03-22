# Generated by Django 3.0.2 on 2020-03-22 18:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imagelabeling', '0005_auto_20200322_1813'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagelabel',
            name='model_classification',
            field=models.CharField(choices=[('0', '0'), ('1', '1'), ('Unknown', 'Unknown')], max_length=100),
        ),
        migrations.AlterField(
            model_name='machinelearningmodel',
            name='title',
            field=models.CharField(max_length=100, unique=True),
        ),
    ]
