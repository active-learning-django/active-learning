# Generated by Django 3.0.4 on 2020-04-04 04:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imagelabeling', '0010_delete_comment'),
    ]

    operations = [
        migrations.CreateModel(
            name='NumOfIteration',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('numInter', models.CharField(choices=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)], max_length=100)),
            ],
        ),
    ]
