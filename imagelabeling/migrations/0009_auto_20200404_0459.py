# Generated by Django 3.0.2 on 2020-04-04 04:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imagelabeling', '0008_auto_20200404_0455'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagelabel',
            name='image_file',
            field=models.ImageField(upload_to='../'),
        ),
    ]
