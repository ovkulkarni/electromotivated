# Generated by Django 2.1.7 on 2019-02-16 02:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('circuits', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='circuit',
            name='name',
            field=models.CharField(default='lol', max_length=256),
            preserve_default=False,
        ),
    ]
