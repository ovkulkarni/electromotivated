from django.db import models

import uuid

NODE_TYPES = [
    ('battery', 'Battery'),
    ('resistor', 'Resistor'),
    ('joint', 'Joint')
]


class Circuit(models.Model):
    name = models.CharField(max_length=256)
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    date_created = models.DateTimeField(auto_now_add=True)
    original_image = models.ImageField()
    processed_image = models.ImageField(blank=True, null=True)
    uuid = models.UUIDField(default=uuid.uuid4)

    class Meta:
        ordering = ['-date_created']


class Node(models.Model):
    connected_to = models.ManyToManyField('self', related_name='+')
    node_type = models.CharField(max_length=256, choices=NODE_TYPES)
    x = models.IntegerField()
    y = models.IntegerField()
