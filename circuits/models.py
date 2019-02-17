from django.db import models

import uuid

NODE_TYPES = [
    ('battery', 'Battery'),
    ('resistor', 'Resistor'),
    ('corner', 'Corner'),
    ('capacitor', 'Capacitor')
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
    circuit = models.ForeignKey('circuits.Circuit', on_delete=models.CASCADE)
    connected_to = models.ManyToManyField('self', related_name='+')
    node_type = models.CharField(max_length=256, choices=NODE_TYPES)
    value = models.CharField(max_length=2048, null=True, blank=True)
    x = models.IntegerField(null=True, blank=True)
    y = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return "{} at ({}, {})".format(self.node_type, self.x, self.y)


class Comment(models.Model):
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    circuit = models.ForeignKey('circuits.Circuit', on_delete=models.CASCADE)
    content = models.CharField(max_length=4096)
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date']
