from django.forms import ModelForm

from .models import Circuit, Comment


class CircuitForm(ModelForm):
    class Meta:
        model = Circuit
        fields = ['name', 'original_image']


class CommentForm(ModelForm):
    class Meta:
        model = Comment
        fields = ['content']
