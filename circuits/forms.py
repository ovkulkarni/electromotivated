from django.forms import ModelForm

from .models import Circuit


class CircuitForm(ModelForm):
    class Meta:
        model = Circuit
        fields = ['name', 'original_image']
