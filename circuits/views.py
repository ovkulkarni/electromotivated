from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse

from .models import Circuit
from .forms import CircuitForm


class CircuitImageView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        uuid = kwargs.get('uuid', -1)
        circuit = get_object_or_404(Circuit, uuid=uuid)
        resp = HttpResponse()
        resp.content = circuit.original_image.read()
        return resp


class CircuitDetailsView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        uuid = kwargs.get('uuid', -1)
        circuit = get_object_or_404(Circuit, uuid=uuid)
        return HttpResponse(circuit.name)


class CircuitUploadView(LoginRequiredMixin, View):
    template_name = 'circuits/upload.html'
    form = CircuitForm

    def post(self, request, *args, **kwargs):
        form = self.form(request.POST)
        if form.is_valid():
            form.save()
            return redirect('user_profile')
        return render(request, self.template_name, {'form': form})

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, {'form': self.form()})
