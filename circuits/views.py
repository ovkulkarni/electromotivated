from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse

from .models import Circuit, Node
from .forms import CircuitForm

from process_image import process

from cv2 import imread

from render_circuit import render_image
import re


class CircuitImageView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        uuid = kwargs.get('uuid', -1)
        circuit = get_object_or_404(Circuit, uuid=uuid)
        resp = HttpResponse()
        resp.content = circuit.original_image.read()
        return resp


class CircuitProcessedImageView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        uuid = kwargs.get('uuid', -1)
        circuit = get_object_or_404(Circuit, uuid=uuid)
        if not circuit.processed_image:
            new_fn = '{}.processed.svg'.format(
                circuit.original_image.path.rsplit('.', 1)[0])
            render_image(circuit.original_image.path, new_fn)
            circuit.processed_image = new_fn
            circuit.save()
        resp = HttpResponse()
        resp.content = re.sub(r'height=".*?"', 'height="95%"', re.sub(r'width=".*?"', 'width="100%"', circuit.processed_image.read().decode().replace(
            '<svg', '<svg preserveAspectRatio="xMidYMin"'))).encode()
        return resp


class CircuitDetailsView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        uuid = kwargs.get('uuid', -1)
        circuit = get_object_or_404(Circuit, uuid=uuid)
        s = ""
        if circuit.node_set.count() == 0:
            out = process(imread(circuit.original_image.path, 0))
            for node in out:
                node.loc = list(node.loc) if node.loc else None
            for node in out:
                node.object = Node.objects.create(
                    circuit=circuit, node_type=node.component, x=node.loc[0] if node.loc else None, y=node.loc[1] if node.loc else None)
            for node in out:
                for adj in node.adjs:
                    node.object.connected_to.add(out[adj].object)
            s += str(out)
            s += "\n"
        for node in circuit.node_set.all():
            s += "Node {} is connected to {}<br />".format(
                node, list(str(x) for x in node.connected_to.all()))
        return render(request, 'circuits/details.html', {'circuit': circuit, 'nodes': circuit.node_set.all()})


class CircuitUploadView(LoginRequiredMixin, View):
    template_name = 'circuits/upload.html'
    form = CircuitForm

    def post(self, request, *args, **kwargs):
        form = self.form(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.user = request.user
            obj.save()
            return redirect('user_profile')
        return render(request, self.template_name, {'form': form})

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, {'form': self.form()})
