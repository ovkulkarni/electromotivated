from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse

from .models import Circuit, Node
from .forms import CircuitForm

from process_image import process

from cv2 import imread


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
        s = ""
        if circuit.node_set.count() == 0:
            out = process(imread(circuit.original_image.path, 0))
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
