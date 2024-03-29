from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, Http404

from .models import Circuit, Node
from .forms import CircuitForm, CommentForm

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
        # if not circuit.processed_image:
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
        nodes = []
        counts = {
            'resistor': 1,
            'capacitor': 1,
            'inductor': 1,
            'emf': 1
        }
        for node in circuit.node_set.all():
            x = {'obj': node}
            if request.user == circuit.user or request.user.is_superuser:
                if node.node_type == "resistor":
                    x['label'] = 'R<sub>{}</sub>'.format(counts['resistor'])
                    counts['resistor'] += 1
                if node.node_type == "inductor":
                    x['label'] = 'L<sub>{}</sub>'.format(counts['inductor'])
                    counts['inductor'] += 1
                if node.node_type == "capacitor":
                    x['label'] = 'C<sub>{}</sub>'.format(counts['capacitor'])
                    counts['capacitor'] += 1
                if node.node_type.endswith("battery"):
                    x['label'] = 'ℰ<sub>{}</sub>'.format(counts['emf'])
                    counts['emf'] += 1
            nodes.append(x)
        types = {}
        for node in circuit.node_set.exclude(node_type="corner"):
            if node.node_type.replace('right', '').replace('left', '').replace('top', '').replace('bottom', '') not in types:
                types[node.node_type.replace('right', '').replace(
                    'left', '').replace('top', '').replace('bottom', '')] = 0
            types[node.node_type.replace('right', '').replace(
                'left', '').replace('top', '').replace('bottom', '')] += 1
        return render(request, 'circuits/details.html', {'circuit': circuit, 'nodes': nodes, 'labeled': list(filter(lambda x: x.get('label'), nodes)), 'types': types.items()})

    def post(self, request, *args, **kwargs):
        uuid = kwargs.get('uuid', -1)
        circuit = get_object_or_404(Circuit, uuid=uuid)
        comment_form = CommentForm(request.POST)
        if comment_form.is_valid():
            cmt = comment_form.save(commit=False)
            cmt.user = request.user
            cmt.circuit = circuit
            cmt.save()
        if circuit.user == request.user or request.user.is_superuser:
            for field in request.POST:
                if field.startswith("node"):
                    pk = int(field.split("_")[-1])
                    node = get_object_or_404(Node, pk=pk)
                    if node.circuit == circuit:
                        node.value = request.POST.get(field)
                        node.save()
        return redirect("/circuit/details/{}/".format(circuit.uuid))


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


class AnalyticsView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        if not request.user.is_superuser:
            raise Http404
        return render(request, "analytics.html", {})
