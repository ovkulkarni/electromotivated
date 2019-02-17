from django.shortcuts import render, redirect
from django.views import View
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.views import LoginView as AuthLoginView, LogoutView as AuthLogoutView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages


class RegistrationView(View):
    template_name = 'users/register.html'
    form = UserCreationForm

    def post(self, request, *args, **kwargs):
        form = self.form(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Successfully registered!")
            return redirect('login')
        return render(request, self.template_name, {'form': form})

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, {'form': self.form()})


class LoginView(AuthLoginView):
    template_name = 'users/login.html'


class LogoutView(AuthLogoutView):
    template_name = 'users/logout.html'


class ProfileView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        if not request.user.circuit_set.exists():
            return redirect("circuit_upload")
        return render(request, 'users/profile.html', {})


class IndexView(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'index.html', {})
