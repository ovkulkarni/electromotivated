"""vthacks URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from users.views import RegistrationView, LoginView, ProfileView, LogoutView, IndexView
from circuits.views import CircuitImageView, CircuitProcessedImageView, CircuitDetailsView, CircuitUploadView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('admin/', admin.site.urls),
    path('register/', RegistrationView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('profile/', ProfileView.as_view(), name='user_profile'),
    path('circuit/upload/', CircuitUploadView.as_view(), name='circuit_upload'),
    path('circuit/image/<uuid:uuid>/',
         CircuitImageView.as_view(), name='circuit_image'),
    path('circuit/image/<uuid:uuid>/processed/',
         CircuitProcessedImageView.as_view(), name='circuit_image_processed'),
    path('circuit/details/',
         CircuitDetailsView.as_view(), name='circuit_details_truncated'),
    path('circuit/details/<uuid:uuid>/',
         CircuitDetailsView.as_view(), name='circuit_details'),
]
