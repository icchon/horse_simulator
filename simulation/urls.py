from django.urls import path
from .import views

app_name = "simulation"

urlpatterns = [
  path("", views.home, name="home"),
  path("simulation", views.simulation, name="simulation"),
]