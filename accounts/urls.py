from django.urls import path
from .views import UserProfileView
from .views import NavigationView

urlpatterns = [
    path('profile/', UserProfileView.as_view(), name='user-profile'),
]

urlpatterns += [
    path('navigation/', NavigationView.as_view(), name='navigation'),
]

