from rest_framework.test import APITestCase
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import RefreshToken
from django.urls import reverse


User = get_user_model()

class DashboardTest(APITestCase):

    def test_dashboard_access(self):
        user = User.objects.create_user(username="test", password="123456")
        refresh = RefreshToken.for_user(user)
        access_token = str(refresh.access_token)

        self.client.credentials(HTTP_AUTHORIZATION='Bearer ' + access_token)
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        
