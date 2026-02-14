from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

User = get_user_model()

class UserProfileTest(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass'
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_get_profile(self):
        response = self.client.get('/api/profile/')
        self.assertEqual(response.status_code, 200)

    def test_update_profile(self):
        response = self.client.put('/api/profile/', {'bio': 'Software Developer'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['bio'], 'Software Developer')

    def test_navigation_access(self):
        response = self.client.get(reverse('navigation'))
        self.assertEqual(response.status_code, 200)

