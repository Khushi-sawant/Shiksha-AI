from rest_framework import serializers
from .models import UserProfile
from .models import NavigationItem

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = '__all__'
        read_only_fields = ['user']

        

class NavigationSerializer(serializers.ModelSerializer):
    class Meta:
        model = NavigationItem
        fields = '__all__'

