from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import InterviewSession
from .serializers import InterviewSessionSerializer

class DashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        sessions = InterviewSession.objects.filter(user=request.user)
        serializer = InterviewSessionSerializer(sessions, many=True)

        return Response({
            "message": "Welcome to Dashboard",
            "username": request.user.username,
            "sessions": serializer.data
        })
