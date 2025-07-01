from django.urls import path
from django.http import JsonResponse
from cpapp.api.chat.views import ChatSessionView, ChatMessageView, SessionDetailsView
from cpapp.api.login.views import SendOtpView, VerifyOtpView, DoctorStaffView


def api_root_view(request):
    return JsonResponse({"message": "Agent API root is working"})

urlpatterns = [
    path('', api_root_view, name='api_root'),
    path('session/', ChatSessionView.as_view(), name='create_session'),
    path('message/', ChatMessageView.as_view(), name='send_message'),
    path('login/send-otp/', SendOtpView.as_view(), name='send_otp'),
    path('login/verify-otp/', VerifyOtpView.as_view(), name='verify_otp'),
    path('login/doctor-staff/', DoctorStaffView.as_view(), name='doctor_staff'),
    path('session-details/<uuid:session_uuid>/', SessionDetailsView.as_view(), name='session_details'),
]