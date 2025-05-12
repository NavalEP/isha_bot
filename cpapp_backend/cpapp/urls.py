from django.urls import path
from cpapp.api.chat.views import ChatSessionView, ChatMessageView, SessionStatusView
from cpapp.api.login.views import SendOtpView, VerifyOtpView

urlpatterns = [
    path('session/', ChatSessionView.as_view(), name='create_session'),
    path('message/', ChatMessageView.as_view(), name='send_message'),
    path('session/<str:session_id>/', SessionStatusView.as_view(), name='session_status'),
    path('login/send-otp/', SendOtpView.as_view(), name='send_otp'),
    path('login/verify-otp/', VerifyOtpView.as_view(), name='verify_otp'),
]