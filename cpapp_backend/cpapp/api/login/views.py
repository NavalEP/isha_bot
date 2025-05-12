from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from cpapp.services.api_client import CarepayAPIClient
from django.conf import settings
import jwt
import datetime

class SendOtpView(APIView):
    """
    API view for sending OTP to the provided phone number
    """
    def post(self, request):
        phone_number = request.data.get('phone_number')
        
        if not phone_number:
            return Response(
                {"error": "Phone number is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Initialize the CarePay API client
        api_client = CarepayAPIClient()
        
        # Send OTP to the provided phone number
        response = api_client.send_otp(phone_number)
        
        # Check if the request was successful
        if response.get('status', 0) >= 400:
            return Response(
                {"error": response.get('error', 'Failed to send OTP')},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        return Response(
            {"message": "OTP sent successfully"}, 
            status=status.HTTP_200_OK
        )

class VerifyOtpView(APIView):
    """
    API view for verifying OTP and generating JWT token
    """
    def post(self, request):
        phone_number = request.data.get('phone_number')
        otp = request.data.get('otp')
        
        if not phone_number or not otp:
            return Response(
                {"error": "Phone number and OTP are required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Initialize the CarePay API client
        api_client = CarepayAPIClient()
        
        # Verify the OTP
        response = api_client.verify_otp(phone_number, otp)
        
        # Handle case when response is a string instead of a dictionary
        if isinstance(response, str):
            return Response(
                {"error": "Invalid response from server"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
        # Check if the verification was successful
        if response.get('status', 0) >= 400:
            return Response(
                {"error": "Invalid OTP or server error"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get doctor details
        doctor_details_response = api_client.get_doctor_details(phone_number)
        
        # Extract doctor ID and name
        doctor_id = None
        doctor_name = None
        
        if doctor_details_response.get("status") == 200 and doctor_details_response.get("data"):
            doctor_data = doctor_details_response.get("data")
            doctor_id = doctor_data.get("doctorId")
            doctor_name = doctor_data.get("name")
            
        # Generate JWT token with doctor details
        payload = {
            'phone_number': phone_number,
            'doctor_id': doctor_id,
            'doctor_name': doctor_name,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }
        
        token = jwt.encode(
            payload, 
            settings.SECRET_KEY, 
            algorithm='HS256'
        )
        
        return Response({
            "message": "OTP verified successfully",
            "token": token,
            "phone_number": phone_number,
            "doctor_id": doctor_id,
            "doctor_name": doctor_name
        }, status=status.HTTP_200_OK)
