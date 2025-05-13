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
        doctor_id = request.data.get('doctorId')
        doctor_name = request.data.get('doctorName')
        
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
            
        # Generate JWT token
        payload = {
            'phone_number': phone_number,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }
        
        # Only add doctor information to the payload if it exists
        if doctor_id:
            payload['doctor_id'] = doctor_id
        

        if doctor_name:
            payload['doctor_name'] = doctor_name
        
        token = jwt.encode(
            payload, 
            settings.SECRET_KEY, 
            algorithm='HS256'
        )
        
        response_data = {
            "message": "OTP verified successfully",
            "token": token,
            "phone_number": phone_number
        }
        print(response_data)
        
        # Only include doctor information in the response if it exists
        if doctor_id:
            response_data["doctor_id"] = doctor_id
        
        if doctor_name:
            response_data["doctor_name"] = doctor_name
        
        return Response(response_data, status=status.HTTP_200_OK)
