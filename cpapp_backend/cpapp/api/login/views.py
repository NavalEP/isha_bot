from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from cpapp.services.api_client import CarepayAPIClient
from django.conf import settings
from cpapp.models.session_data import SessionData
from django.db import models
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
        
        # If doctor_id and doctor_name are not provided, try to find them from existing sessions
        if not doctor_id or not doctor_name:
            try:
                # Search for sessions where phone_number exists in session data
                sessions = SessionData.objects.filter(
                    models.Q(data__phoneNumber=phone_number) |
                    models.Q(data__mobileNumber=phone_number) |
                    models.Q(phone_number=phone_number)
                ).order_by('-created_at')
                
                if sessions.exists():
                    # Get the most recent session
                    latest_session = sessions.first()
                    session_data_dict = latest_session.data or {}
                    
                    # Extract doctor information from session data
                    found_doctor_id = session_data_dict.get('doctor_id') or session_data_dict.get('doctorId')
                    found_doctor_name = session_data_dict.get('doctor_name') or session_data_dict.get('doctorName')
                    
                    # Use found doctor information if available
                    if found_doctor_id and found_doctor_name:
                        doctor_id = found_doctor_id
                        doctor_name = found_doctor_name
                        print(f"Found doctor info from session: doctor_id={doctor_id}, doctor_name={doctor_name}")
                    else:
                        # If no doctor info found in sessions, require them to be provided
                        if not doctor_id or not doctor_name:
                            return Response(
                                {"error": "Doctor ID and doctor name are required. No existing sessions found with this phone number."}, 
                                status=status.HTTP_400_BAD_REQUEST
                            )
                else:
                    # If no sessions found, require doctor info to be provided
                    if not doctor_id or not doctor_name:
                        return Response(
                            {"error": "Doctor ID and doctor name are required. No existing sessions found with this phone number."}, 
                            status=status.HTTP_400_BAD_REQUEST
                        )
            except Exception as e:
                print(f"Error searching for existing sessions: {e}")
                # If there's an error searching sessions, require doctor info to be provided
                if not doctor_id or not doctor_name:
                    return Response(
                        {"error": "Doctor ID and doctor name are required"}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
        # Initialize the CarePay API client
        api_client = CarepayAPIClient()
        
        # Verify the OTP
        response = api_client.verify_otp(phone_number, otp)

        user_id = response.get('data')
        
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
            'doctor_id': doctor_id,
            'doctor_name': doctor_name,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
        }
        
        token = jwt.encode(
            payload, 
            settings.SECRET_KEY, 
            algorithm='HS256'
        )
        
        response_data = {
            "message": "OTP verified successfully",
            "token": token,
            "phone_number": phone_number,
            "doctor_id": doctor_id,
            "doctor_name": doctor_name,
            "userId": user_id
        }
        print(response_data)
        
        return Response(response_data, status=status.HTTP_200_OK)


class DoctorStaffView(APIView):
    """
    API view for doctor staff login
    """
    def post(self, request):
        doctor_code = request.data.get('doctor_code')
        password = request.data.get('password')
        
        if not doctor_code or not password:
            return Response(
                {"error": "Doctor code and password are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Initialize the CarePay API client
        api_client = CarepayAPIClient()

        response = api_client.login_with_password(doctor_code, password)
        
        # Check if the request was successful first
        if response.get('status', 0) >= 400:
            return Response(
                {"error": "Invalid doctor code or password"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Access the nested data from the response
        data = response.get('data', {})
        doctor_id = data.get('doctorId')
        doctor_name = data.get('doctorName')

        payload = {
            'doctor_id': doctor_id,
            'doctor_name': doctor_name,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }

        token = jwt.encode(payload, settings.SECRET_KEY, algorithm='HS256') 

        response_data = {
            "message": "Login successful",
            "token": token,
            "doctor_id": doctor_id,
            "doctor_name": doctor_name
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        