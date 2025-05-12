import json
import logging
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from asgiref.sync import async_to_sync
import datetime
# Import and set up environment variables first
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from setup_env import setup_environment
setup_environment()

from cpapp.services.agent import CarepayAgent
from cpapp.api.login.authentication import JWTAuthentication

logger = logging.getLogger(__name__)

# Initialize agent
carepay_agent = CarepayAgent()


class ChatSessionView(APIView):
    """
    View for creating new chat sessions
    """
    authentication_classes = [JWTAuthentication]
   
    
    def post(self, request):
        try:
            # Get phone number from authenticated user
            phone_number = request.user
            
            # Get doctor details from authentication info
            auth_info = request.auth
            doctor_id = auth_info.get('doctor_id') if isinstance(auth_info, dict) else None
            doctor_name = auth_info.get('doctor_name') if isinstance(auth_info, dict) else None
            
            session_id = carepay_agent.create_session()
            
            # Store doctor details in the session if available
            if doctor_id and doctor_name:
                carepay_agent.sessions[session_id]["data"]["doctor_id"] = doctor_id
                carepay_agent.sessions[session_id]["data"]["doctor_name"] = doctor_name
                print(f"Stored doctor details in session: doctor_id={doctor_id}, doctor_name={doctor_name}")
            
            print(f"session_id: {session_id} created_at: {datetime.datetime.now()} by user: {phone_number}")
            return Response({
                "status": "success",
                "session_id": session_id
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return Response({
                "status": "error",
                "message": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatMessageView(APIView):
    """
    View for handling chat messages
    """
    authentication_classes = [JWTAuthentication]
  
    
    def post(self, request):
        try:
            # Get phone number from authenticated user
            phone_number = request.user
            
            # Get doctor details from authentication info
            auth_info = request.auth
            doctor_id = auth_info.get('doctor_id') if isinstance(auth_info, dict) else None
            doctor_name = auth_info.get('doctor_name') if isinstance(auth_info, dict) else None
            
            # Extract data from request
            session_id = request.data.get("session_id")
            message = request.data.get("message")
            
            # Validate inputs
            if not session_id or not message:
                return Response({
                    "status": "error",
                    "message": "session_id and message are required"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Update doctor details in the session if available
            if doctor_id and doctor_name and session_id in carepay_agent.sessions:
                carepay_agent.sessions[session_id]["data"]["doctor_id"] = doctor_id
                carepay_agent.sessions[session_id]["data"]["doctor_name"] = doctor_name
            
            print(f"session_id: {session_id}")  
            print(f"message: {message}")
            print(f"user: {phone_number}")
            
            # Process message with agent
            response = carepay_agent.run(session_id, message)
            
            # Return response
            return Response({
                "status": "success",
                "session_id": session_id,
                "response": response
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return Response({
                "status": "error",
                "message": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SessionStatusView(APIView):
    """
    View for retrieving session status
    """
    authentication_classes = [JWTAuthentication]
  
    
    def get(self, request, session_id):
        try:
            # Get phone number from authenticated user
            phone_number = request.user
            
            # Get session data
            session_data = carepay_agent.get_session_data(session_id)
            
            # Check if session exists
            if session_data == "Session ID not found":
                return Response({
                    "status": "error",
                    "message": "Session not found"
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Parse session data
            session_data = json.loads(session_data)
            
            # Return session status
            return Response({
                "status": "success",
                "session_id": session_id,
                "session_status": session_data.get("status"),
                "user_id": session_data.get("user_id")
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return Response({
                "status": "error",
                "message": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)