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
import jwt
from django.conf import settings
from cpapp.models.session_data import SessionData
from uuid import UUID

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
            
            # Extract doctor information from the JWT token
            doctor_id = None
            doctor_name = None
            
            # Get the token from the Authorization header
            auth_header = request.META.get('HTTP_AUTHORIZATION')
            if auth_header:
                try:
                    token = auth_header.split(' ')[1]
                    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
                    doctor_id = payload.get('doctor_id')
                    doctor_name = payload.get('doctor_name')
                    logger.info(f"Extracted doctor_id: {doctor_id}, doctor_name: {doctor_name} from JWT token")
                except Exception as e:
                    logger.error(f"Error extracting doctor info from token: {e}")
            
            # Create session with doctor information
            session_id = carepay_agent.create_session(doctor_id=doctor_id, doctor_name=doctor_name, phone_number=phone_number)
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
            
            # Extract data from request
            session_id = request.data.get("session_id")
            message = request.data.get("message")
            
            # Validate inputs
            if not session_id or not message:
                return Response({
                    "status": "error",
                    "message": "session_id and message are required"
                }, status=status.HTTP_400_BAD_REQUEST)
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

class SessionDetailsView(APIView):
    """
    View for handling session details with UUID - either retrieves existing active session or creates new one
    """
    authentication_classes = [JWTAuthentication]
    
    def get(self, request, session_uuid):
        try:
            # Get phone number from authenticated user
            phone_number = request.user
            
            try:
                # Try to parse the UUID to validate format
                session_uuid = UUID(session_uuid)
            except ValueError:
                return Response({
                    "status": "error",
                    "message": "Invalid UUID format"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Try to find existing active session
            try:
                session = SessionData.objects.get(
                    session_id=session_uuid,
                    status='active',
                    phone_number=phone_number
                )
                
                # Return existing session data
                return Response({
                    "status": "success",
                    "session_id": str(session.session_id),
                    "data": {
                        "fullName": session.data.get("fullName"),
                        "phoneNumber": session.phone_number,
                        "bureau_decision_details": session.data.get("bureau_decision_details"),
                        "status": session.status
                    },
                    "created_at": session.created_at,
                    "updated_at": session.updated_at
                }, status=status.HTTP_200_OK)
                
            except SessionData.DoesNotExist:
                # Create new session since existing one not found
                # Extract doctor information from the JWT token
                doctor_id = None
                doctor_name = None
                
                # Get the token from the Authorization header
                auth_header = request.META.get('HTTP_AUTHORIZATION')
                if auth_header:
                    try:
                        token = auth_header.split(' ')[1]
                        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
                        doctor_id = payload.get('doctor_id')
                        doctor_name = payload.get('doctor_name')
                    except Exception as e:
                        logger.error(f"Error extracting doctor info from token: {e}")
                
                # Create new session
                new_session = SessionData.objects.create(
                    phone_number=phone_number,
                    session_id=session_uuid,
                    status='active',
                    data={
                        'doctor_id': doctor_id,
                        'doctor_name': doctor_name
                    }
                )
                
                return Response({
                    "status": "success",
                    "message": "New session created",
                    "session_id": str(new_session.session_id),
                    "data": {
                        "fullName": new_session.data.get("fullName"),
                        "phoneNumber": new_session.phone_number,
                        "bureau_decision_details": new_session.data.get("bureau_decision_details"),
                        "status": new_session.status
                    },
                    "created_at": new_session.created_at
                }, status=status.HTTP_201_CREATED)
                
        except Exception as e:
            logger.error(f"Error processing session details: {e}")
            return Response({
                "status": "error",
                "message": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)