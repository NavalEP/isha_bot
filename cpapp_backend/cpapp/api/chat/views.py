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
            print(f"response: {response}")
            
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
    
    def get(self, request, session_uuid):
        """
        Get session details by UUID
        """
        try:
            # Convert string UUID to UUID object
            try:
                session_uuid = UUID(str(session_uuid))  # Ensure we're working with a string first
            except (ValueError, TypeError):
                return Response({
                    "status": "error",
                    "message": "Invalid session ID format"
                }, status=400)

            try:
                # Try to find existing active session
                    session = SessionData.objects.get(
                        session_id=session_uuid,
                    )
                    
                    # Convert history items to proper format if they exist
                    history = []
                    if session.history:
                        for item in session.history:
                            if isinstance(item, dict):
                                history.append(item)
                            else:
                                # Convert any non-dict items to dict format
                                history.append({
                                    'type': 'AIMessage' if isinstance(item, str) else 'HumanMessage',
                                    'content': str(item)
                                })
                    
                    # Return existing session data
                    return Response({
                        "status": "success",
                        "session_id": str(session.session_id),  # Convert UUID to string
                        "phoneNumber": session.phone_number,
                        "status": session.status,
                        "created_at": session.created_at,
                        "updated_at": session.updated_at,
                        "history": history
                    })
                    

            except Exception as e:
                logger.error(f"Database error in session details: {e}")
                return Response({
                    "status": "error",
                    "message": "Internal server error while accessing session data",
                    "error_details": str(e) if settings.DEBUG else None
                }, status=500)

        except Exception as e:
            logger.error(f"Unexpected error in session details: {e}")
            return Response({
                "status": "error",
                "message": "An unexpected error occurred",
                "error_details": str(e) if settings.DEBUG else None
            }, status=500)