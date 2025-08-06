import json
import logging
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import redirect
from django.http import Http404

from asgiref.sync import async_to_sync
import datetime
# Import and set up environment variables first
import os

from cpapp.services.agent import CarepayAgent
from cpapp.api.login.authentication import JWTAuthentication
from cpapp.services.url_shortener import get_long_url
from cpapp.services.api_client import CarepayAPIClient
import jwt
from django.conf import settings
from cpapp.models.session_data import SessionData
from uuid import UUID
from django.core.exceptions import ObjectDoesNotExist

logger = logging.getLogger(__name__)

# Initialize agent
carepay_agent = CarepayAgent()
# Initialize API client
api_client = CarepayAPIClient()


class ShortlinkRedirectView(APIView):
    """
    View for returning long URLs from short codes
    """
    
    def get(self, request, short_code):
        """
        Get the long URL for a short code
        
        Args:
            request: Django request object
            short_code: The short code from the URL
            
        Returns:
            JSON response containing the long URL
        """
        try:
            # Get the long URL from the short code
            long_url = get_long_url(short_code)
            
            if long_url:
                return JsonResponse({
                    "status": "success",
                    "long_url": long_url
                })
            else:
                # If short code not found, return 404
                return JsonResponse({
                    "status": "error",
                    "message": "Short link not found"
                }, status=404)
                
        except Exception as e:
            # Log the error and return 404
            logger.error(f"Error getting long URL for short code {short_code}: {e}")
            return JsonResponse({
                "status": "error", 
                "message": "Short link not found"
            }, status=404)


class ChatSessionView(APIView):
    """
    View for creating new chat sessions
    """
    authentication_classes = [JWTAuthentication]
   
    
    def post(self, request):
        try:
            # Check if user is authenticated
            if not request.user or request.user == 'AnonymousUser':
                return Response({
                    "status": "error",
                    "message": "Authentication required"
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            # Get user identifier from authenticated user (could be phone_number or doctor_id)
            user_identifier = request.user
            
            # Extract doctor information from the JWT token
            doctor_id = None
            doctor_name = None
            phone_number = None
            
            # Get the token from the Authorization header
            auth_header = request.META.get('HTTP_AUTHORIZATION')
            if auth_header:
                try:
                    token = auth_header.split(' ')[1]
                    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
                    
                    # Extract doctor information
                    doctor_id = payload.get('doctor_id')
                    doctor_name = payload.get('doctor_name')
                    
                    # Extract phone number if available
                    phone_number = payload.get('phone_number')

                    
                    # If user_identifier is a doctor_id, use it as phone_number for session creation
                    if not phone_number and doctor_id and user_identifier == doctor_id:
                        phone_number = f"{doctor_name}/{doctor_id}"  # Create a unique identifier for doctor sessions

    
                    
                    logger.info(f"Extracted doctor_id: {doctor_id}, doctor_name: {doctor_name}, phone_number: {phone_number} from JWT token")
                except Exception as e:
                    logger.error(f"Error extracting doctor info from token: {e}")
            
            # Create session with doctor information
            session_id = carepay_agent.create_session(doctor_id=doctor_id, doctor_name=doctor_name, phone_number=phone_number)
            print(f"session_id: {session_id} created_at: {datetime.datetime.now()} by user: {user_identifier}")
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
            # Check if user is authenticated
            if not request.user or request.user == 'AnonymousUser':
                return Response({
                    "status": "error",
                    "message": "Authentication required"
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            # Get user identifier from authenticated user (could be phone_number or doctor_id)
            user_identifier = request.user
            
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
            print(f"user: {user_identifier}")
            
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
    authentication_classes = [JWTAuthentication]

    def get(self, request, session_uuid):
        """
        Get session details by UUID
        """
        try:
            # Check if user is authenticated
            if not request.user or request.user == 'AnonymousUser':
                return Response({
                    "status": "error",
                    "message": "Authentication required"
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            user_identifier = request.user
            
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

                # Get userId from data field if it exists
                user_id = session.data.get('userId') if session.data else None
                
                # Return existing session data
                return Response({
                    "status": "success",
                    "session_id": str(session.session_id),  # Convert UUID to string
                    "phoneNumber": session.phone_number,
                    "status": session.status,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "history": history,
                    "userId": user_id
                })

            except ObjectDoesNotExist:
                return Response({
                    "message": "Session not found"
                }, status=200)

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

class UserDetailsView(APIView):
    """
    View for retrieving user details from session data and API calls
    """
    # authentication_classes = [JWTAuthentication]

    def get(self, request, session_uuid):
        """
        Get user details by session UUID
        
        Args:
            request: Django request object
            session_uuid: The session UUID from the URL
            
        Returns:
            JSON response containing user details
        """
        try:
            # Check if user is authenticated
            if not request.user or request.user == 'AnonymousUser':
                return Response({
                    "status": "error",
                    "message": "Authentication required"
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            user_identifier = request.user
            
            # Convert string UUID to UUID object
            try:
                session_uuid = UUID(str(session_uuid))
            except (ValueError, TypeError):
                return Response({
                    "status": "error",
                    "message": "Invalid session ID format"
                }, status=400)

            try:
                # Try to find existing session
                session = SessionData.objects.get(session_id=session_uuid)
                
                # Get userId from session data
                user_id = session.data.get('userId') if session.data else None
                
                if not user_id:
                    return Response({
                        "status": "error",
                        "message": "Start Conversation Careena first"
                    }, status=400)
                
                # Call API methods to get user details
                user_details_response = api_client.get_user_details_by_user_id(user_id)
                address_response = api_client.get_user_address_by_user_id(user_id)
                employment_response = api_client.get_user_employment_by_user_id(user_id)
                
                # Initialize response data
                response_data = {
                    "status": "success",
                    "session_id": str(session.session_id),
                    "user_id": user_id,
                    "user_details": {},
                    "address_details": {},
                    "employment_details": {}
                }
                
                # Process user details response
                if user_details_response.get("status") == 200 and user_details_response.get("data"):
                    user_data = user_details_response.get("data", {})
                    response_data["user_details"] = {
                        "firstName": user_data.get("firstName"),
                        "dateOfBirth": user_data.get("dateOfBirth"),
                        "emailId": user_data.get("emailId"),
                        "gender": user_data.get("gender"),
                        "aadhaarNo": user_data.get("aadhaarNo"),
                        "maritalStatus": user_data.get("maritalStatus"),
                        "panNo": user_data.get("panNo"),
                        "educationLevel": user_data.get("educationLevel"),
                        "mobileNumber": user_data.get("mobileNumber")
                    }
                
                # Process address response
                if address_response.get("status") == 200 and address_response.get("data"):
                    address_data = address_response.get("data", {})
                    response_data["address_details"] = {
                        "address": address_data.get("address"),
                        "state": address_data.get("state"),
                        "city": address_data.get("city"),
                        "pincode": address_data.get("pincode")
                    }
                
                # Process employment response
                if employment_response.get("status") == 200 and employment_response.get("data"):
                    employment_data = employment_response.get("data", {})
                    response_data["employment_details"] = {
                        "netTakeHomeSalary": employment_data.get("netTakeHomeSalary"),
                        "employmentType": employment_data.get("employmentType"),
                        "currentCompanyName": employment_data.get("currentCompanyName", None),
                        "nameOfBusiness": employment_data.get("nameOfBusiness", None), 
                        "workplacePincode": employment_data.get("workplacePincode")
                    }
                
                # Log the API responses for debugging
                logger.info(f"User details API response: {user_details_response}")
                logger.info(f"Address API response: {address_response}")
                logger.info(f"Employment API response: {employment_response}")
                
                return Response(response_data, status=status.HTTP_200_OK)

            except ObjectDoesNotExist:
                return Response({
                    "status": "error",
                    "message": "Session not found"
                }, status=404)

            except Exception as e:
                logger.error(f"Database error in user details: {e}")
                return Response({
                    "status": "error",
                    "message": "Internal server error while accessing session data",
                    "error_details": str(e) if settings.DEBUG else None
                }, status=500)

        except Exception as e:
            logger.error(f"Unexpected error in user details: {e}")
            return Response({
                "status": "error",
                "message": "An unexpected error occurred",
                "error_details": str(e) if settings.DEBUG else None
            }, status=500)

class SaveUserBasicDetailsView(APIView):
    """
    View for saving user basic details
    """
    authentication_classes = [JWTAuthentication]

    def post(self, request, session_uuid):
        """
        Save user basic details by session UUID
        
        Args:
            request: Django request object with basic details data
            session_uuid: The session UUID from the URL
            
        Returns:
            JSON response containing save status
        """
        try:
            # Check if user is authenticated
            if not request.user or request.user == 'AnonymousUser':
                return Response({
                    "status": "error",
                    "message": "Authentication required"
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            user_identifier = request.user
            
            # Convert string UUID to UUID object
            try:
                session_uuid = UUID(str(session_uuid))
            except (ValueError, TypeError):
                return Response({
                    "status": "error",
                    "message": "Invalid session ID format"
                }, status=400)

            try:
                # Try to find existing session
                session = SessionData.objects.get(session_id=session_uuid)
                
                # Get userId from session data
                user_id = session.data.get('userId') if session.data else None
                
                if not user_id:
                    return Response({
                        "status": "error",
                        "message": "User ID not found in session data"
                    }, status=400)
                
                # Get basic details from request data
                basic_details = request.data.get('basic_details', {})
                
                if not basic_details:
                    return Response({
                        "status": "error",
                        "message": "Basic details data is required"
                    }, status=400)
                
                # Prepare data for API call
                details_data = {
                    "firstName": basic_details.get("firstName"),
                    "dateOfBirth": basic_details.get("dateOfBirth"),
                    "emailId": basic_details.get("emailId"),
                    "maritalStatus": basic_details.get("maritalStatus"),
                    "gender": basic_details.get("gender"),
                    "panCard": basic_details.get("panNo"),  # Map panNo to panCard
                    "aadhaarNo": basic_details.get("aadhaarNo"),
                    "educationLevel": basic_details.get("educationLevel"),
                    "mobileNumber": session.data.get('phoneNumber') if session.data else None,
                    "formStatus": "completed"
                }
                
                # Call API to save basic details
                save_response = api_client.save_basic_details(user_id, details_data)
                
                if save_response.get("status") == 200:
                    return Response({
                        "status": "success",
                        "message": "Basic details saved successfully",
                        "session_id": str(session.session_id),
                        "user_id": user_id,
                        "api_response": save_response
                    }, status=status.HTTP_200_OK)
                else:
                    return Response({
                        "status": "error",
                        "message": "Failed to save basic details",
                        "api_response": save_response
                    }, status=status.HTTP_400_BAD_REQUEST)

            except ObjectDoesNotExist:
                return Response({
                    "status": "error",
                    "message": "Session not found"
                }, status=404)

            except Exception as e:
                logger.error(f"Database error in saving basic details: {e}")
                return Response({
                    "status": "error",
                    "message": "Internal server error while saving basic details",
                    "error_details": str(e) if settings.DEBUG else None
                }, status=500)

        except Exception as e:
            logger.error(f"Unexpected error in saving basic details: {e}")
            return Response({
                "status": "error",
                "message": "An unexpected error occurred",
                "error_details": str(e) if settings.DEBUG else None
            }, status=500)


class SaveUserAddressDetailsView(APIView):
    """
    View for saving user address details
    """
    authentication_classes = [JWTAuthentication]

    def post(self, request, session_uuid):
        """
        Save user address details by session UUID
        
        Args:
            request: Django request object with address details data
            session_uuid: The session UUID from the URL
            
        Returns:
            JSON response containing save status
        """
        try:
            # Check if user is authenticated
            if not request.user or request.user == 'AnonymousUser':
                return Response({
                    "status": "error",
                    "message": "Authentication required"
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            user_identifier = request.user
            
            # Convert string UUID to UUID object
            try:
                session_uuid = UUID(str(session_uuid))
            except (ValueError, TypeError):
                return Response({
                    "status": "error",
                    "message": "Invalid session ID format"
                }, status=400)

            try:
                # Try to find existing session
                session = SessionData.objects.get(session_id=session_uuid)
                
                # Get userId from session data
                user_id = session.data.get('userId') if session.data else None
                
                if not user_id:
                    return Response({
                        "status": "error",
                        "message": "User ID not found in session data"
                    }, status=400)
                
                # Get address details from request data
                address_details = request.data.get('address_details', {})
                
                if not address_details:
                    return Response({
                        "status": "error",
                        "message": "Address details data is required"
                    }, status=400)
                
                # Prepare data for API call
                address_data = {
                    "address": address_details.get("address"),
                    "state": address_details.get("state"),
                    "city": address_details.get("city"),
                    "pincode": address_details.get("pincode"),
                    "formStatus": "completed"
                }
                
                # Call API to save address details
                save_response = api_client.save_address_details(user_id, address_data)
                
                if save_response.get("status") == 200:
                    return Response({
                        "status": "success",
                        "message": "Address details saved successfully",
                        "session_id": str(session.session_id),
                        "user_id": user_id,
                        "api_response": save_response
                    }, status=status.HTTP_200_OK)
                else:
                    return Response({
                        "status": "error",
                        "message": "Failed to save address details",
                        "api_response": save_response
                    }, status=status.HTTP_400_BAD_REQUEST)

            except ObjectDoesNotExist:
                return Response({
                    "status": "error",
                    "message": "Session not found"
                }, status=404)

            except Exception as e:
                logger.error(f"Database error in saving address details: {e}")
                return Response({
                    "status": "error",
                    "message": "Internal server error while saving address details",
                    "error_details": str(e) if settings.DEBUG else None
                }, status=500)

        except Exception as e:
            logger.error(f"Unexpected error in saving address details: {e}")
            return Response({
                "status": "error",
                "message": "An unexpected error occurred",
                "error_details": str(e) if settings.DEBUG else None
            }, status=500)


class SaveUserEmploymentDetailsView(APIView):
    """
    View for saving user employment details
    """
    authentication_classes = [JWTAuthentication]

    def post(self, request, session_uuid):
        """
        Save user employment details by session UUID
        
        Args:
            request: Django request object with employment details data
            session_uuid: The session UUID from the URL
            
        Returns:
            JSON response containing save status
        """
        try:
            # Check if user is authenticated
            if not request.user or request.user == 'AnonymousUser':
                return Response({
                    "status": "error",
                    "message": "Authentication required"
                }, status=status.HTTP_401_UNAUTHORIZED)
            
            user_identifier = request.user
            
            # Convert string UUID to UUID object
            try:
                session_uuid = UUID(str(session_uuid))
            except (ValueError, TypeError):
                return Response({
                    "status": "error",
                    "message": "Invalid session ID format"
                }, status=400)

            try:
                # Try to find existing session
                session = SessionData.objects.get(session_id=session_uuid)
                
                # Get userId from session data
                user_id = session.data.get('userId') if session.data else None
                
                if not user_id:
                    return Response({
                        "status": "error",
                        "message": "User ID not found in session data"
                    }, status=400)
                
                # Get employment details from request data
                employment_details = request.data.get('employment_details', {})
                
                if not employment_details:
                    return Response({
                        "status": "error",
                        "message": "Employment details data is required"
                    }, status=400)
                
                # Prepare data for API call
                employment_data = {
                    "netTakeHomeSalary": employment_details.get("netTakeHomeSalary"),
                    "employmentType": employment_details.get("employmentType"),
                    "organizationName": employment_details.get("currentCompanyName", None), 
                    "nameOfBusiness": employment_details.get("nameOfBusiness", None), 
                    "workplacePincode": employment_details.get("workplacePincode"),
                    "formStatus": "completed"
                }
                
                # Call API to save employment details
                save_response = api_client.save_employment_details(user_id, employment_data)
                
                if save_response.get("status") == 200:
                    return Response({
                        "status": "success",
                        "message": "Employment details saved successfully",
                        "session_id": str(session.session_id),
                        "user_id": user_id,
                        "api_response": save_response
                    }, status=status.HTTP_200_OK)
                else:
                    return Response({
                        "status": "error",
                        "message": "Failed to save employment details",
                        "api_response": save_response
                    }, status=status.HTTP_400_BAD_REQUEST)

            except ObjectDoesNotExist:
                return Response({
                    "status": "error",
                    "message": "Session not found"
                }, status=404)

            except Exception as e:
                logger.error(f"Database error in saving employment details: {e}")
                return Response({
                    "status": "error",
                    "message": "Internal server error while saving employment details",
                    "error_details": str(e) if settings.DEBUG else None
                }, status=500)

        except Exception as e:
            logger.error(f"Unexpected error in saving employment details: {e}")
            return Response({
                "status": "error",
                "message": "An unexpected error occurred",
                "error_details": str(e) if settings.DEBUG else None
            }, status=500)