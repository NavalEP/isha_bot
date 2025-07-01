from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from django.conf import settings
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError

class JWTAuthentication(BaseAuthentication):
    """
    Custom JWT authentication for the API
    """
    def authenticate(self, request):
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        
        if not auth_header:
            return None
            
        try:
            # Extract the token
            token = auth_header.split(' ')[1]
            
            # Decode the token
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
            
            # Check for phone_number (from OTP verification)
            phone_number = payload.get('phone_number')
            
            # Check for doctor_id (from doctor staff login)
            doctor_id = payload.get('doctor_id')
            
            # Use phone_number if available, otherwise use doctor_id as identifier
            user_identifier = phone_number if phone_number else doctor_id
            
            if not user_identifier:
                raise AuthenticationFailed('Invalid token payload - missing phone_number or doctor_id')
                
            # Return the user identifier and auth info
            return (user_identifier, token)
            
        except IndexError:
            raise AuthenticationFailed('Token prefix missing')
        except ExpiredSignatureError:
            raise AuthenticationFailed('Token has expired')
        except InvalidTokenError:
            raise AuthenticationFailed('Invalid token')
            
    def authenticate_header(self, request):
        return 'Bearer' 