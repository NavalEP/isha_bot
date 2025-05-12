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
            
            # Extract the phone number from the payload
            phone_number = payload.get('phone_number')
            
            if not phone_number:
                raise AuthenticationFailed('Invalid token payload')
            
            # Extract doctor details if available
            doctor_id = payload.get('doctor_id')
            doctor_name = payload.get('doctor_name')
                
            # Return the user identifier (phone_number), auth info, and doctor details
            auth_info = {
                'token': token,
                'doctor_id': doctor_id,
                'doctor_name': doctor_name
            }
            
            return (phone_number, auth_info)
            
        except IndexError:
            raise AuthenticationFailed('Token prefix missing')
        except ExpiredSignatureError:
            raise AuthenticationFailed('Token has expired')
        except InvalidTokenError:
            raise AuthenticationFailed('Invalid token')
            
    def authenticate_header(self, request):
        return 'Bearer' 