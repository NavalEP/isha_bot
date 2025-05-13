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
                
            # Return the user identifier (phone_number) and auth info
            return (phone_number, token)
            
        except IndexError:
            raise AuthenticationFailed('Token prefix missing')
        except ExpiredSignatureError:
            raise AuthenticationFailed('Token has expired')
        except InvalidTokenError:
            raise AuthenticationFailed('Invalid token')
            
    def authenticate_header(self, request):
        return 'Bearer' 