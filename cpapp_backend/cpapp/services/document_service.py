import os
import logging
from typing import Tuple, Optional
import mimetypes

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Service for handling document uploads and validation
    """
    
    def __init__(self):
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.pdf', '.webp'}
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_mime_types = {
            'image/jpeg',
            'image/jpg', 
            'image/png',
            'image/webp',
            'application/pdf'
        }
    
    def validate_file(self, file_obj) -> Tuple[bool, str]:
        """
        Validate uploaded file
        
        Args:
            file_obj: Uploaded file object
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check file size
            if file_obj.size > self.max_file_size:
                return False, f"File size exceeds maximum limit of {self.max_file_size // (1024*1024)}MB"
            
            # Check file extension
            file_name = file_obj.name.lower()
            file_extension = os.path.splitext(file_name)[1]
            
            if file_extension not in self.allowed_extensions:
                return False, f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}"
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(file_name)
            if mime_type and mime_type not in self.allowed_mime_types:
                return False, f"MIME type {mime_type} not allowed"
            
            return True, "File is valid"
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return False, f"Error validating file: {str(e)}"
    
    def get_file_type(self, file_obj) -> str:
        """
        Get the MIME type of the file
        
        Args:
            file_obj: Uploaded file object
            
        Returns:
            MIME type string
        """
        try:
            file_name = file_obj.name.lower()
            mime_type, _ = mimetypes.guess_type(file_name)
            return mime_type or 'application/octet-stream'
        except Exception as e:
            logger.error(f"Error getting file type: {e}")
            return 'application/octet-stream' 