import random
import string
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def generate_short_code(length=7):
    """
    Generate a random short code for URL shortening
    
    Args:
        length: Length of the short code (default: 7)
        
    Returns:
        Random string of specified length
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def shorten_url(long_url: str) -> str:
    """
    Shorten a long URL by creating a short code mapping
    
    Args:
        long_url: The long URL to shorten
        
    Returns:
        Shortened URL in format http://carepay.money/s/{short_code}
    """
    try:
        from cpapp.models.shortlink import ShortLink
        
        # Check if already exists
        existing = ShortLink.objects.filter(long_url=long_url).first()
        if existing:
            logger.info(f"Found existing short link for URL: {existing.short_code}")
            return f"http://carepay.money/s/{existing.short_code}"
        
        # Generate new short code
        short_code = generate_short_code()
        while ShortLink.objects.filter(short_code=short_code).exists():
            short_code = generate_short_code()
        
        # Save mapping
        ShortLink.objects.create(long_url=long_url, short_code=short_code)
        logger.info(f"Created new short link: {short_code} for URL: {long_url}")
        
        return f"http://carepay.money/s/{short_code}"
        
    except Exception as e:
        logger.error(f"Error shortening URL: {e}")
        # Return original URL if shortening fails
        return long_url

def get_long_url(short_code: str) -> Optional[str]:
    """
    Get the long URL from a short code
    
    Args:
        short_code: The short code to look up
        
    Returns:
        The long URL if found, None otherwise
    """
    try:
        from cpapp.models.shortlink import ShortLink
        
        link = ShortLink.objects.filter(short_code=short_code).first()
        if link:
            return link.long_url
        return None
        
    except Exception as e:
        logger.error(f"Error getting long URL for short code {short_code}: {e}")
        return None 