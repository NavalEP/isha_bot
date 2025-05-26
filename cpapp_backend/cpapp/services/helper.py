import re

class Helper:
    """
    Utility class for helper functions.
    """
    @staticmethod
    def clean_url(url: str) -> str:
        """
        Remove common invisible Unicode characters from URLs
        
        Args:
            url: URL string to clean
            
        Returns:
            Cleaned URL string
        """
        if not isinstance(url, str):
            # Or raise TypeError, depending on desired behavior for non-string input
            return str(url) 
            
        # Remove common invisible Unicode characters
        invisible_chars = [
            '\u200B',  # Zero Width Space
            '\u200C',  # Zero Width Non-Joiner
            '\u200D',  # Zero Width Joiner
            '\uFEFF',  # Zero Width No-Break Space
            '\u2060',  # Word Joiner
        ]
        
        cleaned_url = url
        for char in invisible_chars:
            cleaned_url = cleaned_url.replace(char, '')
        
        return cleaned_url