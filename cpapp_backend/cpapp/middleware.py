class IframeEmbeddingMiddleware:
    """
    Middleware to allow iframe embedding by setting appropriate headers
    """
    
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # Remove X-Frame-Options header to allow iframe embedding
        if 'X-Frame-Options' in response:
            del response['X-Frame-Options']
        
        # Set Content-Security-Policy to allow frame-ancestors
        response['Content-Security-Policy'] = "frame-ancestors 'self' *"
        
        # Additional headers for iframe compatibility
        response['X-Content-Type-Options'] = 'nosniff'
        
        return response
