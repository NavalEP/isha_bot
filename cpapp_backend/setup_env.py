import os

# This script sets environment variables in memory
# You can import this at the beginning of manage.py or other entry points

def setup_environment(use_mock=True, custom_api_url=None):
    """
    Set up environment variables for the application
    
    Args:
        use_mock: Whether to use mock APIs (default: True)
        custom_api_url: Custom API base URL to use instead of default
    """
    # OpenAI API Key - Replace with your actual key
    os.environ["OPENAI_API_KEY"] = "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"  # Replace with actual key before using
    
    # Set mock API flag based on parameter
    os.environ["MOCK_API"] = str(use_mock).lower()
    
    # Carepay API configuration
    if custom_api_url:
        os.environ["CAREPAY_API_BASE_URL"] = custom_api_url
    else:
        os.environ["CAREPAY_API_BASE_URL"] = "https://backend.carepay.money"
        
    os.environ["DOCTOR_ID"] = "e71779851b144d1d9a25a538a03612fc"
    os.environ["DOCTOR_NAME"] = "Nikhil_Salkar"
    
    # Development settings
    os.environ["TEST_OTP"] = "123456"  # Default test OTP for development
    
    print(f"Environment variables set successfully. Mock API: {use_mock}, API URL: {os.environ['CAREPAY_API_BASE_URL']}")

# Run this if script is executed directly
if __name__ == "__main__":
    setup_environment() 