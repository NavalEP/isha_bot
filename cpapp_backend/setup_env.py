import os
from dotenv import load_dotenv
from pathlib import Path

# This script sets environment variables in memory
# You can import this at the beginning of manage.py or other entry points

def setup_environment(use_mock=True, custom_api_url=None):
    """
    Set up environment variables for the application
    
    Args:
        use_mock: Whether to use mock APIs (default: True)
        custom_api_url: Custom API base URL to use instead of default
    """
    # Load environment variables from .env file if it exists
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
    
    # OpenAI API Key - Get from environment variable
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or system environment variables.")
    
    # Set mock API flag based on parameter
    os.environ["MOCK_API"] = str(use_mock).lower()
    
    # Carepay API configuration
    if custom_api_url:
        os.environ["CAREPAY_API_BASE_URL"] = custom_api_url
    else:
        os.environ["CAREPAY_API_BASE_URL"] = os.getenv("CAREPAY_API_BASE_URL", "https://backend.carepay.money")
        
    os.environ["DOCTOR_ID"] = os.getenv("DOCTOR_ID", "e71779851b144d1d9a25a538a03612fc")
    os.environ["DOCTOR_NAME"] = os.getenv("DOCTOR_NAME", "Nikhil_Salkar")
    
    # Development settings
    os.environ["TEST_OTP"] = os.getenv("TEST_OTP", "123456")  # Default test OTP for development
    
    print(f"Environment variables set successfully. Mock API: {use_mock}, API URL: {os.environ['CAREPAY_API_BASE_URL']}")

# Run this if script is executed directly
if __name__ == "__main__":
    setup_environment() 