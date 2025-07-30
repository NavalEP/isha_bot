import openai
import base64
import json
import numpy as np
from PIL import Image
import io
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)

# Initialize the client with API key from environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_pincode_from_text(text: str) -> str:
    """
    Extract pincode from text using different keywords and patterns
    
    Args:
        text: Text containing address information
        
    Returns:
        Extracted pincode as string, empty string if not found
    """
    import re
    
    if not text:
        return ""
    
    # Different keywords that might indicate pincode
    pincode_keywords = [
        'pincode', 'pin code', 'pin-code', 'postal code', 'postalcode',
        'zip code', 'zipcode', 'pin', 'code', 'postal', 'zip',
        'pincd', 'pcd', 'pin cd', 'post cd', 'postal cd'
    ]
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Look for patterns like "PIN: 123456" or "Pincode: 123456"
    for keyword in pincode_keywords:
        # Pattern 1: keyword followed by colon/semicolon and 6 digits
        pattern1 = rf'{re.escape(keyword)}[:\s]*(\d{{6}})'
        match1 = re.search(pattern1, text_lower)
        if match1:
            return match1.group(1)
        
        # Pattern 2: keyword followed by 6 digits
        pattern2 = rf'{re.escape(keyword)}\s*(\d{{6}})'
        match2 = re.search(pattern2, text_lower)
        if match2:
            return match2.group(1)
    
    # Look for 6-digit numbers that might be pincodes
    # Common patterns: 6 digits at the end of lines, after state names, etc.
    digit_patterns = [
        r'(\d{6})',  # Any 6-digit number
        r'(\d{6})\s*$',  # 6 digits at end of line
        r'(\d{6})\s*[A-Z]',  # 6 digits followed by letter
    ]
    
    for pattern in digit_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Validate that it's likely a pincode (not Aadhaar number, etc.)
            pincode = str(match)
            # Pincodes typically start with 1-9 (not 0)
            if pincode[0] != '0' and len(pincode) == 6:
                return pincode
    
    return ""

def extract_aadhaar_details(image_path: str) -> dict:
    """
    Extract Aadhaar card details using OpenAI GPT-4 Vision API
    
    Args:
        image_path: Path to the Aadhaar card image
        
    Returns:
        Dictionary containing extracted Aadhaar details
    """
    try:
        # Read and encode the image
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Call GPT-4 Vision to extract info
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in extracting ID information. Extract all details accurately and return in JSON format. Pay special attention to separating relationship information from address."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the following details from the Aadhaar card image and return as JSON: full_name, aadhaar_number, date_of_birth, gender, address, pincode, father_name, husband_name. For the address field, exclude relationship prefixes like 'S/O', 'W/O', 'D/O', 'H/O' - these should go in father_name or husband_name fields. The address should only contain the actual location details (village, post office, district, etc.). Extract the pincode as a separate field - look for 6-digit numbers that appear at the end of address lines or near postal information. Return only valid JSON without any additional text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1000
        )

        # Parse and return the result
        content = response.choices[0].message.content.strip()
        
        # Remove markdown formatting if present
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        result = json.loads(content)
        
        # Map the extracted fields to the expected format
        mapped_result = {
            'name': result.get('full_name', ''),
            'aadhaar_number': result.get('aadhaar_number', ''),
            'dob': result.get('date_of_birth', ''),
            'gender': result.get('gender', ''),
            'address': result.get('address', ''),
            'pincode': result.get('pincode', ''),
            'father_name': result.get('father_name', ''),
            'husband_name': result.get('husband_name', '')
        }
        
        # Clean and validate pincode
        pincode = mapped_result.get('pincode', '')
        if pincode:
            # Remove any non-digit characters and ensure it's 6 digits
            pincode = ''.join(filter(str.isdigit, str(pincode)))
            if len(pincode) == 6:
                mapped_result['pincode'] = pincode
            else:
                mapped_result['pincode'] = ''
        
        # If pincode is not found in the main extraction, try to extract it from address
        if not mapped_result.get('pincode'):
            address = mapped_result.get('address', '')
            extracted_pincode = extract_pincode_from_text(address)
            if extracted_pincode:
                mapped_result['pincode'] = extracted_pincode
        
        logger.info(f"Successfully extracted Aadhaar details: {mapped_result}")
        return mapped_result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Raw response: {response.choices[0].message.content}")
        return {
            'name': '',
            'aadhaar_number': '',
            'dob': '',
            'gender': '',
            'address': '',
            'pincode': '',
            'father_name': '',
            'husband_name': ''
        }
    except Exception as e:
        logger.error(f"Error extracting Aadhaar details: {e}")
        return {
            'name': '',
            'aadhaar_number': '',
            'dob': '',
            'gender': '',
            'address': '',
            'pincode': '',
            'father_name': '',
            'husband_name': ''
        }

def extract_pan_details(image_path: str) -> dict:
    """
    Extract PAN card details using OpenAI GPT-4 Vision API
    
    Args:
        image_path: Path to the PAN card image
        
    Returns:
        Dictionary containing extracted PAN details
    """
    try:
        # Read and encode the image
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # Call GPT-4 Vision to extract info
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in extracting PAN card information. Extract all details accurately and return in JSON format."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the following details from the PAN card image and return as JSON: pan_card_number, person_name, date_of_birth, gender, father_name. The PAN card number should be in the format XXXXX1234X (10 characters). The person_name should be the full name of the card holder. The date of birth should be in DD/MM/YYYY format. Gender should be MALE or FEMALE. Father's name should be the full name as shown on the card. Return only valid JSON without any additional text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1000
        )

        # Parse and return the result
        content = response.choices[0].message.content.strip()
        
        # Remove markdown formatting if present
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        result = json.loads(content)
        
        # Map the extracted fields to the expected format
        mapped_result = {
            'pan_card_number': result.get('pan_card_number', ''),
            'person_name': result.get('person_name', ''),
            'date_of_birth': result.get('date_of_birth', ''),
            'gender': result.get('gender', ''),
            'father_name': result.get('father_name', '')
        }
        
        # Clean and validate PAN card number
        pan_number = mapped_result.get('pan_card_number', '')
        if pan_number:
            # Remove any spaces and convert to uppercase
            pan_number = pan_number.replace(' ', '').upper()
            # Validate PAN format (10 characters: 5 letters + 4 digits + 1 letter)
            if len(pan_number) == 10 and pan_number[:5].isalpha() and pan_number[5:9].isdigit() and pan_number[9].isalpha():
                mapped_result['pan_card_number'] = pan_number
            else:
                logger.warning(f"Invalid PAN card number format: {pan_number}")
                mapped_result['pan_card_number'] = pan_number  # Keep as is for now
        
        # Clean and validate date of birth
        dob = mapped_result.get('date_of_birth', '')
        if dob:
            # Try to standardize date format
            try:
                # If it's in DD/MM/YYYY format, convert to YYYY-MM-DD
                if '/' in dob:
                    day, month, year = dob.split('/')
                    if len(year) == 2:
                        year = '20' + year if int(year) < 50 else '19' + year
                    mapped_result['date_of_birth'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                else:
                    # If already in YYYY-MM-DD format, keep as is
                    mapped_result['date_of_birth'] = dob
            except Exception as e:
                logger.warning(f"Error processing date of birth: {e}")
                mapped_result['date_of_birth'] = dob  # Keep original format
        
        # Clean and validate gender
        gender = mapped_result.get('gender', '')
        if gender:
            gender = gender.upper().strip()
            if gender in ['MALE', 'M', 'BOY']:
                mapped_result['gender'] = 'MALE'
            elif gender in ['FEMALE', 'F', 'GIRL']:
                mapped_result['gender'] = 'FEMALE'
            else:
                mapped_result['gender'] = gender  # Keep as is
        
        logger.info(f"Successfully extracted PAN details: {mapped_result}")
        return mapped_result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Raw response: {response.choices[0].message.content}")
        return {
            'pan_card_number': '',
            'person_name': '',
            'date_of_birth': '',
            'gender': '',
            'father_name': ''
        }
    except Exception as e:
        logger.error(f"Error extracting PAN details: {e}")
        return {
            'pan_card_number': '',
            'person_name': '',
            'date_of_birth': '',
            'gender': '',
            'father_name': ''
        }
