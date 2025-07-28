import openai
import base64
import json
import cv2
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

def extract_photo_from_aadhaar(image_path):
    """Extract the photo portion from Aadhaar card image"""
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image from {image_path}")
            return None
        
        # Convert to RGB for better processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = img_rgb.shape[:2]
        
        # Try multiple photo regions - Aadhaar cards can have photos in different positions
        photo_regions = [
            # Top-right region (most common)
            {
                'x': int(width * 0.65),
                'y': int(height * 0.15),
                'width': int(width * 0.25),
                'height': int(height * 0.35)
            },
            # Top-left region (some cards)
            {
                'x': int(width * 0.05),
                'y': int(height * 0.15),
                'width': int(width * 0.25),
                'height': int(height * 0.35)
            },
            # Center-right region
            {
                'x': int(width * 0.60),
                'y': int(height * 0.25),
                'width': int(width * 0.30),
                'height': int(height * 0.40)
            }
        ]
        
        best_photo = None
        best_score = 0
        
        for i, region in enumerate(photo_regions):
            # Extract the photo region
            photo_region = img_rgb[region['y']:region['y']+region['height'], 
                                  region['x']:region['x']+region['width']]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(photo_region, cv2.COLOR_RGB2GRAY)
            
            # Calculate variance (higher variance usually means more detailed content like a face)
            variance = np.var(gray)
            
            # Check if this region looks more like a photo (higher variance) than text
            if variance > best_score and variance > 1000:  # Threshold to avoid text regions
                best_score = variance
                best_photo = photo_region
                print(f"Region {i+1} selected with variance: {variance:.2f}")
        
        if best_photo is not None:
            # Save the extracted photo
            photo_path = "extracted_photo.jpg"
            cv2.imwrite(photo_path, cv2.cvtColor(best_photo, cv2.COLOR_RGB2BGR))
            print(f"Photo extracted and saved as: {photo_path}")
            return photo_path
        else:
            print("No suitable photo region found")
            return None
        
    except Exception as e:
        print(f"Error extracting photo: {e}")
        return None

def extract_photo_using_ai(image_path):
    """Use AI to identify and extract the photo region more accurately"""
    try:
        # Read and encode the image
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        
        # Ask AI to identify the photo region
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing ID documents. Identify the exact coordinates of the person's photo in the Aadhaar card."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this Aadhaar card image and provide the exact pixel coordinates (x, y, width, height) of the person's photo region. Return only the coordinates as JSON: {\"x\": number, \"y\": number, \"width\": number, \"height\": number}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=200
        )
        
        # Parse the coordinates
        coords_text = response.choices[0].message.content
        coords = json.loads(coords_text)
        
        # Extract the photo using the coordinates
        img = cv2.imread(image_path)
        photo_region = img[coords["y"]:coords["y"]+coords["height"], coords["x"]:coords["x"]+coords["width"]]
        
        # Save the extracted photo
        photo_path = "ai_extracted_photo.jpg"
        cv2.imwrite(photo_path, photo_region)
        
        print(f"AI-extracted photo saved as: {photo_path}")
        return photo_path
        
    except Exception as e:
        print(f"Error in AI photo extraction: {e}")
        return None

def extract_face_using_ai(image_path):
    """Use AI to specifically identify and extract the person's face"""
    try:
        # Read and encode the image
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        
        # Ask AI to identify the face region
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing ID documents and identifying human faces."},
                {"role": "user", "content": [
                    {"type": "text", "text": "In this Aadhaar card image, identify the exact pixel coordinates of the person's face/photo. Look for a human face, not text or other elements. Return only the coordinates as JSON: {\"x\": number, \"y\": number, \"width\": number, \"height\": number}. If no clear face is visible, return null."},
                    {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=200
        )
        
        # Parse the coordinates
        coords_text = response.choices[0].message.content.strip()
        
        # Handle null response
        if coords_text.lower() == "null" or "null" in coords_text:
            print("AI could not identify a clear face in the image")
            return None
            
        coords = json.loads(coords_text)
        
        # Extract the face using the coordinates
        img = cv2.imread(image_path)
        face_region = img[coords["y"]:coords["y"]+coords["height"], coords["x"]:coords["x"]+coords["width"]]
        
        # Save the extracted face
        face_path = "ai_extracted_face.jpg"
        cv2.imwrite(face_path, face_region)
        
        print(f"AI-extracted face saved as: {face_path}")
        return face_path
        
    except Exception as e:
        print(f"Error in AI face extraction: {e}")
        return None
