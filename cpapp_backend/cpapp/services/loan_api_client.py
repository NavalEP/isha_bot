import requests
import logging
import json
from typing import Optional, Dict, Any, List
from django.conf import settings

logger = logging.getLogger(__name__)

class LoanAPIClient:
    """Client for making requests to the external loan API"""
    
    def __init__(self):
        self.base_url = "https://backend.carepay.money"
        self.timeout = 30
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'CarePay-Bot/1.0'
        })
    
    def _make_request(self, endpoint: str, method: str = 'GET', 
                     data: Optional[Dict] = None, params: Optional[Dict] = None, 
                     files: Optional[Dict] = None, response_type: str = 'json') -> Optional[Any]:
        """
        Make a request to the external API
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request data for POST/PUT requests
            params: Query parameters
            files: Files for multipart requests
            response_type: Expected response type ('json' or 'blob')
            
        Returns:
            Response data or None if request failed
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            logger.info(f"Constructed URL: {url}")
            
            # Prepare request kwargs
            kwargs = {
                'timeout': self.timeout
            }
            
            if params:
                kwargs['params'] = params
            if data and method in ['POST', 'PUT', 'PATCH']:
                if files:
                    # For multipart requests, always send data as form data
                    kwargs['data'] = data
                else:
                    kwargs['json'] = data
            if files:
                kwargs['files'] = files
                # For multipart requests, create headers without Content-Type
                # Let requests automatically set the correct multipart Content-Type with boundary
                headers = {k: v for k, v in self.session.headers.items() if k.lower() != 'content-type'}
                kwargs['headers'] = headers
            
            logger.info(f"Making {method} request to {url}")
            logger.info(f"Request kwargs: {kwargs}")
            if files:
                logger.info(f"Files being sent: {list(files.keys())}")
                for key, file_tuple in files.items():
                    logger.info(f"File {key}: name={file_tuple[0]}, content_type={file_tuple[2] if len(file_tuple) > 2 else 'unknown'}")
            
            response = self.session.request(method, url, **kwargs)
            
            if response_type == 'blob':
                return response
            
            # Log response details for debugging
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            try:
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP Error: {e}")
                logger.error(f"Response content: {response.text}")
                # Try to parse error response
                try:
                    error_data = response.json()
                    logger.error(f"Error response JSON: {error_data}")
                    return error_data
                except:
                    logger.error(f"Could not parse error response as JSON")
                    return None
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {endpoint}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for {endpoint}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {str(e)}")
            return None
    
    def get_qr_code(self, doctor_id: str) -> Optional[Dict]:
        """Get QR code for doctor"""
        return self._make_request(f"getQrCode?doctorId={doctor_id}")
    
    def get_activities_log(self, user_id: str) -> Optional[Dict]:
        """Get activities log for user"""
        return self._make_request(f"activitiesLog?userId={user_id}")
    
    def get_assigned_product(self, user_id: str) -> Optional[Dict]:
        """Get assigned product for user"""
        return self._make_request(f"userDetails/getAssignedProductByUserId?userId={user_id}")
    
    def get_bureau_decision(self, loan_id: str) -> Optional[Dict]:
        """Get bureau decision for loan"""
        params = {
            'loanId': loan_id,
            'test': '0',
            'triggeredBy': 'admin'
        }
        return self._make_request("bureauDecisionNew", params=params)
    
    def get_disburse_detail_report(self, user_id: str) -> Optional[requests.Response]:
        """Get disbursal report for user"""
        return self._make_request(
            f"getDisburseDetailForReport?userId={user_id}",
            response_type='blob'
        )
    
    def upload_documents(self, file, user_id: str, file_type: str = 'img', 
                        file_name_param: str = 'treatmentProof') -> Optional[Dict]:
        """Upload documents"""
        # Ensure file has required attributes
        file_name = getattr(file, 'name', 'uploaded_file')
        content_type = getattr(file, 'content_type', 'application/octet-stream')
        
        # Ensure file is at the beginning
        if hasattr(file, 'seek'):
            file.seek(0)
        
        # Try base64 encoding approach first
        try:
            import base64
            file_content = file.read()
            base64_content = base64.b64encode(file_content).decode('utf-8')
            
            # Reset file pointer after reading
            file.seek(0)
            
            # Send as JSON with base64 encoded file
            data = {
                'type': file_type,
                'userId': user_id,
                'fileName': file_name_param,
                'document': base64_content,
                'originalFileName': file_name,
                'contentType': content_type
            }
            
            logger.info(f"UploadDocuments: Sending file as base64: {file_name}, size: {len(file_content)} bytes, content_type: {content_type}")
            logger.info(f"UploadDocuments: Data keys: {list(data.keys())}")
            
            return self._make_request("uploadDocuments/", method='POST', data=data, files=None)
            
        except Exception as e:
            logger.error(f"Error encoding file as base64: {e}")
            # Fallback to multipart method
            files = {'document': (file_name, file, content_type)}
            data = {
                'type': file_type,
                'userId': user_id,
                'fileName': file_name_param
            }
            
            logger.info(f"UploadDocuments: Fallback to multipart - Sending file: {file_name}, content_type: {content_type}")
            return self._make_request("uploadDocuments/", method='POST', data=data, files=files)
    
    def get_loan_transactions(self, doctor_id: str, parent_doctor_id: str = '', clinic_name: str = '', 
                            start_date: str = '', end_date: str = '', loan_status: str = '') -> Optional[Dict]:
        """Get loan transactions for doctor"""
        params = {
            'doctorId': doctor_id,
            'type': 'detail',
            'clinicName': clinic_name,
            'startDate': start_date,
            'endDate': end_date,
            'loanStatus': loan_status
        }
        
        # Add parentDoctorId if provided
        if parent_doctor_id:
            params['parentDoctorId'] = parent_doctor_id
            
        return self._make_request("getAllLoanDetailForDoctorNew", params=params)
    
    def get_loan_count_and_amount_for_doctor(self, doctor_id: str, clinic_name: str = '') -> Optional[Dict]:
        """Get loan count and amount statistics for doctor"""
        params = {
            'doctorId': doctor_id,
            'clinicName': clinic_name
        }
        return self._make_request("getLoanCountAndAmountForDoctor", params=params)
    
    def get_user_loan_status(self, loan_id: str) -> Optional[Dict]:
        """Get user loan status"""
        params = {
            'loanId': loan_id
        }
        return self._make_request("status/getUserLoanStatus", params=params)
    
    def get_all_child_clinics(self, doctor_id: str) -> Optional[Dict]:
        """Get all child clinics for a doctor"""
        params = {
            'doctorId': doctor_id
        }
        return self._make_request("getAllChildClinic", params=params)
    
    def get_matching_emi_plans(self, user_id: str, loan_id: str) -> Dict[str, Any]:
        """
        Get matching EMI plans by combining assigned product and bureau decision APIs
        
        Returns:
            Dict with keys: plans, hasMatchingProduct, isApproved, assignedProductFailed
        """
        try:
            # Step 1: Get assigned product
            assigned_product_result = self.get_assigned_product(user_id)
            assigned_product = assigned_product_result.get('data') if assigned_product_result and assigned_product_result.get('status') == 200 else None
            assigned_product_failed = not assigned_product
            
            # Step 2: Get bureau decision
            bureau_result = self.get_bureau_decision(loan_id)
            
            if not bureau_result or bureau_result.get('status') != 200:
                return {
                    'plans': [],
                    'hasMatchingProduct': False,
                    'isApproved': False,
                    'assignedProductFailed': assigned_product_failed
                }
            
            decision_data = bureau_result.get('data', {}).get('data', {})
            emi_plans = decision_data.get('emiPlanList', [])
            is_approved = decision_data.get('finalDecision', '').lower() == 'approved'
            
            # Step 3: Process results
            if is_approved and emi_plans:
                if assigned_product:
                    # Filter matching plans
                    matching_plans = [
                        plan for plan in emi_plans 
                        if plan.get('productDetailsDO', {}).get('productId') == assigned_product.get('productId')
                    ]
                    
                    if matching_plans:
                        return {
                            'plans': matching_plans,
                            'hasMatchingProduct': True,
                            'isApproved': True,
                            'assignedProductFailed': False
                        }
                    else:
                        return {
                            'plans': [],
                            'hasMatchingProduct': False,
                            'isApproved': True,
                            'assignedProductFailed': False
                        }
                else:
                    # Show all plans when assigned product API fails
                    return {
                        'plans': emi_plans,
                        'hasMatchingProduct': True,
                        'isApproved': True,
                        'assignedProductFailed': True
                    }
            
            # Not approved or no plans
            return {
                'plans': [],
                'hasMatchingProduct': False,
                'isApproved': False,
                'assignedProductFailed': assigned_product_failed
            }
            
        except Exception as e:
            logger.error(f"Error getting matching EMI plans: {str(e)}")
            return {
                'plans': [],
                'hasMatchingProduct': False,
                'isApproved': False,
                'assignedProductFailed': True
            }
