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
            
            # Prepare request kwargs
            kwargs = {
                'timeout': self.timeout
            }
            
            if params:
                kwargs['params'] = params
            if data and method in ['POST', 'PUT', 'PATCH']:
                if files:
                    kwargs['data'] = data
                else:
                    kwargs['json'] = data
            if files:
                kwargs['files'] = files
                # Remove Content-Type header for multipart requests
                headers = self.session.headers.copy()
                headers.pop('Content-Type', None)
                kwargs['headers'] = headers
            
            logger.info(f"Making {method} request to {url}")
            logger.info(f"Request kwargs: {kwargs}")
            
            response = self.session.request(method, url, **kwargs)
            
            if response_type == 'blob':
                return response
            
            response.raise_for_status()
            return response.json()
            
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
        # Create files dict with the file object
        files = {'file': (file.name, file, file.content_type)}
        data = {
            'type': file_type,
            'userId': user_id,
            'fileName': file_name_param
        }
        
        logger.info(f"UploadDocuments: Sending file: {file.name}, size: {file.size}, content_type: {file.content_type}")
        logger.info(f"UploadDocuments: Data: {data}")
        
        return self._make_request("uploadDocuments", method='POST', data=data, files=files)
    
    def get_loan_transactions(self, doctor_id: str, clinic_name: str = '', 
                            start_date: str = '', end_date: str = '') -> Optional[Dict]:
        """Get loan transactions for doctor"""
        params = {
            'doctorId': doctor_id,
            'type': 'detail',
            'clinicName': clinic_name,
            'startDate': start_date,
            'endDate': end_date
        }
        return self._make_request("getAllLoanDetailForDoctorNew", params=params)
    
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
