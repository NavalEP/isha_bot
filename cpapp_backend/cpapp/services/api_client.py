import os
import requests
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CarepayAPIClient:
    """
    Client for interacting with the Carepay API endpoints
    """
    
    def __init__(self):
        self.base_url = os.getenv('CAREPAY_API_BASE_URL', 'https://backend.carepay.money')
        # Default doctor details that can be overridden with actual values from API
        self.doctor_id = os.getenv('DOCTOR_ID', None)
        self.doctor_name = os.getenv('DOCTOR_NAME', None)
        # Flag to track if we have actual doctor details
        self.has_doctor_details = False
        
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the API
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            headers: Request headers
            
        Returns:
            API response
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            logger.info(f"Making {method} request to {url}")
            
            # Log request details for debugging
            if params:
                logger.debug(f"Request params: {params}")
            if data:
                logger.debug(f"Request data: {data}")
            if headers:
                logger.debug(f"Request headers: {headers}")
            
            response = None
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, json=data, headers=headers, timeout=30)
            else:
                error_msg = f"Unsupported method: {method}"
                logger.error(error_msg)
                return {"status": 400, "error": error_msg}
            
            logger.info(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Log response body (truncated for readability)
            response_text = response.text
            if len(response_text) > 1000:
                logger.info(f"Response body (truncated): {response_text[:1000]}...")
            else:
                logger.info(f"Response body: {response_text}")
            
            # Handle HTTP error status codes
            if response.status_code >= 400:
                error_response = {
                    "status": response.status_code, 
                    "error": response_text,
                    "url": url,
                    "method": method
                }
                logger.error(f"HTTP error {response.status_code}: {response_text}")
                return error_response
            
            # Try to parse JSON response
            try:
                json_response = response.json()
                logger.debug(f"Successfully parsed JSON response")
                return json_response
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON response: {e}")
                logger.warning(f"Raw response: {response_text}")
                return {
                    "status": 200, 
                    "data": response_text,
                    "warning": "Response was not valid JSON"
                }
            
        except requests.exceptions.Timeout as e:
            error_msg = f"API request timeout after 30 seconds: {str(e)}"
            logger.error(error_msg)
            return {"status": 408, "error": error_msg, "url": url, "method": method}
        except requests.exceptions.ConnectionError as e:
            error_msg = f"API connection error: {str(e)}"
            logger.error(error_msg)
            return {"status": 503, "error": error_msg, "url": url, "method": method}
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            logger.error(error_msg)
            return {"status": 500, "error": error_msg, "url": url, "method": method}
        except Exception as e:
            error_msg = f"Unexpected error during API request: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": 500, "error": error_msg, "url": url, "method": method}
        
    def send_otp(self, phone_number: str) -> Dict[str, Any]:
        """
        Send OTP to phone number
        
        Args:
            phone_number: Phone number to send OTP to
            
        Returns:
            API response
        """
        print(f"Sending OTP to {phone_number} with base URL: {self.base_url}")
        endpoint = "sendOtp"
        response = self._make_request('GET', endpoint, params={"phoneNumber": phone_number})
        print(f"API response: {response}")
        return response
    
    def verify_otp(self, phone_number: str, otp: str) -> Dict[str, Any]:
        """
        Verify OTP for phone number
        
        Args:
            phone_number: Phone number to verify OTP for
            otp: OTP code to verify
            
        Returns:
            API response
        """
        print(f"Verifying OTP for {phone_number} with base URL: {self.base_url}")
        endpoint = "getOtp"
        response = self._make_request('GET', endpoint, params={
            "phoneNumber": phone_number,
            "otp": otp
        })
        print(f"API response: {response}")
        return response
    
    def get_doctor_details(self, phone_number: str) -> Dict[str, Any]:
        """
        Get doctor details by phone number
        
        Args:
            phone_number: Doctor's phone number
            
        Returns:
            API response with doctor details
        """
        endpoint = "getDoctorDetailsByPhoneNumber"
        response = self._make_request('GET', endpoint, params={"mobileNo": phone_number})
        
        # If we successfully retrieved doctor details, store the doctor_id and doctor_name
        if response.get("status") == 200 and response.get("data"):
            doctor_data = response.get("data")
            if doctor_data.get("doctorId"):
                self.doctor_id = doctor_data.get("doctorId")
                logger.info(f"Updated doctor_id to {self.doctor_id}")
            
            if doctor_data.get("name"):
                self.doctor_name = doctor_data.get("name")
                logger.info(f"Updated doctor_name to {self.doctor_name}")
            
            self.has_doctor_details = True
            
        return response
        
    def get_user_id_from_phone_number(self, phone_number: str) -> Dict[str, Any]:
        """Get user ID from phone number"""
        endpoint = f"userDetails/registerUsingMobileNo"
        headers = {'X-API-KEY': 'carepay'}
        return self._make_request('GET', endpoint, params={"mobileNo": phone_number}, headers=headers)
    
    def save_basic_details(self, user_id: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Save basic personal details"""
        endpoint = f"userDetails/basicDetail"
        data = {
            "aadhaarNo": details.get("aadhaarNo", None),
            "alternateNumber": details.get("alternateNumber", None),
            "dateOfBirth": details.get("dateOfBirth", None),
            "educationLevel": details.get("educationLevel", None),
            "emailId": details.get("emailId", None),
            "fatherName": details.get("fatherName", None),
            "firstName": details.get("firstName", None),
            "formStatus": details.get("formStatus", None),
            "gender": details.get("gender", None),
            "maritalStatus": details.get("maritalStatus", None),
            "mobileNumber": details.get("mobileNumber", None),
            "motherName": details.get("motherName", None),
            "panCard": details.get("panCard", None),
            "referenceName": details.get("referenceName", None),
            "referenceNumber": details.get("referenceNumber", None),
            "referenceRelation": details.get("referenceRelation", None),
            "typeOfEmail": details.get("typeOfEmail", None),
            "userId": user_id
        }
        return self._make_request('POST', endpoint, data=data)
    
    def get_prefill_data(self, user_id: str) -> Dict[str, Any]:
        """Get prefilled data from phone number"""
        endpoint = f"phoneToPrefill"
        headers = {'X-API-KEY': 'carepay'}
        return self._make_request('GET', endpoint, params={"userId": user_id}, headers=headers)
    
    def save_address_details(self, user_id: str, address: Dict[str, Any]) -> Dict[str, Any]:
        """Save address details"""
        endpoint = f"userDetails/addressDetail"
        data = {
            "address": address.get("address", None),
            "addressType": "current", 
            "city": address.get("city", ""),
            "formStatus": address.get("formStatus", ""),
            "pincode": address.get("pincode", None),
            "state": address.get("state", None),
            "userId": user_id
        }
        return self._make_request('POST', endpoint, data=data)
        
    
    def get_employment_verification(self, user_id: str) -> Dict[str, Any]:
        """Get employment verification data"""
        endpoint = f"getEmploymentVerificationSignzy"
        return self._make_request('GET', endpoint, params={"userId": user_id})
    
    def save_employment_details(self, user_id: str, employment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save employment details"""
        endpoint = f"userDetails/employmentDetail"
        data = {
            "employmentType": employment_data.get("employmentType", None),
            "formStatus": "",
            "monthlyFamilyIncome": employment_data.get("monthlyFamilyIncome", None),
            "nameOfBusiness": employment_data.get("nameOfBusiness", None), 
            "netTakeHomeSalary": employment_data.get("netTakeHomeSalary", None),
            "organizationName": employment_data.get("organizationName", None),
            "totalJobExpInMonth": employment_data.get("totalJobExpInMonth", None),
            "totalJobExpInYears": employment_data.get("totalJobExpInYears", None),
            "typeOfBusiness": employment_data.get("typeOfBusiness", None),
            "userId": user_id,
            "workplacePincode": employment_data.get("workplacePincode", None)
        }
        return self._make_request('POST', endpoint, data=data)
    
    def save_loan_details(self, user_id: str, name: str, loan_amount: int, doctor_name: str = None, doctor_id: str = None) -> Dict[str, Any]:
        """Save loan details"""
        endpoint = f"userDetails/saveLoanDetails"
        
        # Use provided doctor details if available, otherwise use instance variables
        doctor_id_to_use = doctor_id if doctor_id is not None else self.doctor_id
        doctor_name_to_use = doctor_name if doctor_name is not None else self.doctor_name
        
        # Log a warning if we're using default doctor details
        if not self.has_doctor_details and doctor_id is None and doctor_name is None:
            logger.warning("Using default doctor details. Call get_doctor_details first to use actual details.")
            
        data = {
            "doctorId": doctor_id_to_use,
            "doctorName": doctor_name_to_use,
            "formStatus": "",
            "loanAmount": loan_amount,
            "treatmentAmount": loan_amount,
            "loanReason": "",
            "Name": name,
            "userId": user_id
        }
        return self._make_request('POST', endpoint, data=data)
    
    def get_loan_details_by_user_id(self, user_id: str) -> Dict[str, Any]:
        """Get loan details by user ID"""
        endpoint = f"userDetails/getLoanDetailsByUserId"
        return self._make_request('GET', endpoint, params={"userId": user_id})
    
    def get_experian_bureau_report(self, loan_id: str) -> Dict[str, Any]:
        """Get Experian bureau report"""
        endpoint = f"experianBureauReport"
        return self._make_request('GET', endpoint, params={"loanId": loan_id})
    
    def get_bureau_decision(self, doctor_id: str, loan_id: str, regenerate_param: int = 1) -> Dict[str, Any]:
        """Get bureau-based decision"""
        endpoint = f"getBureauDecision"
        params = {
            "doctorId": doctor_id,
            "loanId": loan_id,
            "regenerateParam": regenerate_param
        }
        
        result = self._make_request('GET', endpoint, params=params)
        
        # If successful, try to extract eligible EMI information
        if result.get("status") == 200:
            try:
                # Look for max eligible EMI in the response
                data = result.get("data", {})
                
                # Extract maxEligibleEMI if present
                if "maxEligibleEMI" not in data and "eligibleEMI" in data:
                    result["data"]["maxEligibleEMI"] = data["eligibleEMI"]
                
                # If not directly available, try to extract from bureau checks
                if ("maxEligibleEMI" not in data or not data.get("maxEligibleEMI")) and "bureauChecks" in data:
                    bureau_checks = data["bureauChecks"]
                    
                    # Look for eligible EMI in bureau checks
                    for check in bureau_checks:
                        if isinstance(check, dict):
                            # Check for direct EMI information
                            if "eligibleEMI" in check:
                                result["data"]["maxEligibleEMI"] = check["eligibleEMI"]
                                break
                            
                            # Look in policy details
                            if "policyCheck" in check and "EMI" in check["policyCheck"] and "value" in check:
                                result["data"]["maxEligibleEMI"] = check["value"]
                                break
                
                # Add plans to the result if they exist
                if "emiPlans" in data and data["emiPlans"]:
                    # Ensure the plans are properly formatted with amount values as strings
                    formatted_plans = []
                    for plan in data["emiPlans"]:
                        formatted_plan = {}
                        for key, value in plan.items():
                            if key in ["creditLimit", "emi", "downPayment"] and value is not None:
                                formatted_plan[key] = str(value)
                            else:
                                formatted_plan[key] = value
                        formatted_plans.append(formatted_plan)
                    
                    result["data"]["emiPlans"] = formatted_plans
                    
                    # If we have plans but no max eligible EMI, use the highest EMI from plans
                    if ("maxEligibleEMI" not in data or not data.get("maxEligibleEMI")) and formatted_plans:
                        try:
                            highest_emi = max(
                                (float(plan["emi"]) for plan in formatted_plans if plan.get("emi") is not None),
                                default=None
                            )
                            if highest_emi:
                                result["data"]["maxEligibleEMI"] = str(int(highest_emi))
                        except (ValueError, TypeError):
                            logger.warning("Could not determine max eligible EMI from plans")
                
                # Make sure we have decision status
                if "bureauDecision" in data and "status" not in data:
                    result["data"]["status"] = data["bureauDecision"]
                    
                # Check for decision reason
                if "reason" not in data and "bureauChecks" in data:
                    for check in bureau_checks:
                        if isinstance(check, dict) and check.get("autoDecision") == "FAILED":
                            if "policyCheck" in check:
                                result["data"]["reason"] = f"Failed {check['policyCheck']} check"
                                break
                
                logger.info(f"Enhanced bureau decision with eligible EMI info: {result['data']}")
            except Exception as e:
                logger.warning(f"Error enhancing bureau decision: {e}")
        
        return result

    def get_profile_completion_link(self, doctor_id: str = None) -> Dict[str, Any]:
        """
        Get the profile completion link for a doctor
        
        Args:
            doctor_id: Doctor's ID (optional - will use session doctor_id if not provided)
            
        Returns:
            API response containing the profile completion link
        """
        # Use provided doctor_id or fall back to session doctor_id
        doctor_id_to_use = doctor_id if doctor_id is not None else self.doctor_id
        
        if not doctor_id_to_use:
            logger.warning("No doctor_id available for getting profile completion link")
            return {"status": 400, "error": "Doctor ID is required"}
            
        endpoint = "getProfileCompletionLink"
        logger.info(f"Getting profile completion link for doctor {doctor_id_to_use}")
        
        return self._make_request('GET', endpoint, params={"doctorId": doctor_id_to_use})
    
    def save_loan_details_again(self, user_id: str, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save loan details"""
        endpoint = f"userDetails/saveLoanDetails"
        
        # Use provided doctor details if available, otherwise use instance variables
        doctor_id_to_use = loan_data.get("doctorId")
        doctor_name_to_use = loan_data.get("doctorName")
        
        # Log a warning if we're using default doctor details
        if not self.has_doctor_details and doctor_id_to_use is None and doctor_name_to_use is None:
            logger.warning("Using default doctor details. Call get_doctor_details first to use actual details.")
            
        data = {
            "doctorId": doctor_id_to_use,
            "doctorName": doctor_name_to_use,
            "formStatus": "",
            "loanAmount": loan_data.get("treatmentCost"),
            "treatmentAmount": loan_data.get("treatmentCost"),
            "loanReason": loan_data.get("loanReason"),
            "Name": loan_data.get("fullName"),
            "userId": user_id
        }
        return self._make_request('POST', endpoint, data=data)
    
    def pan_verification(self, user_id: str) -> Dict[str, Any]:
        """
        Verify PAN details for a user
        
        Args:
            user_id: User ID to verify PAN for
            
        Returns:
            API response
        """
        endpoint = "panVerify"
        return self._make_request('GET', endpoint, params={"userId": user_id})
    
    def profile_ingestion_for_fibe(self, user_id: str) -> Dict[str, Any]:
        """
        Shares borrower data with the lender 'FIBE'.
        Corresponds to API: profileIngestionForFibe

        Args:
            user_id: The ID of the user.

        Returns:
            API response.
        """
        endpoint = "profileIngestionForFibe"
        params = {
            "userId": user_id,
            "type": "customer"
        }
        data = {}  # Empty body for POST request
        logger.info(f"Initiating profile ingestion for Fibe for userId: {user_id} with type: customer")
        return self._make_request('POST', endpoint, params=params, data=data)

    def check_fibe_flow(self, user_id: str) -> Dict[str, Any]:
        """
        Checks the decision provided by the lender 'Fibe'.
        Corresponds to API: checkFibeFlow

        Args:
            user_id: The ID of the user.

        Returns:
            API response.
        """
        endpoint = "checkFibeFlow"
        params = {"userId": user_id}
        logger.info(f"Checking Fibe flow for userId: {user_id}")
        return self._make_request('GET', endpoint, params=params)

    def check_doctor_mapped_by_nbfc(self, doctor_id: str) -> Dict[str, Any]:
        """
        Checks if a doctor is mapped by the NBFC 'FIBE'.
        Corresponds to API: checkDoctorMappedByNbfc

        Args:
            doctor_id: The ID of the doctor.

        Returns:
            API response.
        """
        endpoint = "checkDoctorMappedByNbfc"
        params = {
            "doctorId": doctor_id,
            "nbfc": "FIBE"  # NBFC is always FIBE as per new requirement
        }
        logger.info(f"Checking if doctor {doctor_id} is mapped by NBFC FIBE")
        return self._make_request('GET', endpoint, params=params)
    
    def check_eligibility_for_jp_cardless(self, loan_id: str) -> Dict[str, Any]:
        """
        Checks doctor-Juspay mapping.
        Corresponds to API: /jp/checkEligibilityForJPCardless

        Args:
            loan_id: The ID of the loan.

        Returns:
            API response.
        """
        endpoint = "jp/checkEligibilityForJPCardless"
        params = {"loanId": loan_id}
        logger.info(f"Checking eligibility for Juspay Cardless for loanId: {loan_id}")
        return self._make_request('GET', endpoint, params=params)

    def establish_eligibility(self, loan_id: str) -> Dict[str, Any]:
        """
        Checks for approval for Juspay.
        Corresponds to API: /jp/establishEligibility

        Args:
            loan_id: The ID of the loan.

        Returns:
            API response.
        """
        endpoint = "jp/establishEligibility"
        params = {"loanId": loan_id}
        logger.info(f"Establishing eligibility for Juspay for loanId: {loan_id}")
        return self._make_request('GET', endpoint, params=params)
    
    def state_and_city_by_pincode(self, pincode: str) -> Dict[str, Any]:
        """
        Get state and city details from pincode.
        Corresponds to API: /userDetails/codeDetail

        Args:
            pincode: The pincode to get state and city for.

        Returns:
            API response with state and city details.
        """
        endpoint = "userDetails/codeDetail"
        params = {
            "code": pincode,
            "type": "zip"
        }
        logger.info(f"Getting state and city for pincode: {pincode}")
        return self._make_request('GET', endpoint, params=params)
    
    def login_with_password(self, doctor_code: str, password: str) -> Dict[str, Any]:
        """
            Login with doctor code and password.
            Corresponds to API: /loginWithPassword

            Args:
                doctor_code: The doctor code to login with.
                password: The password to login with.

            Returns:
                API response with login details.
        """
        endpoint = "loginWithPassword"
        params = {
                    "doctorCode": doctor_code,
                    "password": password
        }
        logger.info(f"Logging in doctor with code: {doctor_code}")
        return self._make_request('GET', endpoint, params=params)
