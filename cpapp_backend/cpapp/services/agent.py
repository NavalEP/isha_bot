import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import re  # for phone number detection and OTP regex
import tempfile
import random

from langchain_community.utilities import BingSearchAPIWrapper
from langchain.agents import Tool, AgentExecutor
from langchain.tools import StructuredTool
from langchain.agents import create_openai_functions_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import ArgsSchema
from cpapp.services.api_client import CarepayAPIClient
from cpapp.models.session_data import SessionData
from cpapp.services.session_manager import SessionManager
from setup_env import setup_environment
from cpapp.services.helper import Helper
from cpapp.services.url_shortener import shorten_url


setup_environment()

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

logger_session = logging.getLogger('session_management')
logger_session.addHandler(logging.StreamHandler())
logger_session.setLevel(logging.DEBUG)

class CarepayAgent:
    """
    Carepay AI Agent using LangChain for managing loan application processes
    """
    
    def __init__(self):
        """Initialize the CarePay agent with LLM and tools"""
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            temperature=0.2,
        )
        
        # Initialize API client
        self.api_client = CarepayAPIClient()
        
        # Define system prompt
        self.system_prompt = """
        You are a healthcare loan application assistant for CarePay. Your role is to help users apply for loans for medical treatments in a professional and friendly manner.

        CRITICAL MESSAGE FORMATTING RULES:
        1. NEVER modify, truncate, or duplicate any formatted messages
        2. Keep ALL markdown formatting (###, **, etc.) exactly as provided
        3. Keep ALL line breaks and spacing exactly as shown
        4. Keep ALL sections including employment type selection
        5. Do NOT add any additional text or instructions
        6. When a tool returns a formatted response, use it EXACTLY as provided - do not reformat or modify
        7. NEVER concatenate or merge multiple formatted responses
        8. If you receive a complete formatted message from a tool, return it verbatim without any changes
        9. Stick with the exact formatting provided by tools - do not try to improve or modify the format
        10. Preserve all punctuation, spacing, and structure exactly as received

        CRITICAL EXECUTION RULE: You MUST execute ALL steps in sequence. Do NOT stop until you reach step 10. If any step succeeds, immediately proceed to the next step.

        Follow these steps sequentially to process a loan application: and don't miss any step and any tool calling

        1. Initial Data Collection:
           - Greet the user warmly and introduce yourself as CarePay's healthcare loan assistant
           - Collect and validate these four essential pieces of information:
              * Patient's full name
              * Patient's phoneNumber (must be a 10 digit number; if not, return an error message and ask the user to enter a valid phone number)
              * treatmentCost (minimum ₹3,000) must be a positive number; if not, return an error message and ask the user to enter a valid treatment cost)
              * monthlyIncome (must be a positive number; if not, return an error message and ask the user to enter a valid monthly income)
              If any of these four details are missing from the user's message, ask for the remaining ones.
           - CRITICAL: If the treatmentCost is less than ₹3,000, IMMEDIATELY STOP the process and return this message (do not proceed to any further steps):

             "I understand your treatment cost is below ₹3,000. Currently, I can only process loan applications for treatments costing ₹3,000 or more. Please let me know if your treatment cost is ₹3,000 or above, and I'll be happy to help you with the loan application process."

           - Use the store_user_data tool and save this information in the session with the four parameters: fullName, phoneNumber, treatmentCost, monthlyIncome
           - IMMEDIATELY proceed to step 2 after completion (unless the treatmentCost is below ₹3,000, in which case the process must stop as above)

        2. User ID Creation:
           - didn't miss this step
           - Use get_user_id_from_phone_number tool to get userId from the phone number
           - if get_user_id_from_phone_number will show status 500 then return message like ask to enter valid phone number
           - Extract the userId from the response and store it in the session
           - Use this userId for all subsequent API calls
           - IMMEDIATELY proceed to step 3 after completion

        3. Basic Details Submission:
           - didn't miss this step
           - Retrieve name and phone number from session data
           - IMPORTANT: When calling save_basic_details, call this tool using session_id.
           - IMMEDIATELY proceed to step 4 after completion

        4. Save Loan Details:
           - didn't miss this step
           - Use save_loan_details tool to submit:
              * User's fullName (from initial data collection)
              * treatmentCost (from initial data collection)
              * userId (from initial data collection)
           - IMMEDIATELY proceed to step 5 after completion

        5. Check for Cardless Loan:
           - Call check_jp_cardless tool with session_id as input
           - If response status is "ELIGIBLE":
             * Show the approval message from the response
             * Skip remaining steps and end the process
           - If response status is "NOT_ELIGIBLE" or "API_ERROR":
             * Continue with remaining steps
           - IMMEDIATELY proceed to step 6 after completion

        6. Data Prefill:
           - didn't miss this step
           - if get_prefill_data will show status 200 then continue complete below steps
           - Use get_prefill_data tool by calling it with session_id 
           - Use the process_prefill_data_for_basic_details tool by calling it with session_id 
           - If get_prefill_data returns status 500 or 404:
             * Call get_profile_link tool with session_id to obtain the profile link
             * Respond: "Thank you [patient name]. I've collected your details. You can continue your loan application through this link: [profile link]"
           - IMMEDIATELY proceed to step 6 after completion

        7. Address Processing:
           - didn't miss this step
           - After processing the prefill data, use the process_address_data tool by session_id  
           - Use pan_verification tool using session_id 
           - CRITICAL: If pan_verification returns status 500 or error, STOP the process and inform the user about the PAN verification failure
           - Only continue to step 8 if pan_verification returns status 200
           - IMMEDIATELY proceed to step 8 after completion

        8. Employment Verification:
           - didn't miss this step
           - Use get_employment_verification tool to check employment status using session_id
           - IMMEDIATELY proceed to step 8 after completion

        9. Save Employment Details:
           - didn't miss this step
           - Use save_employment_details tool using session_id
           - IMMEDIATELY proceed to step 9 after completion

        10. Process Loan Application:
           - didn't miss this step
           - Use get_bureau_decision tool to get final loan decision using session_id
           - IMMEDIATELY proceed to step 10 after completion

        CRITICAL: Execute these steps in sequence, but STOP when you reach step 10 (get_bureau_decision) and provide the formatted response. Do NOT restart the process.

        CRITICAL: NEVER deviate from these exact steps and templates. Do not add, modify, or skip any steps.

        CRITICAL BUREAU DECISION HANDLING:
        - When get_bureau_decision tool returns a formatted response, use it EXACTLY as provided
        - Do NOT add any additional formatting or text to the bureau decision response
        - Do NOT duplicate the employment type question or any other part of the response
        - Return the complete formatted response from the tool without any modifications
        - If the tool response includes "What is the Employment Type of the patient?", keep it exactly as shown
        """
        
    
    def create_session(self, doctor_id=None, doctor_name=None, phone_number=None) -> str:
        """
        Create a new chat session
        
        Args:
            doctor_id: Optional doctor ID from URL parameters
            doctor_name: Optional doctor name from URL parameters
            phone_number: Optional phone number from URL parameters
        Returns:
            Session ID
        """
        try:
            # Generate a unique session ID
            session_id = str(uuid.uuid4())

            # Create initial greeting message
            initial_message = (
                "Hello! I'm here to assist you with your patient's medical loan. "
                "Let's get started. First, kindly provide me with the following details?\n"
                "1. Patient's full name\n"
                "2. Patient's phone number (linked to their PAN)\n"
                "3. The cost of the treatment\n"
                "4. Monthly income of your patient.\n\n"
                "**Example input format: name: John Doe phone number: 1234567890 treatment cost: 10000 monthly income: 50000**"
            )
            
            # Create session with initial data using single history approach
            session = {
                "id": session_id,
                "created_at": datetime.now().isoformat(),
                "status": "active",  
                "history": [{
                    "type": "AIMessage",
                    "content": initial_message
                }],  # Use serializable format directly
                "data": {},  
                "phone_number": phone_number
            }
            
            # Store doctor information if provided
            if doctor_id:
                session["data"]["doctor_id"] = doctor_id
                logger.info(f"Stored doctor_id {doctor_id} in session {session_id}")
            
            if doctor_name:
                session["data"]["doctor_name"] = doctor_name
                logger.info(f"Stored doctor_name {doctor_name} in session {session_id}")
            
            # Save to database
            SessionManager.update_session_in_db(session_id, session)
            
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    def run(self, session_id: str, message: str) -> str:
        """
        Process a user message within a session
        
        Args:
            session_id: Session identifier
            message: User message
            
        Returns:
            Agent response
        """
        try:
            # Get session from database once
            session = SessionManager.get_session_from_db(session_id)
            
            current_status = session.get("status", "active")
          
            # Special handling for collecting additional details state
            if current_status == "collecting_additional_details":
                logger.info(f"Session {session_id}: Entering additional details collection mode")
                # Use direct sequential flow instead of full agent for efficiency
                ai_message = self._handle_additional_details_collection(session_id, message)
                
                # Update history efficiently
                self._update_session_history(session_id, message, ai_message)
                return ai_message
            
            logger.info(f"Session {session_id}: Using full agent executor (status: {current_status})")
            # Create session-specific agent executor with session-aware tools
            session_tools = self._create_session_aware_tools(session_id)
            
            # Create the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            # Create session-specific agent
            agent = create_openai_functions_agent(self.llm, session_tools, prompt)
            session_agent_executor = AgentExecutor(
                agent=agent,
                tools=session_tools,
                verbose=True,
                max_iterations=50,
                handle_parsing_errors=True,
            )
            
            # Get chat history for LangChain (convert from serializable format)
            chat_history = self._convert_to_langchain_messages(session.get("history", []))
            chat_history.append(HumanMessage(content=message))
            
            # Use session-specific agent executor
            response = session_agent_executor.invoke({
                "input": message, 
                "chat_history": chat_history
            })
            
            # Extract AI message from response
            ai_message = response.get("output", "I'm processing your request. Please wait.")
            
            # Validate that all required steps were executed
            self._validate_and_handle_early_chain_finish(session_id, response, ai_message)
            
            # Check if this is the loan application decision with approval or income verification required
            # Add status update for additional details collection when employment type is being asked
            # Check for the specific employment type prompt and update status accordingly
            employment_type_prompt = (
                "What is the Employment Type of the patient?" in ai_message
                and "1. SALARIED" in ai_message
                and "2. SELF_EMPLOYED" in ai_message
                and "Please Enter input 1 or 2 only" in ai_message
            )
            if employment_type_prompt:
                SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                SessionManager.update_session_data_field(session_id, "data.collection_step", "employment_type")
                
                # Initialize additional_details if it doesn't exist
                if session.get("data", {}).get("additional_details") is None:
                    SessionManager.update_session_data_field(session_id, "data.additional_details", {})
                
                logger.info(f"Session {session_id} marked as collecting_additional_details")
                
                # Update history efficiently
                self._update_session_history(session_id, message, ai_message)
                return ai_message
            
            # Update conversation history efficiently (only if we haven't already updated it for bureau decision)
            if current_status != "collecting_additional_details" or "what is the employment type of the patient?" not in ai_message.lower():
                self._update_session_history(session_id, message, ai_message)
            
            return ai_message
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "Please start a new chat session to continue our conversation."

    def _convert_to_langchain_messages(self, history: List[Dict[str, Any]]) -> List:
        """
        Convert serializable history to LangChain message objects
        
        Args:
            history: List of serializable message dictionaries
            
        Returns:
            List of LangChain message objects
        """
        langchain_messages = []
        for msg in history:
            if isinstance(msg, dict):
                msg_type = msg.get('type', 'HumanMessage')
                content = msg.get('content', '')
                if msg_type == 'AIMessage':
                    langchain_messages.append(AIMessage(content=content))
                else:
                    langchain_messages.append(HumanMessage(content=content))
            elif hasattr(msg, 'content'):  # Already a LangChain message object
                langchain_messages.append(msg)
            else:
                # Fallback for any other format
                langchain_messages.append(HumanMessage(content=str(msg)))
        return langchain_messages

    def _update_session_history(self, session_id: str, user_message: str, ai_message: str) -> None:
        """
        Efficiently update session history with new messages
        
        Args:
            session_id: Session identifier
            user_message: User's message
            ai_message: AI's response
        """
        try:
            # Get current history
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return
            
            current_history = session.get("history", [])
            
            # Add new messages to history
            current_history.append({
                "type": "HumanMessage",
                "content": user_message
            })
            current_history.append({
                "type": "AIMessage",
                "content": ai_message
            })
            
            # Update history in database (single operation)
            SessionManager.update_session_data_field(session_id, "history", current_history)
            
        except Exception as e:
            logger.error(f"Error updating session history: {e}")

    def get_session_data(self, session_id: str = None) -> str:
        """
        Get session data for the current session
        
        Args:
            session_id: Session ID (required)
            
        Returns:
            Session data as JSON string with comprehensive data view
        """
        if not session_id:
            return "Session ID not found"
        
        session = SessionManager.get_session_from_db(session_id)
        if not session:
            return "Session ID not found"
        
        # Create a comprehensive view of session data
        comprehensive_session = {
            "session_info": {
                "id": session.get("id"),
                "status": session.get("status"),
                "created_at": session.get("created_at"),
                "phone_number": session.get("phone_number")
            },
            "user_data": session.get("data", {}),
            "conversation_history": []
        }
        
        # History is already in serializable format
        if "history" in session:
            comprehensive_session["conversation_history"] = session["history"]
        
        # Add summary of stored data
        data = session.get("data", {})
        data_summary = {
            "core_user_data": {
                "userId": data.get("userId"),
                "fullName": data.get("fullName") or data.get("name"),
                "phoneNumber": data.get("phoneNumber") or data.get("phone"),
                "treatmentCost": data.get("treatmentCost"),
                "monthlyIncome": data.get("monthlyIncome"),
            },
            "api_responses_count": len(data.get("api_responses", {})),
            "api_requests_count": len(data.get("api_requests", {})),
            "prefill_data_available": bool(data.get("prefill_data")),
            "employment_data_available": bool(data.get("employment_data")),
            "bureau_decision_available": bool(data.get("bureau_decision_details")),
            "additional_details_available": bool(data.get("additional_details")),
        }
        
        comprehensive_session["data_summary"] = data_summary
        
        return json.dumps(comprehensive_session, indent=2)
    
    # Tool implementations
    
    def store_user_data(self, input_str: str, session_id: str) -> str:
        """
        Store user data in the session
        
        Args:
            input_str: JSON string with data to store
            
        Returns:
            Confirmation message
        """
        try:
            data = json.loads(input_str)
            
            if not session_id:
                return "Session ID not found or invalid"
            
            # Normalize field names for consistency
            # Convert "phone" to "phoneNumber" for consistency
            if "phone" in data and "phoneNumber" not in data:
                data["phoneNumber"] = data.pop("phone")
            
            # Convert "name" to "fullName" for consistency 
            if "name" in data and "fullName" not in data:
                data["fullName"] = data.pop("name")
            
            # Validate treatment cost - minimum requirement is ₹3,000
            treatment_cost = data.get("treatmentCost")
            if treatment_cost is not None:
                try:
                    # Convert to float, handling various formats (₹, commas, etc.)
                    cost_str = str(treatment_cost).replace('₹', '').replace(',', '').strip()
                    cost_value = float(cost_str)
                    
                    if cost_value < 3000:
                        return f"I understand your treatment cost is ₹{cost_value:,.0f}. Currently, I can only process loan applications for treatments costing ₹18,000 or more. Please let me know if your treatment cost is ₹18,000 or above, and I'll be happy to help you with the loan application process."
                except (ValueError, TypeError):
                    # If we can't parse the cost, continue with normal flow
                    logger.warning(f"Could not parse treatment cost: {treatment_cost}")
            
            # Check if user_id is present in the data
            if 'user_id' in data or 'userId' in data:
                user_id = data.get('user_id') or data.get('userId')
                # Store userId using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.userId", user_id)
            
            # Store each piece of user data systematically
            for key, value in data.items():
                if key not in ['user_id']:  # Skip user_id as we handle it above as userId
                    SessionManager.update_session_data_field(session_id, f"data.{key}", value)
            
            # Also store the raw input for reference
            SessionManager.update_session_data_field(session_id, "data.user_input.store_user_data", data)
            
            logger.info(f"User data stored systematically in session {session_id}: {data}")
            
            return f"Data successfully stored in session {session_id}"
        except Exception as e:
            logger.error(f"Error storing user data: {e}")
            return f"Error storing data: {str(e)}"
        
    def get_user_id_from_phone_number(self, phone_number: str, session_id: str) -> str:
        """
        Get user ID from phone number
        
        Args:
            phone_number: User's phone number
            session_id: Session identifier
            
        Returns:
            API response as JSON string with userId
        """
        try:
            result = self.api_client.get_user_id_from_phone_number(phone_number)
            logger.info(f"API response from get_user_id_from_phone_number: {result}")
            
            # Store the complete API response in session data
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.api_responses.get_user_id_from_phone_number", result)
            
            # If successful, extract userId and store in session
            if result.get("status") == 200:
                # Parse the data field if it's a JSON string
                data = result.get("data")
                user_id_from_api = None
                
                if isinstance(data, str):
                    # First, try to parse as JSON (for the second response format with userId and prefill_data)
                    try:
                        parsed_data = json.loads(data)
                        user_id_from_api = parsed_data.get("userId")
                        logger.info(f"Successfully parsed JSON data and extracted clean userId: {user_id_from_api}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse 'data' field as JSON: {e}")
                        # Try to extract userId using regex as fallback for incomplete JSON
                        userId_match = re.search(r'"userId"\s*:\s*"([^"]+)"', data)
                        if userId_match:
                            user_id_from_api = userId_match.group(1)
                            logger.info(f"Extracted userId using regex fallback: {user_id_from_api}")
                        else:
                            # If regex also fails, treat it as a direct userId string (first response format)
                            if data and data.strip():
                                user_id_from_api = data.strip()
                                logger.info(f"Treating data as direct clean userId string: {user_id_from_api}")
                            else:
                                user_id_from_api = None
                elif isinstance(data, dict):
                    user_id_from_api = data.get("userId")
                    logger.info(f"Extracted userId from dict data: {user_id_from_api}")
                else:
                    user_id_from_api = None
                    logger.warning(f"Unexpected data type: {type(data).__name__}")
                
                # Ensure extracted_user_id is a non-empty string and validate it's a clean userId
                if isinstance(user_id_from_api, str) and user_id_from_api:
                    # Additional validation to ensure we never save a JSON string as userId
                    if user_id_from_api.startswith('{') or user_id_from_api.startswith('"'):
                        logger.error(f"Attempted to save JSON string as userId: {user_id_from_api}")
                        user_id_from_api = None
                    else:
                        if session_id:
                            # Store clean userId in session.data.userId
                            SessionManager.update_session_data_field(session_id, "data.userId", user_id_from_api)
                            logger.info(f"Stored clean userId '{user_id_from_api}' in session data for session {session_id}")
                
                if not user_id_from_api:
                    logger.warning(
                        f"UserId not found or is not a valid string in API response's 'data' field. "
                        f"Received: '{data}' (type: {type(data).__name__}) "
                        f"for session {session_id}."
                    )
            
            return user_id_from_api if user_id_from_api else "userId not found in API response"
        except Exception as e:
            logger.error(f"Error getting user ID from phone number: {e}")
            return f"Error getting user ID from phone number: {str(e)}"
        
    def get_prefill_data(self, user_id: str = None, session_id: str = None) -> str:
        """
        Get prefilled user data
        
        Args:
            user_id: User identifier, optional if available in session
            session_id: Session identifier
            
        Returns:
            Prefilled data as JSON string
        """
        try:
            if not session_id:
                return "Session ID is required"
            
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return "Session not found"
            
            user_id = session.get("data", {}).get("userId")
            
            if not user_id:
                return "User ID is required to get prefill data"
                
            result = self.api_client.get_prefill_data(user_id)
            # Store the complete API response in session data
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.api_responses.get_prefill_data", result)
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error getting prefill data: {e}")
            return f"Error getting prefill data: {str(e)}"
        
            

    def get_employment_verification(self, session_id: str) -> str:
        """
        Get employment verification data
        
        Args:
            session_id: Session identifier
            
        Returns:
            Employment verification data as JSON string
        """
        try:

            # If user_id is not provided, try to get from session
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                if session and session.get("data", {}).get("userId"):
                    user_id = session["data"]["userId"]
                    
                
            result = self.api_client.get_employment_verification(user_id)
            
            # Store the complete API response in session data
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.api_responses.get_employment_verification", result)
            
            # If successful, store important employment data in session
            if result.get("status") == 200 and session_id:
                try:
                    data = result.get("data", {})
                    employment_data = {}
                    
                    # Determine employment type
                    if "employmentSummary" in data:
                        summary = data["employmentSummary"]
                        if summary.get("is_employed", False):
                            employment_data["employmentType"] = "SALARIED"
                    
                    # Extract organization name if present
                    if "employmentSummary" in data and "recent_employer_data" in data["employmentSummary"]:
                        employer_data = data["employmentSummary"]["recent_employer_data"]
                        if "establishment_name" in employer_data:
                            employment_data["organizationName"] = employer_data["establishment_name"]
                    
                    # Store in session using update_session_data_field
                    if employment_data:
                        SessionManager.update_session_data_field(session_id, "data.employment_data", employment_data)
                        logger.info(f"Stored employment data in session: {employment_data}")
                except Exception as e:
                    logger.warning(f"Error processing employment data: {e}")
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error getting employment verification: {e}")
            return f"Error getting employment verification: {str(e)}"
    
    def save_basic_details(self, session_id: str) -> str:
        """
        Save basic user details, always prioritizing session data over input_str.

        Args:
            input_str: (ignored, always use session data)
            session_id: Session identifier

        Returns:
            Save result as JSON string
        """
        try:
            # Always use session data, ignore input_str
            if not session_id:
                return "Session ID is required"

            session = SessionManager.get_session_from_db(session_id)
            if not session or not session.get("data", {}):
                return "Session data not found"

            session_data = session["data"]

            # Get userId from session data
            user_id = session_data.get("userId")
            if not user_id:
                return "User ID is required"

            # Build data dict from session data, mapping to expected API fields
            data = {}

            # Name fields
            if session_data.get("fullName"):
                data["firstName"] = session_data.get("fullName")
            elif session_data.get("name"):
                data["firstName"] = session_data.get("name")

            # Phone number fields
            if session_data.get("mobileNumber"):
                data["mobileNumber"] = session_data.get("mobileNumber")
            elif session_data.get("phoneNumber"):
                data["mobileNumber"] = session_data.get("phoneNumber")
            elif session_data.get("phone"):
                data["mobileNumber"] = session_data.get("phone")
            else:
                return "Phone number is required"

            # Add other possible fields from session data if present - comprehensive mapping
            field_mappings = {
                "panCard": ["panCard", "pan", "panNo", "panNumber", "pan_card", "pan_number"],
                "gender": ["gender", "sex"],
                "dateOfBirth": ["dateOfBirth", "dob", "birthDate", "birth_date", "date_of_birth"],
                "emailId": ["emailId", "email", "email_id", "emailAddress", "email_address"],
                "firstName": ["firstName", "name", "first_name", "fullName", "full_name", "givenName", "given_name"],
                "treatmentCost": ["treatmentCost", "treatment_cost", "loanAmount", "loan_amount", "amount"],
                "monthlyIncome": ["monthlyIncome", "monthly_income", "income", "salary", "netTakeHomeSalary", "net_take_home_salary"]
            }
            
            # Apply field mappings
            for target_field, source_fields in field_mappings.items():
                for source_field in source_fields:
                    if session_data.get(source_field) is not None:
                        data[target_field] = session_data.get(source_field)
                        break  # Use first found value

            # Store the data being sent to the API
            SessionManager.update_session_data_field(session_id, "data.api_requests.save_basic_details", {
                "user_id": user_id,
                "data": data.copy()
            })

            result = self.api_client.save_basic_details(user_id, data)

            # Store the API response
            SessionManager.update_session_data_field(session_id, "data.api_responses.save_basic_details", result)

            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error saving basic details: {e}")
            return f"Error saving basic details: {str(e)}"
        
    def save_employment_details(self, session_id: str) -> str:
        """
        Save employment details

        Args:
            input_str: JSON string with employment data or user ID string
            session_id: Session identifier

        Returns:
            Save result as JSON string
        """
        try:
           
            user_id = None
            data = {}
            # Try to get user ID from session if not provided in input
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                if session and session.get("data", {}).get("userId"):
                    user_id = session["data"]["userId"]

            if not user_id:
                return "User ID is required"

            # Get employment verification API response from session
            employment_verification = None
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                if session:
                    employment_verification = session.get("data", {}).get("api_responses", {}).get("get_employment_verification")

            # Default to SELF_EMPLOYED
            employment_type = "SELF_EMPLOYED"
            organization_name = None

            if employment_verification and isinstance(employment_verification, dict):
                status = employment_verification.get("status")
                if status == 200:
                    employment_type = "SALARIED"
                    # Try to extract establishmentName from the deeply nested responseBody
                    data_field = employment_verification.get("data", {})
                    response_body = data_field.get("responseBody")
                    if response_body:
                        try:
                            # responseBody is a JSON string, so parse it
                            import json
                            response_json = json.loads(response_body)
                            # Traverse to result > result > summary > recentEmployerData > establishmentName
                            result_outer = response_json.get("result", {})
                            result_inner = result_outer.get("result", {})
                            summary = result_inner.get("summary", {})
                            recent_employer_data = summary.get("recentEmployerData", {})
                            establishment_name = recent_employer_data.get("establishmentName")
                            if establishment_name:
                                organization_name = establishment_name
                        except Exception as parse_exc:
                            logger.warning(f"Could not parse establishmentName from employment_verification: {parse_exc}")

            # Set employmentType in data
            data["employmentType"] = employment_type
            if organization_name:
                data["organizationName"] = organization_name

            # Get monthly income from session data if not in the input
            if ('netTakeHomeSalary' not in data or 'monthlyFamilyIncome' not in data) and session_id:
                session = SessionManager.get_session_from_db(session_id)
                if session and 'data' in session:
                    session_data = session['data']
                    if session_data.get('monthlyIncome'):
                        income = session_data.get('monthlyIncome')
                        if 'netTakeHomeSalary' not in data:
                            data['netTakeHomeSalary'] = income
                        if 'monthlyFamilyIncome' not in data:
                            data['monthlyFamilyIncome'] = income

            # Make sure we have the form status
            if 'formStatus' not in data:
                data['formStatus'] = "Employment"

            # Store the data being sent to the API
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.api_requests.save_employment_details", {
                    "user_id": user_id,
                    "data": data.copy()
                })

            result = self.api_client.save_employment_details(user_id, data)

            # Store the API response
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.api_responses.save_employment_details", result)

            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error saving employment details: {e}")
            return f"Error saving employment details: {str(e)}"
    
    def save_loan_details(self, input_str: str, session_id: str) -> str:
        """
        Save loan details

        Args:
            input_str: JSON string with loan data (ignored, details picked up from session data)
            
        Returns:
            Save result as JSON string
        """
        try:
            data = json.loads(input_str)
            user_id = data.get("userId")
            name = data.get("fullName")
            loan_amount = data.get("treatmentCost")

            # Try to get doctor_id and doctor_name from session data if not present in input
            doctor_id = data.get("doctorId") or data.get("doctor_id")
            doctor_name = data.get("doctorName") or data.get("doctor_name")

            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                if session and "data" in session:
                    session_data = session["data"]
                    # Try to get doctor_id and doctor_name from session data if not already set
                    if not doctor_id:
                        doctor_id = session_data.get("doctorId") or session_data.get("doctor_id")
                    if not doctor_name:
                        doctor_name = session_data.get("doctorName") or session_data.get("doctor_name")

            logger.info(f"Retrieved doctor_id {doctor_id} and doctor_name {doctor_name} from session for loan details")

            if not user_id or not name or not loan_amount:
                return "User ID, name, and loan amount are required"

            # Store the data being sent to the API
            SessionManager.update_session_data_field(session_id, "data.api_requests.save_loan_details", {
                "user_id": user_id,
                "name": name,
                "loan_amount": loan_amount,
                "doctor_name": doctor_name,
                "doctor_id": doctor_id
            })

            result = self.api_client.save_loan_details(user_id, name, loan_amount, doctor_name, doctor_id)
            
            # Store the API response
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.api_responses.save_loan_details", result)
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error saving loan details: {e}")
            return f"Error saving loan details: {str(e)}"
    

    
    def get_bureau_decision(self, session_id: str) -> str:
        """
        Get bureau decision for a loan
        
        Args:
            session_id: Session identifier
            
        Returns:
            Bureau decision as JSON string
        """
        try:
            # Initialize variables first
            loan_id = None

            # First try to get data from session
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                logger.info(f"Session retrieved: {session is not None}")
                
                if session and "data" in session:
                    session_data = session["data"]
                    logger.info(f"Session data keys: {list(session_data.keys())}")
                    
                    # Check if we already have bureau decision in session
                    if "api_responses" in session_data and "get_bureau_decision" in session_data["api_responses"]:
                        existing_decision = session_data["api_responses"]["get_bureau_decision"]
                        if existing_decision.get("status") == 200:
                            logger.info(f"Using existing bureau decision from session")
                            return json.dumps(existing_decision)
                    
                    # Try to get loan_id from different possible locations in session data
                    if "loanId" in session_data:
                        loan_id = session_data["loanId"]
                        logger.info(f"Found loan_id in session data: {loan_id}")

                    
                    # Also try to get from save_loan_details response
                    if not loan_id and "api_responses" in session_data and "save_loan_details" in session_data["api_responses"]:
                        save_loan_response = session_data["api_responses"]["save_loan_details"]
                        logger.info(f"save_loan_details response: {save_loan_response}")
                        if isinstance(save_loan_response, dict) and save_loan_response.get("status") == 200:
                            if "data" in save_loan_response and isinstance(save_loan_response["data"], dict):
                                loan_id = save_loan_response["data"].get("loanId")
                                logger.info(f"Found loan_id in save_loan_details response: {loan_id}")
                    
                    # Debug: Show what we have in api_responses
                    if "api_responses" in session_data:
                        logger.info(f"Available API responses: {list(session_data['api_responses'].keys())}")
                else:
                    logger.warning(f"No session data found for session_id: {session_id}")


            # Validate required parameters
            logger.info(f"Final loan_id before validation: '{loan_id}' (type: {type(loan_id)})")
            
            if not loan_id:
                logger.error("Loan ID is missing for bureau decision")
                logger.error(f"loan_id value: '{loan_id}', type: {type(loan_id)}")
                return json.dumps({"status": 400, "error": "Loan ID is required"})
                
            # Additional validation for loan_id
            if not isinstance(loan_id, str):
                logger.error(f"loan_id is not a string: {type(loan_id)}")
                return json.dumps({"status": 400, "error": "loan_id must be a string"})
                
            if loan_id.strip() == "":
                logger.error(f"loan_id is empty after stripping: '{loan_id}'")
                return json.dumps({"status": 400, "error": "loan_id is empty"})
                
            logger.info(f"Making bureau decision API call with loan_id: {loan_id}")
            logger.info(f"loan_id type: {type(loan_id)}, loan_id value: '{loan_id}'")
            
            # Make the API call
            try:
                result = self.api_client.get_bureau_decision(loan_id)
                logger.info(f"API call successful, result type: {type(result)}")
            except Exception as api_error:
                logger.error(f"API call failed with error: {api_error}")
                logger.error(f"loan_id passed to API: '{loan_id}' (type: {type(loan_id)})")
                raise
            
            # Store the complete API response in session data
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.api_responses.get_bureau_decision", result)
            
            # Log the raw API response for debugging
            logger.info(f"Bureau decision API response for loan ID {loan_id}: {json.dumps(result)}")
            
            # Process result to extract and format eligible EMI information
            if isinstance(result, dict) and result.get("status") == 200:
                bureau_result = self.extract_bureau_decision_details(result, session_id)
                # Store the result in session for easy reference using update_session_data_field
                if session_id:
                    SessionManager.update_session_data_field(session_id, "data.bureau_decision_details", bureau_result)
                    # logger.info(f"Stored bureau decision details in session: {bureau_result}")
                
                # Format the response using the new function
                formatted_response = self._format_bureau_decision_response(bureau_result, session_id)
                logger.info(f"Formatted response: {formatted_response}")
                
                # Ensure we always return a string
                if formatted_response is None:
                    logger.error("Formatted response is None, returning default message")
                    return "There was an error processing the loan decision. Please try again."
                
                return formatted_response
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error getting bureau decision: {e}")
            return json.dumps({
                "status": 500,
                "error": f"Error getting bureau decision: {str(e)}"
            })

    def extract_bureau_decision_details(self, bureau_result: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Extract and format eligible EMI details from a bureau decision result
        
        Args:
            bureau_result: Bureau decision API response
            
        Returns:
            Dictionary with formatted bureau decision details
        """
        try:
            details = {
                "status": None,
                "reason": None,
                "maxEligibleEMI": None,
                "emiPlans": [],
                "creditLimitCalculated": None
            }
            
            if not isinstance(bureau_result, dict) or bureau_result.get("status") != 200:
                return details
                
            data = bureau_result.get("data", {})
            
            # Extract status
            if "status" in data:
                details["status"] = data["status"]
            elif "bureauDecision" in data:
                details["status"] = data["bureauDecision"]
            
            # Log the extracted status for debugging
            logger.info(f"Extracted bureau decision status: {details['status']}")
            
            # Extract reason
            if "reason" in data:
                details["reason"] = data["reason"]
            elif "decisionReason" in data:
                details["reason"] = data["decisionReason"]
            elif "bureauChecks" in data and isinstance(data["bureauChecks"], list):
                for check in data["bureauChecks"]:
                    if isinstance(check, dict) and check.get("autoDecision") == "FAILED":
                        if "policyCheck" in check:
                            details["reason"] = f"Failed {check['policyCheck']} check"
                            break
            
            # Extract max eligible EMI
            if "maxEligibleEMI" in data:
                details["maxEligibleEMI"] = data["maxEligibleEMI"]
            elif "eligibleEMI" in data:
                details["maxEligibleEMI"] = data["eligibleEMI"]
            
            # Extract EMI plans and find max credit limit
            # Support both "emiPlanList" (original API) and "emiPlans" (formatted result)
            emi_plans_data = None
            if "emiPlanList" in data and isinstance(data["emiPlanList"], list):
                emi_plans_data = data["emiPlanList"]
            elif "emiPlans" in data and isinstance(data["emiPlans"], list):
                emi_plans_data = data["emiPlans"]
            
            if emi_plans_data:
                details["emiPlans"] = emi_plans_data
                
                # Find maximum creditLimit from all plans (note: field is "creditLimit" not "creditLimitCalculated")
                try:
                    max_credit_limit = max(
                        (float(plan.get("creditLimit", 0)) for plan in emi_plans_data if plan.get("creditLimit")),
                        default=None
                    )
                    if max_credit_limit:
                        details["creditLimitCalculated"] = str(int(max_credit_limit))
                except (ValueError, TypeError):
                    pass
                
                # If we have plans but no max eligible EMI, use the highest EMI
                if not details["maxEligibleEMI"] and emi_plans_data:
                    try:
                        highest_emi = max(
                            (float(plan.get("emi", 0)) for plan in emi_plans_data if plan.get("emi")),
                            default=None
                        )
                        if highest_emi:
                            details["maxEligibleEMI"] = str(int(highest_emi))
                    except (ValueError, TypeError):
                        pass
            
            # Ensure credit limit, EMI, and down payment values are strings
            for plan in details["emiPlans"]:
                for key in ["creditLimit", "emi", "downPayment"]:
                    if key in plan and plan[key] is not None and not isinstance(plan[key], str):
                        plan[key] = str(plan[key])
            
            # # Log the complete details dictionary for debugging
            # logger.info(f"Extracted bureau decision details: {details}")
            
            return details
        except Exception as e:
            logger.error(f"Error extracting bureau decision details: {e}")
            return {
                "status": None,
                "reason": None,
                "maxEligibleEMI": None,
                "emiPlans": [],
                "creditLimit": None
            }

    def process_prefill_data_for_basic_details(self, session_id: str) -> str:
        """
        Process prefill data and return a properly formatted JSON string for save_basic_details.
        This version fetches prefill data details directly from the API response stored in the session.

        Args:
            session_id: Session identifier.

        Returns:
            JSON string for save_basic_details
        """
        try:
            # 1. Get user_id if not provided
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                
            user_id = session.get("data", {}).get("userId")
            if not user_id:
                return "User ID is required to process prefill data"

            # 2. Get prefill data from API response in session
            prefill_data = {}
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                if session and "data" in session:
                    session_data = session["data"]
                    api_responses = session_data.get("api_responses", {})
                    prefill_api_result = api_responses.get("get_prefill_data")
                    if prefill_api_result and isinstance(prefill_api_result, dict):
                        # The actual prefill data is usually under "data" -> "response"
                        prefill_data = prefill_api_result.get("data", {}).get("response", {})
                    # Fallback: try "prefill_api_response" if not found above
                    if not prefill_data and "prefill_api_response" in session_data:
                        prefill_data = session_data["prefill_api_response"]

            # 3. Build the data for save_basic_details
            data = {"userId": user_id, "formStatus": "Basic"}

            # 4. Get name and phone from session if available
            session_data = {}
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                if session and "data" in session:
                    session_data = session["data"]

            # Add name from session if available
            if "name" in session_data:
                data["firstName"] = session_data["name"]
            elif "fullName" in session_data:
                data["firstName"] = session_data["fullName"]

            # Add phone number from session if available
            if "phone" in session_data:
                data["mobileNumber"] = session_data["phone"]
            elif "phoneNumber" in session_data:
                data["mobileNumber"] = session_data["phoneNumber"]
            elif "mobileNumber" in session_data:
                data["mobileNumber"] = session_data["mobileNumber"]

            # 5. Extract fields from prefill_data (from API response)
            field_mappings = {
                "panCard": ["pan", "panCard", "panNo", "panNumber", "pan_card", "pan_number"],
                "gender": ["gender", "sex"],
                "dateOfBirth": ["dateOfBirth", "dob", "birthDate", "birth_date", "date_of_birth"],
                "emailId": ["emailId", "email", "email_id", "emailAddress", "email_address"],
                "firstName": ["firstName", "name", "first_name", "fullName", "full_name", "givenName", "given_name"]
            }

            for target_field, source_fields in field_mappings.items():
                for source in source_fields:
                    if source in prefill_data and prefill_data[source]:
                        value = prefill_data[source]
                        
                        # Special handling for name fields to ensure we get a string
                        if target_field == "firstName":
                            if isinstance(value, dict):
                                # If it's a dict, try to extract firstName or name
                                if "firstName" in value:
                                    data[target_field] = str(value["firstName"])
                                elif "name" in value:
                                    data[target_field] = str(value["name"])
                                elif "fullName" in value:
                                    data[target_field] = str(value["fullName"])
                                else:
                                    # If no clear name field, skip this value
                                    continue
                            elif isinstance(value, str):
                                data[target_field] = value
                            else:
                                # Convert to string for other types
                                data[target_field] = str(value)
                        else:
                            # For non-name fields, ensure it's a string
                            if isinstance(value, (dict, list)):
                                # Skip complex objects for non-name fields
                                continue
                            else:
                                data[target_field] = str(value)
                        break

            # Special handling for email if it's a list or dict
            if "email" in prefill_data and prefill_data["email"] and "emailId" not in data:
                email_data = prefill_data["email"]
                if isinstance(email_data, list) and email_data:
                    if isinstance(email_data[0], dict) and "email" in email_data[0]:
                        data["emailId"] = email_data[0]["email"]
                    else:
                        data["emailId"] = email_data[0]
                else:
                    data["emailId"] = email_data

            # Also extract from nested "response" if it exists (sometimes API nests again)
            if "response" in prefill_data and isinstance(prefill_data["response"], dict):
                response = prefill_data["response"]
                for target_field, source_fields in field_mappings.items():
                    for source in source_fields:
                        if source in response and response[source] and target_field not in data:
                            value = response[source]
                            
                            # Special handling for name fields to ensure we get a string
                            if target_field == "firstName":
                                if isinstance(value, dict):
                                    # If it's a dict, try to extract firstName or name
                                    if "firstName" in value:
                                        data[target_field] = str(value["firstName"])
                                    elif "name" in value:
                                        data[target_field] = str(value["name"])
                                    elif "fullName" in value:
                                        data[target_field] = str(value["fullName"])
                                    else:
                                        # If no clear name field, skip this value
                                        continue
                                elif isinstance(value, str):
                                    data[target_field] = value
                                else:
                                    # Convert to string for other types
                                    data[target_field] = str(value)
                            else:
                                # For non-name fields, ensure it's a string
                                if isinstance(value, (dict, list)):
                                    # Skip complex objects for non-name fields
                                    continue
                                else:
                                    data[target_field] = str(value)
                            break
                # Special handling for email in nested response
                if "email" in response and response["email"] and "emailId" not in data:
                    email_data = response["email"]
                    if isinstance(email_data, list) and email_data:
                        if isinstance(email_data[0], dict) and "email" in email_data[0]:
                            data["emailId"] = email_data[0]["email"]
                        else:
                            data["emailId"] = email_data[0]
                    else:
                        data["emailId"] = email_data
                # Handle phone number in response if needed
                if "mobile" in response and response["mobile"] and "mobileNumber" not in data:
                    data["mobileNumber"] = response["mobile"]

            # Debug log what we're sending
            logger.info(f"Sending to save_basic_details: user_id={user_id}, data={data}")

            # Store the processed data in session for other methods to use
            if session_id:
                for key, value in data.items():
                    if key != "userId":  # Don't overwrite userId
                        SessionManager.update_session_data_field(session_id, f"data.{key}", value)
                logger.info(f"Stored processed prefill data in session: {data}")

            # Call the API client to save basic details
            result = self.api_client.save_basic_details(user_id, data)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error processing prefill data: {e}")
            if 'user_id' in locals() and user_id:
                return json.dumps({"userId": user_id, "error": str(e)})
            else:
                return json.dumps({"error": str(e)})

    def process_address_data(self, session_id: str) -> str:
        """
        Extract address information from prefill data and save it using save_address_details.
        Looks for 'Primary' address type and extracts postal code (pincode) and address line.
        If pincode is available, calls state_and_city_by_pincode API to get accurate city and state.

        Args:
            input_str: JSON string with userId or userId and prefill_data

        Returns:
            Save result as JSON string
        """
        user_id = None  # Ensure user_id is always defined
        try:
            if not session_id:
                return "Session ID is required"

            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return "Session not found"

            user_id = session.get("data", {}).get("userId")

            # Get prefill data from session if available
            prefill_data = None
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                if session and "data" in session:
                    session_data = session["data"]
                    # Try to get from data.api_responses.get_prefill_data first
                    api_responses = session_data.get("api_responses", {})
                    prefill_api_result = api_responses.get("get_prefill_data")
                    if prefill_api_result and isinstance(prefill_api_result, dict):
                        # Try to get the nested response
                        prefill_data = prefill_api_result.get("data", {}).get("response")
                    # Fallback to prefill_api_response if not found above
                    if not prefill_data and "prefill_api_response" in session_data:
                        prefill_data = session_data["prefill_api_response"]

            # Extract address information
            address_data = {}

            # Check if it's already in the right format
            if isinstance(prefill_data, dict) and "address" in prefill_data and isinstance(prefill_data["address"], list):
                address_list = prefill_data["address"]
                primary_address = None

                # First, try to find address with Type "Primary" or "Permanent"
                for addr in address_list:
                    addr_type = addr.get("Type", "").lower()
                    if addr_type in ["primary", "permanent"]:
                        primary_address = addr
                        break

                # If no primary address found, use the first one in the list
                if not primary_address and address_list:
                    primary_address = address_list[0]

                if primary_address:
                    # Extract address details
                    address_data["address"] = primary_address.get("Address", "")
                    address_data["pincode"] = primary_address.get("Postal", "")
                    address_data["state"] = primary_address.get("State", "")

                    # Clean up the pincode if needed (ensure it's a 6-digit code)
                    if address_data["pincode"] and len(address_data["pincode"]) > 0:
                        # Make sure it's 6 digits (if shorter, pad with zeros)
                        pincode = address_data["pincode"].strip()
                        if pincode.isdigit() and len(pincode) <= 6:
                            address_data["pincode"] = pincode.zfill(6)

                            # If we have a valid pincode, get city and state from API
                            try:
                                pincode_data = self.api_client.state_and_city_by_pincode(address_data["pincode"])
                                logger.info(f"Pincode API response for pincode {address_data['pincode']}: {pincode_data}")
                                if pincode_data and pincode_data.get("status") == "success":
                                    # Only update if we get valid non-null data
                                    if pincode_data.get("city") and pincode_data["city"] is not None:
                                        address_data["city"] = pincode_data["city"]
                                    if pincode_data.get("state") and pincode_data["state"] is not None:
                                        address_data["state"] = pincode_data["state"]
                            except Exception as e:
                                logger.warning(f"Failed to get city/state from pincode API: {e}")
                                # Continue with original data if API call fails

                    logger.info(f"Extracted address data: {address_data}")

                    # Store the extracted address data in session
                    if session_id:
                        SessionManager.update_session_data_field(session_id, "data.extracted_address_data", address_data)

                    # Save the address details
                    result = self.api_client.save_address_details(user_id, address_data)

                    # Store the API response
                    if session_id:
                        SessionManager.update_session_data_field(session_id, "data.api_responses.process_address_data", result)

                    return json.dumps(result)
                else:
                    return json.dumps({"error": "No address found in prefill data"})
            else:
                return json.dumps({"error": "Prefill data doesn't contain address information in expected format"})

        except Exception as e:
            logger.error(f"Error processing address data: {e}")
            return json.dumps({
                "error": f"Error processing address data: {str(e)}",
                "userId": user_id
            })
        
        
    def pan_verification(self, session_id: str) -> str:
        """
        Verify PAN details for a user
        
        Args:
            session_id: Session identifier
        
        Returns:
            Verification result as JSON string
        """
        try:
            user_id = None  # Initialize user_id variable
            
            # Try to get user ID from session
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                if session and session.get("data", {}).get("userId"):
                    user_id = session["data"]["userId"]
            
            if not user_id:
                return json.dumps({"status": 400, "error": "User ID is required for PAN verification"})
            
            logger.info(f"Performing PAN verification for user ID: {user_id}")
            result = self.api_client.pan_verification(user_id)
        
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.api_responses.pan_verification", result)
            return json.dumps({"status": 200, "data": result})
                
        except Exception as e:
            logger.error(f"Error verifying PAN: {e}")
            # Return a clear error response that the LLM should not ignore
            return json.dumps({
                "status": 500,
                "error": f"PAN verification failed: {str(e)}",
                "should_stop": True  # Flag to indicate this should stop the flow
            })

    def save_additional_user_details(self, input_str: str, session_id: str) -> str:
        """
        Save additional user details collected after bureau decision
        
        Args:
            input_str: JSON string with additional user details
            
        Returns:
            Confirmation message
        """
        try:
            data = json.loads(input_str)
            
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return "Session ID not found or invalid"
            
            # Extract additional details
            employment_type = data.get('employment_type')
            marital_status = data.get('marital_status')
            education_qualification = data.get('education_qualification')
            treatment_reason = data.get('treatment_reason')
            organization_name = data.get('organization_name', '')
            business_name = data.get('business_name', '')
            workplace_pincode = data.get('workplacePincode', '')
            
            # Create additional_details field if it doesn't exist
            current_additional_details = {}
            current_session = SessionManager.get_session_from_db(session_id)
            if current_session and "data" in current_session and "additional_details" in current_session["data"]:
                current_additional_details = current_session["data"]["additional_details"]
            
            # Update additional details
            additional_details = {
                "employment_type": employment_type,
                "marital_status": marital_status,
                "education_qualification": education_qualification,
                "treatment_reason": treatment_reason,
                "workplacePincode": workplace_pincode
            }
            
            # Add organization or business name based on employment type
            if employment_type == "SALARIED" and organization_name:
                additional_details["organization_name"] = organization_name
            elif employment_type == "SELF_EMPLOYED" and business_name:
                additional_details["business_name"] = business_name
                
            # Merge with existing additional details
            current_additional_details.update(additional_details)
                
            # Use update_session_data_field to preserve existing API audit trail data
            SessionManager.update_session_data_field(session_id, "data.additional_details", current_additional_details)
            
            # Get user ID from current session (fetch fresh data)
            current_session = SessionManager.get_session_from_db(session_id)
            user_id = None
            if current_session and "data" in current_session:
                user_id = current_session["data"].get("userId")
            
            # If we have a user ID, send employment details to API
            if user_id:
                employment_data = self._process_employment_data_from_additional_details(session_id)
                if employment_data:
                    try:
                        self.api_client.save_employment_details(user_id, employment_data)
                        # print(f"Successfully saved employment details for user {user_id}: {result}")
                        logger.info(f"Successfully saved employment details for user {user_id}: {employment_data}")
                    except Exception as e:
                        logger.error(f"Error saving employment details for user {user_id}: {e}")

            if user_id:
                loan_data = self._process_loan_data_from_additional_details(session_id)
                if loan_data:
                    try:
                        # Convert loan_data to JSON string for save_loan_details
                        self.api_client.save_loan_details_again(user_id, loan_data)
                        # print(f"Successfully saved loan details for user {user_id}: {result}")
                        logger.info(f"Successfully saved loan details for user {user_id}")

                        # logger.info(f"Successfully saved loan details for user {user_id}: {loan_data}")
                    except Exception as e:
                        logger.error(f"Error saving loan details for user {user_id}: {e}")

            if user_id:
                data = self._process_basic_details_from_additional_details(session_id)
                if data:
                    try:
                        self.api_client.save_basic_details(user_id, data)
                        # print(f"Successfully saved basic details for user {user_id}: {result}")
                        logger.info(f"Successfully saved basic details for user {user_id}: {data}")
                    except Exception as e:
                        logger.error(f"Error saving basic details for user {user_id}: {e}")

            return f"Additional details saved successfully for session {session_id}"
        except Exception as e:
            logger.error(f"Error saving additional user details: {e}")
            return f"Error saving details: {str(e)}"

    def _handle_additional_details_collection(self, session_id: str, message: str) -> str:
        """
        Handle the collection of additional details after bureau decision
        
        Args:
            session_id: Session identifier
            message: User message
            
        Returns:
            AI response message
        """
        import json
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return "Session not found. Please start a new conversation."
            
            # Ensure additional_details exists in session data
            if "additional_details" not in session["data"]:
                session["data"]["additional_details"] = {}
                
            additional_details = session["data"]["additional_details"]
            
            # Keep track of the current collection step
            # This is stored in session data to remember where we are in the collection flow
            collection_step = session["data"].get("collection_step", "employment_type")
            
            # Log current step for debugging
            logger.info(f"Session {session_id}: Processing step '{collection_step}' with message: {message.strip()}")
            
            # Function to save the current collection step and refresh session
            def update_collection_step(new_step):
                # Use update_session_data_field to preserve existing data
                SessionManager.update_session_data_field(session_id, "data.collection_step", new_step)
                SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                logger.info(f"Session {session_id}: Updated collection step to '{new_step}'")
            
            # Handle employment type input (first step)
            if collection_step == "employment_type":
                if "1" in message:
                    additional_details["employment_type"] = "SALARIED"
                    selected_option = "SALARIED"
                elif "2" in message:
                    additional_details["employment_type"] = "SELF_EMPLOYED"
                    selected_option = "SELF_EMPLOYED"
                else:
                    return "Please select a valid option for Employment Type: 1. SALARIED or 2. SELF_EMPLOYED"
                
                # Update session data with employment type using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step and ask for marital status
                update_collection_step("marital_status")
                return f"""You selected: {selected_option}

What is the Marital Status of the patient?
1. Married
2. Unmarried/Single\n
please Enter input 1 or 2 only"""
            
            # Handle marital status input
            elif collection_step == "marital_status":
                if "1" in message:
                    additional_details["marital_status"] = "1"
                    selected_option = "Married"
                elif "2" in message:
                    additional_details["marital_status"] = "2"
                    selected_option = "Unmarried/Single"
                else:
                    return "Please select a valid option for Marital Status: 1. Married or 2. Unmarried/Single"
                
                # Update session data with marital status using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step and ask for education qualification
                update_collection_step("education_qualification")
                return f"""You selected: {selected_option}

What is the Education Qualification of the patient?
1. Less than 10th
2. Passed 10th
3. Passed 12th
4. Diploma
5. Graduation
6. Post graduation
7. P.H.D\n
Please Enter input between 1 to 7 only"""
            
            # Handle education qualification input
            elif collection_step == "education_qualification":
                education_options = {
                    "1": "Less than 10th",
                    "2": "Passed 10th", 
                    "3": "Passed 12th",
                    "4": "Diploma",
                    "5": "Graduation",
                    "6": "Post graduation",
                    "7": "P.H.D"
                }
                
                if message.strip() in education_options:
                    additional_details["education_qualification"] = message.strip()
                    selected_option = education_options[message.strip()]
                else:
                    return "Please select a valid option for Education Qualification (1-7)"
                
                # Update session data with education qualification using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step and ask for treatment reason
                update_collection_step("treatment_reason")
                return f"""You selected: {selected_option}

What is the name of treatment?"""
            
            # Handle treatment reason input
            elif collection_step == "treatment_reason":
                additional_details["treatment_reason"] = message.strip()
                
                # Update session data with treatment reason using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)

                # Check if employment_type is SALARIED and if employment_verification API response is status 200
                if additional_details.get("employment_type") == "SALARIED":
                    # Fetch session to get api_responses
                    session = SessionManager.get_session_from_db(session_id)
                    session_data = session.get("data", {}) if session else {}
                    api_responses = session_data.get("api_responses", {})
                    employment_verification = api_responses.get("get_employment_verification")
                    organization_name = None

                    # Check if employment_verification is status 200 and try to extract organization name
                    if (
                        employment_verification
                        and isinstance(employment_verification, dict)
                        and employment_verification.get("status") == 200
                    ):
                        data_field = employment_verification.get("data", {})
                        response_body = data_field.get("responseBody")
                        if response_body:
                            try:
                                import json
                                response_json = json.loads(response_body)
                                # Traverse to result > result > summary > recentEmployerData > establishmentName
                                result_outer = response_json.get("result", {})
                                result_inner = result_outer.get("result", {})
                                summary = result_inner.get("summary", {})
                                recent_employer_data = summary.get("recentEmployerData", {})
                                establishment_name = recent_employer_data.get("establishmentName")
                                if establishment_name:
                                    organization_name = establishment_name
                            except Exception as parse_exc:
                                logger.warning(f"Could not parse establishmentName from employment_verification: {parse_exc}")

                    if organization_name:
                        additional_details["organization_name"] = organization_name
                        # Update session data with organization name
                        SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                        # Skip asking for organization name, go directly to workplace pincode
                        update_collection_step("workplace_pincode")
                        return f"""Treatment reason noted: {message.strip()}

What is the workplace/office pincode? (This is different from your home address pincode)
Please enter 6 digits:"""
                    else:
                        # If not found, ask for organization name as usual
                        additional_details["organization_name"] = ""  # Initialize organization name
                        update_collection_step("organization_name")
                        return f"""Treatment reason noted: {message.strip()}

What is the Organization Name where the patient works?"""
                else:
                    additional_details["business_name"] = ""  # Initialize business name
                    update_collection_step("business_name")
                    return f"""Treatment reason noted: {message.strip()}

What is the Business Name of the patient?"""
            
            # Handle organization name input (for SALARIED)
            elif collection_step == "organization_name":
                additional_details["organization_name"] = message.strip()
                
                # Update session data using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step to ask for workplace pincode
                update_collection_step("workplace_pincode")
                return f"""Organization name noted: {message.strip()}

What is the workplace/office pincode? (This is different from your home address pincode - we need the pincode where you work)
Please enter 6 digits:"""
            
            # Handle business name input (for SELF_EMPLOYED)
            elif collection_step == "business_name":
                additional_details["business_name"] = message.strip()
                
                # Update session data using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step to ask for workplace pincode
                update_collection_step("workplace_pincode")
                return f"""Business name noted: {message.strip()}

What is your business location pincode? (This is different from your home address pincode - we need the pincode where your business is located)
Please enter 6 digits:"""

            # Handle workplace pincode input
            elif collection_step == "workplace_pincode":
                # Validate pincode (6 digit number)
                pincode = message.strip()
                if not pincode.isdigit() or len(pincode) != 6:
                    return "Please enter a valid 6-digit workplace pincode (numbers only)."
                
                additional_details["workplacePincode"] = pincode
                
                # Update session data using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Mark collection as complete
                update_collection_step("complete")
                
                # Save all collected details using the tool
                # Make sure to create a new copy to avoid reference issues
                details_to_save = dict(additional_details)
                result = self.save_additional_user_details(json.dumps(details_to_save), session_id)
                
                # Use update_session_data_field to preserve existing data instead of overwriting
                SessionManager.update_session_data_field(session_id, "status", "additional_details_completed")
                SessionManager.update_session_data_field(session_id, "data.details_collection_timestamp", datetime.now().isoformat())
                
                # Get profile link to show to the user
                profile_link = self._get_profile_link(session_id)

                # Check if doctor is mapped by FIBE
                doctor_id = session["data"].get("doctorId") or session["data"].get("doctor_id")
                logger.info(f"Session {session_id}: Doctor ID: {doctor_id}")
                fibe_link_to_display = None  # Initialize a variable to hold the Fibe link if applicable
                
                if doctor_id:
                    try:
                        # Check if the method exists before calling it
                        if hasattr(self.api_client, 'check_doctor_mapped_by_nbfc'):
                            check_doctor_mapped_by_nbfc_response = self.api_client.check_doctor_mapped_by_nbfc(doctor_id)
                            logger.info(f"Session {session_id}: Check doctor mapped by FIBE response for doctor_id {doctor_id}: {json.dumps(check_doctor_mapped_by_nbfc_response)}")

                            # Handle both status 200 and non-500 statuses that indicate success
                            if check_doctor_mapped_by_nbfc_response.get("status") == 200:
                                doctor_mapped_by_nbfc = check_doctor_mapped_by_nbfc_response.get("data")
                                if doctor_mapped_by_nbfc == "true":
                                    logger.info(f"Session {session_id}: Doctor {doctor_id} is mapped by FIBE. Proceeding to FIBE flow.")
                                    
                                    # Doctor is mapped, get user_id and proceed with FIBE flow
                                    user_id = session["data"].get("userId")
                                    if user_id:
                                        try:
                                            # Call profile_ingestion_for_fibe API first
                                            profile_ingestion_response = self.api_client.profile_ingestion_for_fibe(user_id)
                                            logger.info(f"Session {session_id}: Fibe profile ingestion response for user_id {user_id}: {json.dumps(profile_ingestion_response) if profile_ingestion_response else 'None'}")
                                            
                                            # Store the API response in session data
                                            SessionManager.update_session_data_field(session_id, "data.api_responses.profile_ingestion_for_fibe", profile_ingestion_response)

                                            # Store potential fibe link from profile ingestion
                                            potential_fibe_link = None
                                            if profile_ingestion_response and isinstance(profile_ingestion_response, dict):
                                                response_status = profile_ingestion_response.get("status")
                                                if response_status == 200:
                                                    ingestion_data = profile_ingestion_response.get("data")
                                                    if ingestion_data and isinstance(ingestion_data, dict):
                                                        potential_fibe_link = ingestion_data.get("bitlyUrl")
                                                        if potential_fibe_link:
                                                            logger.info(f"Session {session_id}: Retrieved Fibe Bitly URL for user_id {user_id}: {potential_fibe_link}")
                                                        else:
                                                            logger.warning(f"Session {session_id}: No bitlyUrl found in Fibe profile ingestion response for user_id {user_id}: {ingestion_data}")
                                                    else:
                                                        lead_status = ingestion_data.get('leadStatus') if isinstance(ingestion_data, dict) else "N/A"
                                                        logger.info(f"Session {session_id}: Fibe profile ingestion for user_id {user_id} did not result in APPROVED leadStatus. Status: {lead_status}")
                                                else:
                                                    logger.error(f"Session {session_id}: Fibe profile ingestion API call failed for user_id {user_id}. Status: {response_status}")
                                            else:
                                                logger.error(f"Session {session_id}: Fibe profile ingestion API call returned invalid response for user_id {user_id}. Response: {profile_ingestion_response}")

                                            # Now call check_fibe_flow API
                                            check_fibe_response = self.api_client.check_fibe_flow(user_id)
                                            logger.info(f"Session {session_id}: Fibe flow check response for user_id {user_id}: {json.dumps(check_fibe_response)}")

                                            # Store the API response in session data
                                            SessionManager.update_session_data_field(session_id, "data.api_responses.check_fibe_flow", check_fibe_response)

                                            # Process based on check_fibe_flow response
                                            if check_fibe_response.get("status") == 200:
                                                fibe_status_data = check_fibe_response.get("data")
                                                if fibe_status_data == "GREEN":
                                                    logger.info(f"Session {session_id}: Fibe flow is GREEN for user_id {user_id}. Using Fibe link.")
                                                    # Use the fibe link for GREEN status
                                                    fibe_link_to_display = potential_fibe_link
                                                elif fibe_status_data == "AMBER":
                                                    logger.info(f"Session {session_id}: Fibe flow is AMBER for user_id {user_id}. Using Fibe link.")
                                                    # Use the fibe link for AMBER status as well
                                                    fibe_link_to_display = potential_fibe_link
                                                elif fibe_status_data == "RED":
                                                    logger.info(f"Session {session_id}: Fibe flow is RED (REJECTED) for user_id {user_id}. Using profile link.")
                                                else:
                                                    logger.warning(f"Session {session_id}: Fibe flow check for user_id {user_id} returned an unexpected data value: {fibe_status_data}. Using profile link.")
                                            else:
                                                logger.warning(f"Session {session_id}: Fibe flow check API call failed or returned non-200 status for user_id {user_id}. Using profile link.")
                                        
                                        except Exception as e:
                                            logger.error(f"Session {session_id}: Exception during Fibe flow processing for user_id {user_id}: {e}", exc_info=True)
                                    else:
                                        logger.warning(f"Session {session_id}: 'userId' not found in session data. Skipping Fibe flow.")
                                else:
                                    logger.info(f"Session {session_id}: Doctor {doctor_id} is not mapped by FIBE (status: {doctor_mapped_by_nbfc}). Skipping Fibe flow.")
                            elif check_doctor_mapped_by_nbfc_response.get("status") == 500:
                                logger.warning(f"Session {session_id}: Check doctor mapped by FIBE API returned status 500 for doctor_id {doctor_id}. Falling back to profile link.")
                            else:
                                logger.warning(f"Session {session_id}: Check doctor mapped by FIBE API call failed for doctor_id {doctor_id}. Status: {check_doctor_mapped_by_nbfc_response.get('status')}")
                        else:
                            logger.warning(f"Session {session_id}: check_doctor_mapped_by_nbfc method not available in API client. Falling back to profile link.")
                    except Exception as e:
                        logger.error(f"Session {session_id}: Exception during doctor mapping check for doctor_id {doctor_id}: {e}", exc_info=True)
                else:
                    logger.warning(f"Session {session_id}: Doctor ID not found in session data. Skipping FIBE flow.")

                # Determine which link to return - always fallback to profile_link if fibe_link_to_display is None
                link_to_display = fibe_link_to_display if fibe_link_to_display else profile_link

                # Use the new centralized decision logic
                decision_result = self._determine_loan_decision(session_id, profile_link, fibe_link_to_display)
                decision_status = decision_result["status"]
                link_to_display = decision_result["link"]
                
                # Only show link if not rejected
                if decision_status == "REJECTED":
                    return f"""Workplace pincode noted: {pincode}

Thank you! Your application is now complete. Your Loan application decision: {decision_status}."""
                else:
                    return f"""Workplace pincode noted: {pincode}

Thank you! Your application is now complete. Loan application decision: {decision_status}. Please check your application status by visiting the following:
{link_to_display}"""
            
            # If collection is complete, give a final message
            elif collection_step == "complete":
                return "All required information has been collected. Thank you for providing the details."
            
            # Fallback for unknown state
            else:
                return "I'm not sure what information to collect next. Please start a new session."
                
        except Exception as e:
            logger.error(f"Error handling additional details collection: {e}")
            return "There was an error processing your information. Please try again."

    def _get_profile_link(self, session_id: str) -> str:
        """
        Get the profile completion link for the user
        
        Args:
            session_id: Session identifier
            
        Returns:
            Profile completion link URL (shortened)
        """  
        try:
            if not session_id:
                logger.error(f"Session ID not found")
                return "Session ID not found"
            
            session = SessionManager.get_session_from_db(session_id)
    
            # Get doctor ID from session
            doctor_id = session["data"].get("doctorId") or session["data"].get("doctor_id")
            
            # Call API to get profile completion link
            profile_link_response = self.api_client.get_profile_completion_link(doctor_id)
            logger.info(f"Profile completion link response: {json.dumps(profile_link_response)}")
            
            # Extract link from response
            if isinstance(profile_link_response, dict) and profile_link_response.get("status") == 200:
                profile_link = profile_link_response.get("data", "")
                profile_link = Helper.clean_url(profile_link)
            
                session["data"]["profile_completion_link"] = profile_link  # Shorten the URL before returning
                
                short_link = shorten_url(profile_link)
                logger.info(f"Shortened profile link: {short_link}")
                
                return short_link
        
        except Exception as e:
            logger.error(f"Error getting profile completion link: {e}")
            fallback_url = "https://carepay.money/patient/Gurgaon/Nikhil_Dental_Clinic/Nikhil_Salkar/e71779851b144d1d9a25a538a03612fc/"
            return Helper.clean_url(fallback_url)

    def _process_employment_data_from_additional_details(self, session_id: str) -> Dict[str, Any]:
        """
        Process employment data from additional details collected
        
        Args:
            session_id: Session identifier
            
        Returns:
            Employment data dict ready for API
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return {}
                
            additional_details = session["data"].get("additional_details", {})
            session_data = session["data"]
            
            # Create employment data structure
            employment_data = {}

            if "monthlyIncome" in session_data:
                employment_data["monthlyFamilyIncome"] = session_data["monthlyIncome"]
            elif "monthlyFamilyIncome" in session_data:
                employment_data["monthlyFamilyIncome"] = session_data["monthlyFamilyIncome"]

            if "monthlyIncome" in session_data:
                employment_data["netTakeHomeSalary"] = session_data["monthlyIncome"]
            
            # Map employment type
            if additional_details.get("employment_type"):
                employment_data["employmentType"] = additional_details["employment_type"]
            
            # Map organization or business name
            if employment_data.get("employmentType") == "SALARIED" and additional_details.get("organization_name"):
                employment_data["organizationName"] = additional_details["organization_name"]
            elif employment_data.get("employmentType") == "SELF_EMPLOYED" and additional_details.get("business_name"):
                employment_data["nameOfBusiness"] = additional_details["business_name"]
                
            # Map workplace pincode if available
            if additional_details.get("workplacePincode"):
                employment_data["workplacePincode"] = additional_details["workplacePincode"]
            
            # Return the employment data for API
            return employment_data
            
        except Exception as e:
            logger.error(f"Error processing employment data from additional details: {e}")
            return {}

    def _process_loan_data_from_additional_details(self, session_id: str) -> Dict[str, Any]:
        """
        Process loan data from additional details collected
        
        Args:
            session_id: Session identifier

        Returns:
            Loan data dict ready for API
        """
        try:
            # Get session from database instead of self.sessions
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return {}
                
            additional_details = session["data"].get("additional_details", {})
            user_id = session["data"].get("userId")
            session_data = session["data"]
            
            # Create loan data structure with required fields
            loan_data = {
                "userId": user_id
            }
            
            # Add fullName from session data
            if "name" in session_data:
                loan_data["fullName"] = session_data["name"]
            elif "fullName" in session_data:
                loan_data["fullName"] = session_data["fullName"]

            if "doctor_id" in session_data:
                loan_data["doctorId"] = session_data["doctor_id"]
            elif "doctorId" in session_data:
                loan_data["doctorId"] = session_data["doctorId"]

            if "doctor_name" in session_data:
                loan_data["doctorName"] = session_data["doctor_name"]
            elif "doctorName" in session_data:
                loan_data["doctorName"] = session_data["doctorName"]
            
            # Add treatment cost from session data
            if "treatmentCost" in session_data:
                loan_data["treatmentCost"] = session_data["treatmentCost"]
            
            # Map treatment reason as loanReason
            if "treatment_reason" in additional_details:
                loan_data["loanReason"] = additional_details["treatment_reason"]

            # Return the loan data for API
            return loan_data
        
        except Exception as e:
            logger.error(f"Error processing loan data from additional details: {e}")
            return {}
        

    def _process_basic_details_from_additional_details(self, session_id: str) -> Dict[str, Any]:
        """
        Process basic details from additional details collected
        
        Args:
            session_id: Session identifier

        Returns:
            Basic details dict ready for API
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return {}
                
            additional_details = session["data"].get("additional_details", {})  
            user_id = session["data"].get("userId")
            session_data = session["data"]
         
            # Create basic details structure with userId
            basic_details = {
                "userId": user_id
            }

            if "name" in session_data:
                basic_details["fullName"] = session_data["name"]
            elif "fullName" in session_data:
                basic_details["fullName"] = session_data["fullName"]
            
            if "phoneNumber" in session_data:
                basic_details["mobileNumber"] = session_data["phoneNumber"]
            elif "phone" in session_data:
                basic_details["mobileNumber"] = session_data["phone"]
            
            # Map marital status: 1 -> Yes, 2 -> No
            if "marital_status" in additional_details:
                marital_status_map = {
                    "1": "Yes",
                    "2": "No"
                }
                basic_details["maritalStatus"] = marital_status_map.get(additional_details["marital_status"], additional_details["marital_status"])
            
            # Map education qualification to appropriate values
            if "education_qualification" in additional_details:
                education_level_map = {
                    "1": "LESS THAN 10TH",
                    "2": "PASSED 10TH",
                    "3": "PASSED 12TH",
                    "4": "DIPLOMA",
                    "5": "GRADUATION",
                    "6": "POST GRADUATION",
                    "7": "P.H.D."
                }
                basic_details["educationLevel"] = education_level_map.get(additional_details["education_qualification"], additional_details["education_qualification"])
            
            return basic_details

        except Exception as e:
            logger.error(f"Error processing basic details from additional details: {e}")
            return {}

    def _create_session_aware_tools(self, session_id: str):
        """
        Create tools that are aware of the current session_id
        
        Args:
            session_id: Current session identifier
            
        Returns:
            List of tools with session_id bound
        """
        return [
            Tool(
                name="store_user_data",
                func=lambda input_str: self.store_user_data(input_str, session_id),
                description="Store user data in session with the four parameters: fullName, phoneNumber, treatmentCost , monthlyIncome ",
            ),
            Tool(
                name="get_user_id_from_phone_number",
                func=lambda phone_number: self.get_user_id_from_phone_number(phone_number, session_id),
                description="Get userId from response of API call get_user_id_from_phone_number",
            ),
            Tool(
                name="save_basic_details",
                func=lambda session_id: self.save_basic_details(session_id),
                description="Save user's basic personal details. Call this tool using session_id ",
            ),
            Tool(
                name="save_loan_details",
                func=lambda input_str: self.save_loan_details(input_str, session_id),
                description="Save user's loan details. Must pass either a user ID as a string or a JSON object with userId and other fields like fullName, treatmentCost, etc.",
            ),
            Tool(
                name="check_jp_cardless",
                func=lambda session_id: self.check_jp_cardless(session_id),
                description="Check eligibility for Juspay Cardless",
            ),
            Tool(
                name="get_prefill_data",
                func=lambda user_id=None: self.get_prefill_data(user_id, session_id),
                description="Get prefilled user data from user ID",
            ),
            
             Tool(
                name="process_prefill_data",
                func=lambda session_id : self.process_prefill_data_for_basic_details(session_id),
                description="Convert prefill data from get_prefill_data_for_basic_details to a properly formatted JSON for save_basic_details. Call this tool using session_id.",
            ),
            Tool(
                name="process_address_data",
                func=lambda session_id: self.process_address_data(session_id),
                description="Extract address information from prefill data and save it using save_address_details. Call this after process_prefill_data. Must include session_id parameter.",
            ),
            Tool(
                name="pan_verification",
                func=lambda session_id: self.pan_verification(session_id),
                description="Verify PAN details for a user using session_id",
            ),
            Tool(
                name="get_employment_verification",
                func=lambda session_id: self.get_employment_verification(session_id),
                description="Get employment verification data using session_id",
            ),
           
            Tool(
                name="save_employment_details",
                func=lambda session_id: self.save_employment_details(session_id),
                description="Save user's employment details using session_id",
            ),
            

            Tool(
                name="get_bureau_decision",
                func=lambda session_id: self.get_bureau_decision(session_id),
                description="Get bureau decision for loan application using session_id",
            ),
            Tool(
                name="get_session_data",
                func=lambda: self.get_session_data(session_id),
                description="Get current session data using session_id",
            ),
            
            Tool(
                name="get_profile_link",
                func=lambda session_id: self._get_profile_link(session_id),
                description="Get profile link for a user using session_id",
            ),
        ]

    def _determine_loan_decision(self, session_id: str, profile_link: str, fibe_link: str = None) -> Dict[str, str]:
        """
        Determine loan decision based on the complete decision flow:
        
        Decision Flow:
        1. If Fibe GREEN -> APPROVED with Fibe link
        2. If Fibe AMBER:
           - If bureau APPROVED -> APPROVED with profile link
           - Otherwise -> INCOME_VERIFICATION_REQUIRED with Fibe link
        3. If Fibe RED or profile ingestion 500 error -> Fall back to bureau decision with profile link
        4. If no Fibe status -> Use bureau decision with profile link
        5. If no decisions available -> PENDING with profile link
        
        Args:
            session_id: Session identifier
            profile_link: Profile completion link
            fibe_link: Fibe completion link (optional)
            
        Returns:
            Dictionary with 'status' and 'link' keys
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return {"status": "PENDING", "link": profile_link}
            
            # Get Fibe and bureau decisions from session
            api_responses = session["data"].get("api_responses", {})
            check_fibe_flow = api_responses.get("check_fibe_flow")
            profile_ingestion = api_responses.get("profile_ingestion_for_fibe")
            bureau_decision = session["data"].get("bureau_decision_details")
            
            fibe_status = None
            bureau_status = None
            
            # Check for profile ingestion 500 error
            if profile_ingestion and profile_ingestion.get("status") == 500:
                logger.info(f"Session {session_id}: Profile ingestion returned 500 error - treating as RED status")
                fibe_status = "RED"
            # Extract Fibe status if no 500 error
            elif check_fibe_flow and check_fibe_flow.get("status") == 200:
                fibe_status = check_fibe_flow.get("data")
                logger.info(f"Session {session_id}: Fibe status: {fibe_status}")
            
            # Extract bureau status
            if bureau_decision:
                bureau_status = bureau_decision.get("status")
                logger.info(f"Session {session_id}: Bureau status: {bureau_status}")
            
            # Apply decision flow logic
            decision_status = None
            link_to_use = profile_link
            
            # 1. If Fibe GREEN -> APPROVED with Fibe link
            if fibe_status == "GREEN":
                decision_status = "APPROVED"
                link_to_use = fibe_link if fibe_link else profile_link
                logger.info(f"Session {session_id}: Fibe GREEN -> APPROVED with Fibe link")
            
            # 2. If Fibe AMBER
            elif fibe_status == "AMBER":
                # If bureau APPROVED -> APPROVED with profile link
                if bureau_status and bureau_status.upper() == "APPROVED":
                    decision_status = "APPROVED"
                    link_to_use = profile_link
                    logger.info(f"Session {session_id}: Fibe AMBER + Bureau APPROVED -> APPROVED with profile link")
                # Otherwise -> INCOME_VERIFICATION_REQUIRED with Fibe link
                else:
                    decision_status = "INCOME_VERIFICATION_REQUIRED"
                    link_to_use = fibe_link if fibe_link else profile_link
                    logger.info(f"Session {session_id}: Fibe AMBER + Bureau not APPROVED -> INCOME_VERIFICATION_REQUIRED with Fibe link")
            
            # 3. If Fibe RED or profile ingestion 500 error -> Fall back to bureau decision with profile link
            elif fibe_status == "RED":
                if bureau_status and bureau_status.upper() == "APPROVED":
                    decision_status = "APPROVED"
                elif bureau_status and bureau_status.upper() == "REJECTED":
                    decision_status = "REJECTED"
                elif bureau_status and bureau_status.upper() == "INCOME_VERIFICATION_REQUIRED":
                    decision_status = "INCOME_VERIFICATION_REQUIRED"
                else:
                    decision_status = "PENDING"
                link_to_use = profile_link
                logger.info(f"Session {session_id}: Fibe RED or profile ingestion 500 error -> Using bureau decision ({bureau_status}) with profile link")
            
            # 4. If no Fibe status -> Use bureau decision with profile link
            elif fibe_status is None:
                if bureau_status and bureau_status.upper() == "APPROVED":
                    decision_status = "APPROVED"
                elif bureau_status and bureau_status.upper() == "REJECTED":
                    decision_status = "REJECTED"
                elif bureau_status and bureau_status.upper() == "INCOME_VERIFICATION_REQUIRED":
                    decision_status = "INCOME_VERIFICATION_REQUIRED"
                else:
                    decision_status = "PENDING"
                link_to_use = profile_link
                logger.info(f"Session {session_id}: No Fibe status -> Using bureau decision ({bureau_status}) with profile link")
            
            # 5. If no decisions available -> PENDING with profile link
            if decision_status is None:
                decision_status = "PENDING"
                link_to_use = profile_link
                logger.info(f"Session {session_id}: No decisions available -> PENDING with profile link")
            
            logger.info(f"Session {session_id}: Final decision - Status: {decision_status}, Link: {link_to_use}")
            
            return {
                "status": decision_status,
                "link": link_to_use
            }
            
        except Exception as e:
            logger.error(f"Error determining loan decision for session {session_id}: {e}")
            return {"status": "PENDING", "link": profile_link}

    def check_jp_cardless(self, session_id: str) -> Dict[str, Any]:
        """
        Establish eligibility for Juspay Cardless
        """
        logger.info(f"Session {session_id}: Starting check_jp_cardless")
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session or "data" not in session:
                logger.error(f"Session {session_id}: Session data not found for check_jp_cardless.")
                return {"status": "ERROR", "message": "Session data not found."}

            session_data = session["data"]
            loan_id = session_data.get("loanId")
            # Try to get loanId from API response with safe access
            api_responses = session_data.get("api_responses", {})
            if api_responses and "save_loan_details" in api_responses:
                save_loan_response = api_responses["save_loan_details"]
                if isinstance(save_loan_response, dict) and "data" in save_loan_response:
                    # API response format: {"status": 200, "data": {"loanId": "...", ...}}
                    loan_id = save_loan_response["data"].get("loanId") or loan_id
                elif isinstance(save_loan_response, dict) and "loanId" in save_loan_response:
                    # Direct loanId in response
                    loan_id = save_loan_response.get("loanId") or loan_id
            
            logger.info(f"Session {session_id}: Retrieved loan_id: {loan_id} for check_jp_cardless")

            if not loan_id:
                logger.error(f"Session {session_id}: loanId not found in session data for check_jp_cardless.")
                return {"status": "ERROR", "message": "loanId not found in session."}

            result = self.api_client.check_eligibility_for_jp_cardless(loan_id)
            logger.info(f"Session {session_id}: check_eligibility_for_jp_cardless API response: {result}")
            profile_link = self._get_profile_link(session_id)

            if result and result.get("status") == 200:
                if result.get("data") == "ELIGIBLE":
                    logger.info(f"Session {session_id}: User is ELIGIBLE for Juspay Cardless based on check_eligibility.")
                    result1 = self.api_client.establish_eligibility(loan_id)
                    logger.info(f"Session {session_id}: establish_eligibility API response: {result1}")
                    # Check if status is 200 AND data is not empty/null
                    if result1 and result1.get("status") == 200:
                        data = result1.get("data")
                        # Check if data is not empty/null
                        if data and (isinstance(data, list) and len(data) > 0) or (isinstance(data, dict) and data) or (isinstance(data, str) and data.strip()):
                            logger.info(f"Session {session_id}: Juspay Cardless eligibility ESTABLISHED with valid data.")
                            # Update session status to indicate Juspay Cardless approval
                            SessionManager.update_session_data_field(session_id, "data.juspay_cardless_status", "APPROVED")
                            
                            # Get patient name from session data
                            patient_name = session_data.get("name") or session_data.get("fullName", "Patient")
                            
                            # Create Juspay Cardless specific approval message
                            formatted_response = f"""### Loan Application Decision:

🎉 Congratulations, {patient_name}! Your loan application has been **APPROVED** for Cardless EMI.

Continue your journey with the link here:
{profile_link}"""
                            
                            return {"status": "ELIGIBLE", "message": formatted_response}
                        else:
                            logger.info(f"Session {session_id}: Juspay Cardless eligibility NOT established - data is empty/null. Data: {data}")
                            # Update session status to indicate Juspay Cardless rejection
                            SessionManager.update_session_data_field(session_id, "data.juspay_cardless_status", "REJECTED")
                            return {"status": "NOT_ELIGIBLE", "message": "This application is not eligible for Juspay Cardless."}
                    else:
                        logger.info(f"Session {session_id}: Juspay Cardless eligibility NOT established or API error. API response: {result1}")
                        # Update session status to indicate Juspay Cardless rejection
                        SessionManager.update_session_data_field(session_id, "data.juspay_cardless_status", "REJECTED")
                        return {"status": "NOT_ELIGIBLE", "message": "This application is not eligible for Juspay Cardless."}
                else:
                    logger.info(f"Session {session_id}: User is NOT_ELIGIBLE for Juspay Cardless based on check_eligibility. Data: {result.get('data')}")
                    # Update session status to indicate Juspay Cardless rejection
                    SessionManager.update_session_data_field(session_id, "data.juspay_cardless_status", "REJECTED")
                    return {"status": "NOT_ELIGIBLE", "message": "This application is not eligible for Juspay Cardless."}
            else:
                logger.warning(f"Session {session_id}: check_eligibility_for_jp_cardless API call failed or returned non-200 status. Response: {result}")
                # Update session status to indicate Juspay Cardless error
                SessionManager.update_session_data_field(session_id, "data.juspay_cardless_status", "ERROR")
                return {"status": "API_ERROR", "message": "Could not check Juspay Cardless eligibility due to an API error."}
            
        except Exception as e:
            logger.error(f"Error establishing eligibility for Juspay Cardless for session {session_id}: {e}", exc_info=True)
            # Update session status to indicate Juspay Cardless error
            SessionManager.update_session_data_field(session_id, "data.juspay_cardless_status", "ERROR")
            return {"status": "EXCEPTION", "message": "An unexpected error occurred while checking Juspay Cardless eligibility."}

    def _validate_and_handle_early_chain_finish(self, session_id: str, response: Dict[str, Any], ai_message: str) -> None:
        """
        Validate that all required steps were executed and handle early chain finish
        
        Args:
            session_id: Session identifier
            response: Agent response
            ai_message: AI response message
        """
        try:
            # Get session to check current state
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return
            
            # Check if we're in initial data collection phase (no userId yet)
            if session.get("status") == "active" and not session.get("data", {}).get("userId"):
                # This is initial data collection, check if all required steps were executed
                executed_tools = []
                if "intermediate_steps" in response:
                    executed_tools = [step[0].tool for step in response["intermediate_steps"]]
                
                # Define required steps for initial flow
                required_steps = [
                    "store_user_data",
                    "get_user_id_from_phone_number", 
                    "save_basic_details",
                    "save_loan_details",
                    "check_jp_cardless",
                    "get_prefill_data",
                    "process_prefill_data",
                    "process_address_data",
                    "pan_verification",
                    "get_employment_verification",
                    "save_employment_details",
                    "get_bureau_decision"
                ]
                
                # Check for missing critical steps
                missing_critical_steps = []
                for step in required_steps[:5]:  # Check first 5 critical steps
                    if step not in executed_tools:
                        missing_critical_steps.append(step)
                
                # Only log if we have some tools executed but missing critical ones
                if len(executed_tools) > 0 and missing_critical_steps:
                    logger.warning(f"Session {session_id}: Early chain finish detected. Missing steps: {missing_critical_steps}")
                    logger.warning(f"Session {session_id}: Executed tools: {executed_tools}")
                    
                    # Log this for debugging
                    SessionManager.update_session_data_field(session_id, "data.early_chain_finish", {
                        "timestamp": datetime.now().isoformat(),
                        "missing_steps": missing_critical_steps,
                        "executed_tools": executed_tools,
                        "ai_message": ai_message
                    })
                    
        except Exception as e:
            logger.error(f"Error in early chain finish validation: {e}")

    # def _should_retry_early_chain_finish(self, session_id: str, response: Dict[str, Any]) -> bool:
    #     """
    #     Check if we should retry due to early chain finish
        
    #     Args:
    #         session_id: Session identifier
    #         response: Agent response
            
    #     Returns:
    #         True if retry is needed
    #     """
    #     try:
    #         session = SessionManager.get_session_from_db(session_id)
    #         if not session:
    #             return False
            
    #         # Check if we have a userId - if yes, we're past initial data collection
    #         if session.get("data", {}).get("userId"):
    #             return False  # Don't retry if we have userId
            
    #         # Only retry if we're in initial data collection and missing critical steps
    #         if session.get("status") == "active":
    #             executed_tools = []
    #             if "intermediate_steps" in response:
    #                 executed_tools = [step[0].tool for step in response["intermediate_steps"]]
                
    #             # Check if we're missing critical steps
    #             critical_steps = ["save_loan_details", "check_jp_cardless", "get_prefill_data"]
    #             missing_critical = [step for step in critical_steps if step not in executed_tools]
                
    #             # Only retry if we have some tools executed but missing critical ones
    #             return len(executed_tools) > 0 and len(missing_critical) > 0
                
    #         return False
    #     except Exception as e:
    #         logger.error(f"Error checking retry condition: {e}")
    #         return False

    # def _retry_with_explicit_continuation(self, session_id: str, message: str) -> str:
    #     """
    #     Retry with explicit continuation prompt
        
    #     Args:
    #         session_id: Session identifier
    #         message: Original user message
            
    #     Returns:
    #         Retry response
    #     """
    #     try:
    #         logger.info(f"Session {session_id}: Retrying with explicit continuation")
            
    #         # Create a more explicit prompt for continuation
    #         explicit_prompt = """
    #         CRITICAL: You have completed some steps but need to continue. You MUST execute the remaining steps:

    #         1. Call save_loan_details with the user data
    #         2. Call check_jp_cardless 
    #         3. Call get_prefill_data
    #         4. Call process_prefill_data
    #         5. Call process_address_data
    #         6. Call pan_verification
    #         7. Call get_employment_verification
    #         8. Call save_employment_details
    #         9. Call get_bureau_decision

    #         Do NOT stop until you complete ALL remaining steps. Continue immediately.
    #         """
            
    #         # Create session-specific agent with explicit prompt
    #         session_tools = self._create_session_aware_tools(session_id)
            
    #         # Create the prompt with explicit continuation
    #         prompt = ChatPromptTemplate.from_messages([
    #             ("system", explicit_prompt),
    #             ("human", "{input}"),
    #             MessagesPlaceholder(variable_name="chat_history"),
    #             MessagesPlaceholder(variable_name="agent_scratchpad"),
    #         ])

    #         # Create session-specific agent
    #         agent = create_openai_functions_agent(self.llm, session_tools, prompt)
    #         session_agent_executor = AgentExecutor(
    #             agent=agent,
    #             tools=session_tools,
    #             verbose=True,
    #             max_iterations=50,
    #             handle_parsing_errors=True,
    #         )
            
    #         # Get session for chat history
    #         session = SessionManager.get_session_from_db(session_id)
    #         chat_history = session.get("history", []) if session else []
            
    #         # Execute with explicit continuation
    #         response = session_agent_executor.invoke({
    #             "input": "Continue with the remaining loan application steps", 
    #             "chat_history": chat_history
    #         })
            
    #         ai_message = response.get("output", "Continuing with loan application steps...")
            
    #         # Update conversation history efficiently
    #         self._update_session_history(session_id, message, ai_message)
            
    #         return ai_message
            
    #     except Exception as e:
    #         logger.error(f"Error in retry with explicit continuation: {e}")
    #         return "I'm continuing with your loan application. Please wait while I process the remaining steps."

    def _format_bureau_decision_response(self, bureau_decision: Dict[str, Any], session_id: str) -> str:
        """
        Format the bureau decision response based on the status and details
        
        Args:
            bureau_decision: Bureau decision response data
            session_id: Session identifier
            
        Returns:
            Formatted response message
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return "Session not found. Please start a new conversation."
            
            # Get patient name from session data
            patient_name = session["data"].get("name") or session["data"].get("fullName", "Patient")
            
            # Get treatment cost from session data
            treatment_cost = session["data"].get("treatmentCost")
            show_detailed_approval = False
            
            if treatment_cost:
                try:
                    cost_value = float(str(treatment_cost).replace(',', '').replace('₹', ''))
                    show_detailed_approval = cost_value >= 100000
                except (ValueError, TypeError):
                    show_detailed_approval = False
            
            # Get status from bureau decision
            status = bureau_decision.get("status")
            logger.info(f"Bureau decision status: '{status}' (type: {type(status)})")
            
            # Format response based on status (case-insensitive)
            if status and status.upper() == "APPROVED":
                if show_detailed_approval:
                    # Get loan amount from bureau decision
                    loan_amount = bureau_decision.get("loanAmount", 0)
                    
                    # Try to get down payment and gross treatment amount from emiPlans
                    emi_plans = bureau_decision.get("emiPlanList", [])
                    down_payment = treatment_cost - 100000
                    gross_treatment_amount = 100000
                    
                    try:
                        if emi_plans and isinstance(emi_plans, list) and len(emi_plans) > 0:
                            first_plan = emi_plans[0]
                            if isinstance(first_plan, dict):
                                # Get down payment from first plan
                                down_payment = first_plan.get("downPayment", 0)
                                # If gross treatment amount is not in plan, use loan amount
                                if not first_plan.get("grossTreatmentAmount"):
                                    gross_treatment_amount = loan_amount
                                else:
                                    gross_treatment_amount = first_plan.get("grossTreatmentAmount")
                    except Exception as e:
                        logger.error(f"Error processing EMI plan details: {e}")
                    
                    # Ensure numeric values for formatting
                    try:
                        gross_treatment_amount = float(str(gross_treatment_amount).replace(',', '').replace('₹', '')) if gross_treatment_amount else 0
                    except (ValueError, TypeError):
                        gross_treatment_amount = 0
                    
                    try:
                        down_payment = float(str(down_payment).replace(',', '').replace('₹', '')) if down_payment else 0
                    except (ValueError, TypeError):
                        down_payment = 0
                    
                    return f"""### Loan Application Decision:

🎉 Congratulations, {patient_name}! Your loan application has been **APPROVED**.

**Approval Details:**
- Gross Treatment Amount: ₹{gross_treatment_amount:,.0f}
- DownPayment: ₹{down_payment:,.0f}

Would you like to proceed without down payment? If yes, income verification will be required.

What is the Employment Type of the patient?   
1. SALARIED
2. SELF_EMPLOYED
Please Enter input 1 or 2 only"""
                else:
                    # Ensure numeric value for credit limit formatting
                    credit_limit = bureau_decision.get("creditLimitCalculated", 0)
                    try:
                        credit_limit = float(str(credit_limit).replace(',', '').replace('₹', '')) if credit_limit else 0
                    except (ValueError, TypeError):
                        credit_limit = 0
                    
                    return f"""### Loan Application Decision:

🎉 Congratulations, {patient_name}! Your loan application has been **APPROVED**.

What is the Employment Type of the patient?   
1. SALARIED
2. SELF_EMPLOYED
Please Enter input 1 or 2 only"""
            
            elif status and status.upper() == "REJECTED":
                return f"""Dear {patient_name}! Your application is still not Approved. We need 5 more information so that we can check your eligibility for a loan application.
What is the Employment Type of the patient?
1. SALARIED
2. SELF_EMPLOYED
Please Enter input 1 or 2 only"""
            
            elif status and "income verification" in status.lower():
                return f"""Dear {patient_name}! Your application is still not Approved. We need more 5 more info so that we will check your eligibility of loan Application
What is the Employment Type of the patient?
1. SALARIED
2. SELF_EMPLOYED
Please Enter input 1 or 2 only"""
            
            else:
                # Default case for unknown status
                logger.warning(f"Unknown bureau decision status: '{status}'")
                return f"""Dear {patient_name}! We are processing your loan application. Please wait while we check your eligibility.
What is the Employment Type of the patient?
1. SALARIED
2. SELF_EMPLOYED
Please Enter input 1 or 2 only"""
                
        except Exception as e:
            logger.error(f"Error formatting bureau decision response: {e}")
            return "There was an error processing the loan decision. Please try again."