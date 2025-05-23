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
from setup_env import setup_environment


setup_environment()

logger = logging.getLogger(__name__)

class CarepayAgent:
    """
    Carepay AI Agent using LangChain for managing loan application processes
    """
    
    def __init__(self):
        """Initialize the CarePay agent with LLM and tools"""
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=0.2,
        )
        
        # Initialize API client
        self.api_client = CarepayAPIClient()
        
        # Define system prompt
        self.system_prompt = """
        You are a healthcare loan application assistant for CarePay. Your role is to help users apply for loans for medical treatments in a professional and friendly manner.

        Follow these steps sequentially to process a loan application: and don't miss any step and any tool calling

        1. Initial Data Collection:
           - Greet the user warmly and introduce yourself as CarePay's healthcare loan assistant
           - Collect and validate these four essential pieces of information:
              * Patient's full name
              * Patient's phone number
              * Treatment cost (between ₹30,000 to ₹20,00,000)
              * Monthly income
           - Use the store_user_data tool to save this information in the session

        2. User ID Creation:
           - didn't miss this step
           - Use get_user_id_from_phone_number tool to get userId from the phone number
           - Extract the userId from the response and store it in the session
           - Use this userId for all subsequent API calls

        3. Basic Details Submission:
           - didn't miss this step
           - Retrieve name and phone number from session data
           - Use save_basic_details tool to submit these details along with the userId
           - IMPORTANT: When calling save_basic_details, format the data as a proper JSON object with the userId field included

        4. Data Prefill:
           - didn't miss this step
           - Use get_prefill_data tool to retrieve user details using the userId and renmber this userID  that will use in process_prefill_data_for_basic_details
           - Extract from the response: PAN number, gender, DOB, email (if available)
           - didn't forget the userId what give to process_prefill_data_for_basic_details
           - Use the process_prefill_data_for_basic_details tool to formate process_prefill_data for save_basic_details in process_prefill_data_for_basic_details there i calling save_basic_details with userId and prefill data as input  and also give userId in process_prefill_data_for_basic_details

        5. Address Processing:
           - didn't miss this step
           - After processing the prefill data, use the process_address_data tool to extract and save the user's address
           - Pass the userId to the tool to ensure it can retrieve the prefill data and extract the address information
           - This tool will extract the address line, pincode (postal code), and state from the primary/permanent address in the prefill data
           - The tool will automatically call save_address_details with the extracted information

        6. Employment Verification:
           - didn't miss this step
           - Use get_employment_verification tool to check employment status using userId
           - Determine if user is SALARIED or SELF-EMPLOYED based on the response if found then save emploment_Details with userId and employment_Details accordingly
           - If employment data is not found, message: no records found then go with SALARIED

        7. Save Employment Details:
           - didn't miss this step
           - Use save_employment_details tool to submit:
              * Employment type (SALARIED or SELF-EMPLOYED)
              * Monthly income (from initial data collection)
              * Organization name (if available from verification) other wise pass empty string
           - IMPORTANT: Format the data as a proper JSON with the userId and required fields

        8. Save Loan Details:
           - didn't miss this step
           - Use save_loan_details tool to submit:
              * User's full name (from initial data collection)
              * Treatment cost (from initial data collection)
              * User ID

        9. Process Loan Application:
           - didn't miss this step
           - Use get_loan_details tool to retrieve loanId using userId
           - Use get_bureau_report tool to check if the bureau report API call is successful (this tool only returns API status, not the full report) that using by loanId( for API call) and didn't forget the hit this tool APU call

           - Use get_bureau_decision tool to get final loan decision using loanId and doctorId

        10. Decision Communication and Additional Information Collection:
           - didn't miss this step
           - Format your loan decision response based on the status received from get_bureau_decision:
           
           - For APPROVED status ONLY:
             ```
             ### Loan Application Decision:
             
             🎉 Congratulations, [PATIENT_NAME]! Your loan application has been **APPROVED**.
             
             What is the Employment Type of the patient?   
             1. SALARIED
             2. SELF-EMPLOYED
             ```
           
           - For REJECTED status:
             ```
             ### Loan Application Decision:
             
             We regret to inform you that your loan application has been **REJECTED**.
             Reason: [REJECTION_REASON]
             ```
             
           - For INCOME_VERIFICATION_REQUIRED status:
             ```
             ### Loan Application Decision:
             
             Your application requires **INCOME VERIFICATION**. 
             
             What is the Employment Type of the patient?    
             1. SALARIED
             2. SELF-EMPLOYED
             ```
             
           - For any other status:
             ```
             ### Loan Application Decision:
             
             Your loan application status is: **[BUREAU_DECISION]**.
         
             ```
           
           - IMPORTANT: NEVER display "APPROVED" status for any application that has status INCOME_VERIFICATION_REQUIRED or any other status that is not explicitly "APPROVED".
           - IMPORTANT: Strictly use the exact status value received from the get_bureau_decision API response.
           - IMPORTANT: Keep your response simple and clean. Do NOT include any EMI plans, tenure details, or payment schedules in your response.
           
       11. Additional Information Collection (For APPROVED or INCOME_VERIFICATION_REQUIRED status only):
           - After the user selects an Employment Type (1 or 2), collect the following information in sequence:
           
           - Step 1: Ask for Marital Status:
             ```
             What is the Marital Status of the patient?
             1. Married
             2. Unmarried/Single
             ```
           
           - Step 2: Ask for Education Qualification:
             ```
             What is the Education Qualification of the patient?
             1. Less than 10th
             2. Passed 10th
             3. Passed 12th
             4. Diploma
             5. Graduation
             6. Post graduation
             7. P.H.D
             ```
           
           - Step 3: Ask for Treatment Reason:
             ```
             What is the reason for treatment?
             ```
           
           - Step 4: Ask for Organization/Business Name:
             - If SALARIED, ask:
               ```
               What is the Organization Name where the patient works?
               ```
             - If SELF-EMPLOYED, ask:
               ```
               What is the Business Name of the patient?
               ```
           
           - Step 5: Save Additional Details:
             - Use the save_additional_user_details tool to save all the collected information
             - Format the data as a JSON object with the following fields:
               * employment_type: "SALARIED" or "SELF-EMPLOYED"
               * marital_status: "1" (Married) or "2" (Unmarried/Single)
               * education_qualification: "1" through "7" based on selection
               * treatment_reason: (text entered by user)
               * organization_name: (if SALARIED)
               * business_name: (if SELF-EMPLOYED)
           

        Always maintain a professional, helpful tone throughout the conversation.
        """
        
        # Initialize session storage
        self.sessions = {}
        
        # Setup agent with tools and prompt
        self.setup_agent()
        
    def setup_agent(self):
        """
        Set up agent with tools
        
        Returns:
            Agent with tools
        """
        # Define tools
        tools = [
            Tool(
                name="get_user_id_from_phone_number",
                func=self.get_user_id_from_phone_number,
                
                description="Get userId from response of API call get_user_id_from_phone_number",
            ),
            Tool(
                name="save_basic_details",
                func=self.save_basic_details,
               
                description="Save user's basic personal details. Must pass either a user ID as a string or a JSON object with userId and other fields like panCard, gender, dateOfBirth, etc.",
            ),
            Tool(
                name="get_prefill_data",
                func=self.get_prefill_data,
                
                description="Get prefilled user data from user ID",
            ),
             Tool(
                name="process_prefill_data",
                func=self.process_prefill_data_for_basic_details,
               
                description="Convert prefill data from get_prefill_data to a properly formatted JSON for save_basic_details. MUST include both prefill_data and user_id parameters.",
            ),
            Tool(
                name="process_address_data",
                func=self.process_address_data,
               
                description="Extract address information from prefill data and save it using save_address_details. Call this after process_prefill_data. Must include userId parameter.",
            ),
            Tool(
                name="save_address_details",
                func=self.save_address_details,
               
                description="Save address details for a user. Requires userId and address object.",
            ),
            Tool(
                name="get_employment_verification",
                func=self.get_employment_verification,
               
                description="Get employment verification data for a user ID",
            ),
           
            Tool(
                name="save_employment_details",
                func=self.save_employment_details,
                
                description="Save user's employment details",
            ),
            Tool(
                name="save_loan_details",
                func=self.save_loan_details,
                
                description="Save loan details for the user",
            ),
            Tool(
                name="get_loan_details",
                func=self.get_loan_details,
                
                description="Get loan details for a user",
            ),
            Tool(
                name="get_bureau_report",
                func=self.get_bureau_report,
                
                description="Get bureau report for a loan",
            ),
            Tool(
                name="get_bureau_decision",
                func=self.get_bureau_decision,
                
                description="Get bureau decision for loan application and input is loanId and doctorId that come from get_loan_details",
            ),
            Tool(
                name="get_session_data",
                func=self.get_session_data,
                
                description="Get current session data",
            ),
            Tool(
                name="store_user_data",
                func=self.store_user_data,
                
                description="Store user data in session",
            ),
        ]

        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=30,
            handle_parsing_errors=True,
        )

        return agent_executor
    
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
                "4. Monthly income of your patient.\n"
                "**Example input format: name: John Doe phone number: 1234567890 treatment cost: 10000 monthly income: 50000**"
            )
            
            # Create session with initial data
            session = {
                "id": session_id,
                "created_at": datetime.now().isoformat(),
                "status": "active",  
                "history": [AIMessage(content=initial_message)],  # Add initial message to history
                "data": {},  
                "phone_number": phone_number
            }
            
            # Make sure to serialize the AIMessage when first storing the session
            serializable_history = [{
                "type": "AIMessage",
                "content": initial_message
            }]
            
            # Use serializable history for database storage
            session["serializable_history"] = serializable_history
            
            # Store doctor information if provided
            if doctor_id:
                session["data"]["doctor_id"] = doctor_id
                logger.info(f"Stored doctor_id {doctor_id} in session {session_id}")
            
            if doctor_name:
                session["data"]["doctor_name"] = doctor_name
                logger.info(f"Stored doctor_name {doctor_name} in session {session_id}")
            
            # Store session
            self.sessions[session_id] = session
            logger.info(f"Created new session: {session_id}")
            
            # Save to database
            self.save_session_to_db(session_id)
            
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
            # Validate session_id
            if session_id not in self.sessions:
                # Try to load from database
                try:
                    session_data = SessionData.objects.get(session_id=session_id)
                    if session_data:
                        # Convert serialized history back to Message objects
                        history = []
                        if session_data.history:
                            for msg in session_data.history:
                                if isinstance(msg, dict) and 'type' in msg and 'content' in msg:
                                    if msg['type'] == 'AIMessage':
                                        history.append(AIMessage(content=msg['content']))
                                    elif msg['type'] == 'HumanMessage':
                                        history.append(HumanMessage(content=msg['content']))
                                    else:
                                        history.append(msg)  # Keep as is if not recognized
                                else:
                                    history.append(msg)  # Keep as is if not in expected format
                        
                        # Restore session from database
                        self.sessions[session_id] = {
                            "id": str(session_data.session_id),
                            "data": session_data.data or {},
                            "history": history,  # Use the converted history
                            "status": session_data.status or "active",
                            "phone_number": session_data.phone_number
                        }
                        logger.info(f"Session {session_id} restored from database")
                except SessionData.DoesNotExist:
                    logger.warning(f"Session {session_id} not found in database")
                    # Still continue with an empty session to avoid breaking the flow
                    self.sessions[session_id] = {
                        "id": session_id,
                        "data": {},
                        "history": [],
                        "status": "active",
                        "created_at": datetime.now().isoformat()
                    }
            
            # Set current session for helper methods
            self._current_session_id = session_id
            
            # Get session data
            session = self.sessions[session_id]
            current_status = session.get("status", "active")
            
            # We're no longer using the 'expired' status to prevent early session termination
            # Instead, we'll keep the session active regardless of status
            # This commented code is left for reference
            # if current_status == "expired" and "decision_reviewed" in session.get("data", {}):
            #     return "Session has expired after loan decision. Please start a new chat to continue."
            
            phone = session.get("data", {}).get("phone", None)
            
            logger.info(f"Processing message in session {session_id} with status {current_status} and phone {phone}")
            
            # Add user message to history
            chat_history = session.get("history", [])
            
            # Check if this is an acknowledgment of bureau decision review
            if current_status == "bureau_decision_sent" and any(keyword in message.lower() for keyword in ["understood", "ok", "thank", "got it", "received"]):
                session["data"]["decision_reviewed"] = True
                # Instead of expiring the session, mark it for additional data collection
                session["status"] = "collecting_additional_details"
                # Initialize the additional_details dictionary and collection_step
                if "additional_details" not in session["data"]:
                    session["data"]["additional_details"] = {}
                session["data"]["collection_step"] = "employment_type"
                self.save_session_to_db(session_id)
                return "Thank you for confirming. Now I'll collect some additional information. What is the Employment Type of the patient?\n1. SALARIED\n2. SELF-EMPLOYED"

            chat_history.append(HumanMessage(content=message))
            
            # Special handling for collecting additional details state
            if current_status == "collecting_additional_details":
                # Use direct sequential flow instead of full agent for efficiency
                ai_message = self._handle_additional_details_collection(session_id, message)
                chat_history.append(AIMessage(content=ai_message))
                session["history"] = chat_history
                
                # Update serializable_history for this special case too
                if "serializable_history" not in session:
                    session["serializable_history"] = []
                
                # Add the human message to serializable history
                session["serializable_history"].append({
                    "type": "HumanMessage",
                    "content": message
                })
                
                # Add the AI response to serializable history
                session["serializable_history"].append({
                    "type": "AIMessage",
                    "content": ai_message
                })
                
                self.save_session_to_db(session_id)
                return ai_message
            
            # Use LLM to generate response
            agent_executor = self.setup_agent()
            response = agent_executor.invoke({"input": message, "chat_history": chat_history})
            
            # Log all intermediate steps to ensure visibility of tool outputs
            if "intermediate_steps" in response:
                for i, step in enumerate(response["intermediate_steps"]):
                    action = step[0]
                    action_output = step[1]
                    
                    # Log each step with its tool name
                    logger.info(f"Step {i+1}: Tool: {action.tool}")
                    
                    # Special handling for bureau report to ensure it's visible
                    if action.tool == "get_bureau_report":
                        logger.info(f"Bureau Report Status: {action_output}")
                    else:
                        # Truncate other outputs if they're too long
                        output_str = str(action_output)
                        if len(output_str) > 1000:
                            logger.info(f"Tool Output (truncated): {output_str[:1000]}...")
                        else:
                            logger.info(f"Tool Output: {output_str}")
            
            # Extract LLM response and add to history
            ai_message = response.get("output", "I don't know how to respond to that.")
            chat_history.append(AIMessage(content=ai_message))
            
            # Also update serializable_history
            if "serializable_history" not in session:
                session["serializable_history"] = []
            
            # Add the human message
            session["serializable_history"].append({
                "type": "HumanMessage",
                "content": message
            })
            
            # Add the AI response
            session["serializable_history"].append({
                "type": "AIMessage",
                "content": ai_message
            })
            
            # Check if this is a bureau decision response
            if "bureau decision" in ai_message.lower() and any(plan in ai_message for plan in ["Plan:", "Plan", "EMI:", "/0", "/5", "/1"]):
                # Mark as decision sent (but we'll proceed directly to additional data collection)
                session["status"] = "bureau_decision_sent"
                logger.info(f"Session {session_id} marked as bureau_decision_sent")
                
                # We no longer need explicit acknowledgment since we're moving directly to additional data collection
                # Add a simple prompt that prepares for continuing instead
                ai_message += "\n\nPlease respond with 'OK' to continue with additional information collection."
            
            # Check if this is the loan application decision with approval or income verification required
            if (("loan application has been **APPROVED**" in ai_message or 
                "application requires **INCOME VERIFICATION**" in ai_message) and
                "what is the employment type of the patient?" in ai_message.lower()):
                # Mark session as collecting additional details
                session["status"] = "collecting_additional_details"
                # Initialize the additional_details dictionary and collection_step
                if "additional_details" not in session["data"]:
                    session["data"]["additional_details"] = {}
                session["data"]["collection_step"] = "employment_type"
                logger.info(f"Session {session_id} marked as collecting_additional_details")

            # Save updated history to session
            session["history"] = chat_history
            self.sessions[session_id] = session
            
            # Save updated session to database
            self.save_session_to_db(session_id)
            
            return ai_message
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "Please start a new chat session to continue our conversation."

    def get_session_data(self, session_id: str = None) -> str:
        """
        Get session data for the current session
        
        Args:
            session_id: Optional session ID, defaults to current session
            
        Returns:
            Session data as JSON string
        """
        if not session_id and hasattr(self, '_current_session_id'):
            session_id = self._current_session_id
        
        if not session_id or session_id not in self.sessions:
            return "Session ID not found"
        
        session = self.sessions[session_id]
        
        # Create a serializable copy of the session
        serializable_session = {
            "id": session.get("id"),
            "status": session.get("status"),
            "created_at": session.get("created_at"),
            "data": session.get("data", {}),
            "phone_number": session.get("phone_number")
        }
        
        # Convert history to serializable format
        serializable_history = []
        if "history" in session:
            for msg in session["history"]:
                if hasattr(msg, "content"):  # Check if it's a Message object
                    serializable_history.append({
                        "type": msg.__class__.__name__,
                        "content": msg.content
                    })
                elif isinstance(msg, dict):
                    serializable_history.append(msg)
                else:
                    serializable_history.append(str(msg))
        
        serializable_session["history"] = serializable_history
        
        return json.dumps(serializable_session)
    
    # Tool implementations
    
    def store_user_data(self, input_str: str) -> str:
        """
        Store user data in the session
        
        Args:
            input_str: JSON string with data to store
            
        Returns:
            Confirmation message
        """
        try:
            data = json.loads(input_str)
            session_id = self._current_session_id
            
            if not session_id or session_id not in self.sessions:
                return "Session ID not found or invalid"
            
            # Check if user_id is present in the data
            if 'user_id' in data or 'userId' in data:
                user_id = data.get('user_id') or data.get('userId')
                # Also store in session data for completeness
                self.sessions[session_id]["data"]["userId"] = user_id
            
                
            # Update session data
            self.sessions[session_id]["data"].update(data)
            logger.info(f"Session data updated: {self.sessions[session_id]['data']}")
            
            # Save updated session to database
            self.save_session_to_db(session_id)
            
            return f"Data successfully stored in session {session_id}"
        except Exception as e:
            logger.error(f"Error storing user data: {e}")
            return f"Error storing data: {str(e)}"
        
    def get_user_id_from_phone_number(self, phone_number: str) -> str:
        """
        Get user ID from phone number
        
        Args:
            phone_number: User's phone number
            
        Returns:
            API response as JSON string with userId
        """
        try:
            result = self.api_client.get_user_id_from_phone_number(phone_number)
            logger.info(f"API response from get_user_id_from_phone_number: {result}")
            
            # If successful, extract userId and store in session
            if result.get("status") == 200 and hasattr(self, '_current_session_id'):
                session_id = self._current_session_id
                # print(f"Session ID: {session_id}") # Debug print, can be removed
                if session_id in self.sessions:
                    # According to the problem description, result.get("data") is the userId string.
                    user_id_from_api = result.get("data")
                    
                    # Ensure extracted_user_id is a non-empty string
                    if isinstance(user_id_from_api, str) and user_id_from_api:
                        # Store userId in session.data.userId as per instruction
                        self.sessions[session_id]["data"]["userId"] = user_id_from_api
                        logger.info(f"Stored userId '{user_id_from_api}' in session data for session {session_id}")
                        
                        # The original code also updated self.sessions[session_id]["user_id"].
                        # Maintaining this for consistency, assuming it might be used elsewhere.
                        self.sessions[session_id]["user_id"] = user_id_from_api
                        logger.info(f"Also updated user_id '{user_id_from_api}' at session root for session {session_id}")
                        
                        # Save updated session to database
                        self.save_session_to_db(session_id)
                    else:
                        logger.warning(
                            f"UserId not found or is not a string in API response's 'data' field. "
                            f"Received: '{user_id_from_api}' (type: {type(user_id_from_api).__name__}) "
                            f"for session {session_id}."
                        )
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error getting user ID from phone number: {e}")
            return f"Error getting user ID from phone number: {str(e)}"
    
    def get_prefill_data(self, user_id: str = None) -> str:
        """
        Get prefilled user data
        
        Args:
            user_id: User identifier, optional if available in session
            
        Returns:
            Prefilled data as JSON string
        """
        try:
            # If user_id is not provided, try to get from session
            if not user_id and hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session and session.get("user_id"):
                    user_id = session.get("user_id")
                    
            if not user_id:
                return "User ID is required to get prefill data"
                
            result = self.api_client.get_prefill_data(user_id)
            
            # If successful, store important prefill data in session
            if result.get("status") == 200 and hasattr(self, '_current_session_id'):
                session_id = self._current_session_id
                try:
                    # Store the full API response for use by other methods like process_address_data
                    if "data" in result and "response" in result["data"]:
                        self.sessions[session_id]["data"]["prefill_api_response"] = result["data"]["response"]
                        logger.info(f"Stored full prefill API response in session for user ID: {user_id}")
                    
                    response_data = result.get("data", {}).get("response", {})
                    prefill_data = {}
                    prefill_data["userId"] = user_id
                    
                    # Extract and store key fields from prefill data
                    if "pan" in response_data:
                        prefill_data["panCard"] = response_data["pan"]
                    
                    if "gender" in response_data:
                        prefill_data["gender"] = response_data["gender"]
                    
                    if "dob" in response_data:
                        prefill_data["dateOfBirth"] = response_data["dob"]
                    
                    if "name" in response_data and isinstance(response_data["name"], dict):
                        name_data = response_data["name"]
                        if "firstName" in name_data:
                            prefill_data["firstName"] = name_data["firstName"]
                    
                    # Handle email if present
                    if "email" in response_data and response_data["email"]:
                        emails = response_data["email"]
                        if emails and isinstance(emails, list) and len(emails) > 0:
                            first_email = emails[0]
                            if isinstance(first_email, dict) and "email" in first_email:
                                prefill_data["emailId"] = first_email["email"]
                    
                    # Store in session data
                    if prefill_data:
                        self.sessions[session_id]["data"]["prefill_data"] = prefill_data
                        logger.info(f"Stored prefill data in session: {prefill_data}")
                        
                        # Save updated session to database
                        self.save_session_to_db(session_id)
                except Exception as e:
                    logger.warning(f"Error processing prefill data: {e}")
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error getting prefill data: {e}")
            return f"Error getting prefill data: {str(e)}"
        
    def save_address_details(self, input_str: str) -> str:
        """
        Save address details
        
        Args:
            input_str: JSON string with address data or user ID string
        """
        try:
            data = json.loads(input_str)
            user_id = data.get("userId")
            address = data.get("address")
            
            if not user_id:
                return "User ID is required"
                
            result = self.api_client.save_address_details(user_id, address)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error saving address details: {e}")
            return f"Error saving address details: {str(e)}"
            

    
    
    def get_employment_verification(self, user_id: str = None) -> str:
        """
        Get employment verification data
        
        Args:
            user_id: User identifier, optional if available in session
            
        Returns:
            Employment verification data as JSON string
        """
        try:
            # If user_id is not provided, try to get from session
            if not user_id and hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session and session.get("user_id"):
                    user_id = session.get("user_id")
                    
            if not user_id:
                return "User ID is required to get employment verification"
                
            result = self.api_client.get_employment_verification(user_id)
            
            # If successful, store important employment data in session
            if result.get("status") == 200 and hasattr(self, '_current_session_id'):
                session_id = self._current_session_id
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
                    
                    # Store in session
                    if employment_data:
                        self.sessions[session_id]["data"]["employment_data"] = employment_data
                        logger.info(f"Stored employment data in session: {employment_data}")
                        
                        # Save updated session to database
                        self.save_session_to_db(session_id)
                except Exception as e:
                    logger.warning(f"Error processing employment data: {e}")
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error getting employment verification: {e}")
            return f"Error getting employment verification: {str(e)}"
    
    def save_basic_details(self, input_str: str) -> str:
        """
        Save basic user details
        
        Args:
            input_str: JSON string with user details or user ID string
            
        Returns:
            Save result as JSON string
        """
        try:
            # Check if input_str is just a user ID (not JSON)
            if input_str and input_str.strip() and not input_str.strip().startswith('{'):
                user_id = input_str.strip()
                data = {}
            else:
                # Try to parse as JSON
                data = json.loads(input_str)
                user_id = data.pop("userId", None) or data.pop("user_id", None)
                
                # Extract fullName and phoneNumber from input
                if "fullName" in data:
                    data["firstName"] = data.pop("fullName")
                if "phoneNumber" in data:
                    data["mobileNumber"] = data.pop("phoneNumber")

            
            # Ensure we have a valid user ID
            if not user_id:
                # Try to get user ID from session if not provided in input
                if hasattr(self, '_current_session_id'):
                    session = self.sessions.get(self._current_session_id)
                    if session and session.get("user_id"):
                        user_id = session.get("user_id")
                
            if not user_id:
                return "User ID is required"
                
            result = self.api_client.save_basic_details(user_id, data)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error saving basic details: {e}")
            return f"Error saving basic details: {str(e)}"
    def save_employment_details(self, input_str: str) -> str:
        """
        Save employment details
        
        Args:
            input_str: JSON string with employment data or user ID string
            
        Returns:
            Save result as JSON string
        """
        try:
            # Check if input_str is just a user ID (not JSON)
            if input_str and input_str.strip() and not input_str.strip().startswith('{'):
                user_id = input_str.strip()
                data = {}
            else:
                # Try to parse as JSON
                data = json.loads(input_str)
                user_id = data.pop("userId", None) or data.pop("user_id", None)
            
            # Try to get user ID from session if not provided in input
            if not user_id and hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session and session.get("user_id"):
                    user_id = session.get("user_id")
            
            if not user_id:
                return "User ID is required"
            
            # Ensure proper income fields
            if 'monthlyIncome' in data and 'netTakeHomeSalary' not in data:
                data['netTakeHomeSalary'] = data['monthlyIncome']
                
            if 'monthlyIncome' in data and 'monthlyFamilyIncome' not in data:
                data['monthlyFamilyIncome'] = data['monthlyIncome']
                
            # Get monthly income from session data if not in the input
            if ('netTakeHomeSalary' not in data or 'monthlyFamilyIncome' not in data) and hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session and 'data' in session:
                    session_data = session['data']
                    if 'monthlyIncome' in session_data:
                        income = session_data['monthlyIncome']
                        if 'netTakeHomeSalary' not in data:
                            data['netTakeHomeSalary'] = income
                        if 'monthlyFamilyIncome' not in data:
                            data['monthlyFamilyIncome'] = income
            
            # Make sure we have the form status
            if 'formStatus' not in data:
                data['formStatus'] = "Employment"
                
            result = self.api_client.save_employment_details(user_id, data)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error saving employment details: {e}")
            return f"Error saving employment details: {str(e)}"
    
    def save_loan_details(self, input_str: str) -> str:
        """
        Save loan details
        
        Args:
            input_str: JSON string with loan data
            
        Returns:
            Save result as JSON string
        """
        try:
            data = json.loads(input_str)
            user_id = data.get("userId")
            name = data.get("fullName")
            loan_amount = data.get("treatmentCost")
            
            # Get doctor details from session if available
            doctor_id = None
            doctor_name = None
            
            if hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session and "data" in session:
                    doctor_id = session["data"].get("doctor_id")
                    doctor_name = session["data"].get("doctor_name")
                    logger.info(f"Retrieved doctor_id {doctor_id} and doctor_name {doctor_name} from session for loan details")
            
            if not user_id or not name or not loan_amount:
                return "User ID, name, and loan amount are required"
                
            result = self.api_client.save_loan_details(user_id, name, loan_amount, doctor_name, doctor_id)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error saving loan details: {e}")
            return f"Error saving loan details: {str(e)}"
    
    def get_loan_details(self, user_id: str = None) -> str:
        """
        Get loan details by user ID
        
        Args:
            user_id: User identifier, optional if available in session
            
        Returns:
            Loan details as JSON string
        """
        try:
            # If user_id is not provided, try to get from session
            if not user_id and hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session and session.get("user_id"):
                    user_id = session.get("user_id")
                    
            if not user_id:
                return "User ID is required to get loan details"
                
            result = self.api_client.get_loan_details_by_user_id(user_id)
            
            # If successful, extract loanId and store in session for future use
            if result.get("status") == 200 and hasattr(self, '_current_session_id'):
                session_id = self._current_session_id
                data = result.get("data", {})
                if isinstance(data, dict) and "loanId" in data:
                    loan_id = data["loanId"]
                    self.sessions[session_id]["data"]["loanId"] = loan_id
                    logger.info(f"Stored loanId {loan_id} in session {session_id}")
                    
                    # Save updated session to database
                    self.save_session_to_db(session_id)
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error getting loan details: {e}")
            return f"Error getting loan details: {str(e)}"
    
    def get_bureau_report(self, loan_id: str = None) -> str:
        """
        Get bureau report for a loan
        
        Args:
            loan_id: Loan identifier, optional if available in session
            
        Returns:
            Bureau report status as JSON string
        """
        try:
            # If loan_id is not provided, try to get from session
            if not loan_id and hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session and session.get("data", {}).get("loanId"):
                    loan_id = session["data"]["loanId"]
                    
            if not loan_id:
                return "Loan ID is required to get bureau report"
            
            logger.info(f"Requesting bureau report for loan ID: {loan_id}")    
            result = self.api_client.get_experian_bureau_report(loan_id)
            
            # Create a truncated version of the result with only essential information
            truncated_result = {
                "status": result.get("status"),
                "message": "Bureau report retrieved successfully" if result.get("status") == 200 else "Error retrieving bureau report"
            }
            
            # Store the full bureau report in the session for reference, but don't return it
            if hasattr(self, '_current_session_id') and self._current_session_id in self.sessions:
                self.sessions[self._current_session_id]["data"]["bureau_report_status"] = truncated_result
                # Also store if we successfully retrieved the report (for use in later steps)
                self.sessions[self._current_session_id]["data"]["bureau_report_retrieved"] = (result.get("status") == 200)
                logger.info(f"Stored bureau report status in session for loan ID: {loan_id}")
                
                # Save updated session to database
                self.save_session_to_db(self._current_session_id)
            
            # Enhanced logging to ensure visibility
            logger.info(f"Bureau Report Response - Status: {result.get('status')}")
            
            # Log some additional info but don't include in response
            if result.get("status") == 200:
                logger.info("Bureau report retrieved successfully")
                # Log some key parts from the data if present (truncated for readability)
                if "data" in result:
                    data_keys = list(result["data"].keys()) if isinstance(result["data"], dict) else "Not a dictionary"
                    logger.info(f"Bureau report data keys: {data_keys}")
            else:
                # Log error details
                logger.error(f"Bureau report error: {result.get('error', 'Unknown error')}")
                
            # Return only the truncated result to avoid token limit issues
            return json.dumps(truncated_result)
        except Exception as e:
            logger.error(f"Error getting bureau report: {e}")
            return json.dumps({
                "status": 500,
                "message": f"Error getting bureau report: {str(e)}"
            })
    
    def get_bureau_decision(self, input_str: str) -> str:
        """
        Get bureau decision for a loan
        
        Args:
            input_str: JSON string with loan_id 
            
        Returns:
            Bureau decision as JSON string
        """
        try:
            # Check if input_str is just a loan ID (not JSON)
            if input_str and input_str.strip() and not input_str.strip().startswith('{'):
                loan_id = input_str.strip()
                regenerate_param = 1
            else:
                # Try to parse as JSON
                data = json.loads(input_str)
                loan_id = data.get("loan_id") or data.get("loanId")
                regenerate_param = data.get("regenerate_param", 1) or data.get("regenerateParam", 1)
            
            # If loan_id is not provided, try to get from session
            if not loan_id and hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session and "data" in session and "loanId" in session["data"]:
                    loan_id = session["data"]["loanId"]
            
            # Get doctor_id from session if available, otherwise use default
            doctor_id = "e71779851b144d1d9a25a538a03612fc"  # Default doctor ID as fallback
            if hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session and "data" in session:
                    doctor_id = session["data"].get("doctor_id") or session["data"].get("doctorId", doctor_id)
                    logger.info(f"Using doctor_id {doctor_id} from session for bureau decision")
            
            if not loan_id:
                return "Loan ID is required"
                
            result = self.api_client.get_bureau_decision(doctor_id, loan_id, regenerate_param)
            
            # Log the raw API response for debugging
            logger.info(f"Bureau decision API response for loan ID {loan_id}: {json.dumps(result)}")
            
            
            # Check if the response contains the special INCOME_VERIFICATION_REQUIRED status
            if (isinstance(result, dict) and result.get("status") == 200 and 
                isinstance(result.get("data"), dict)):
                
                data = result.get("data", {})
                status = data.get("status") or data.get("bureauDecision")
                
                if status and "INCOME" in status.upper() and "VERIFICATION" in status.upper():
                    logger.info(f"INCOME VERIFICATION status detected in bureau decision: {status}")
            
            # Process result to extract and format eligible EMI information
            if isinstance(result, dict) and result.get("status") == 200:
                bureau_result = self.extract_bureau_decision_details(result)
                # Store the result in session for easy reference
                if hasattr(self, '_current_session_id'):
                    session_id = self._current_session_id
                    if session_id in self.sessions:
                        self.sessions[session_id]["data"]["bureau_decision_details"] = bureau_result
                        logger.info(f"Stored bureau decision details in session: {bureau_result}")
                        
                        # Save updated session to database
                        self.save_session_to_db(session_id)
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error getting bureau decision: {e}")
            return f"Error getting bureau decision: {str(e)}"

    def extract_bureau_decision_details(self, bureau_result: Dict[str, Any]) -> Dict[str, Any]:
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
                "emiPlans": []
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
            
            # Extract EMI plans
            if "emiPlans" in data and isinstance(data["emiPlans"], list):
                details["emiPlans"] = data["emiPlans"]
                
                # If we have plans but no max eligible EMI, use the highest EMI
                if not details["maxEligibleEMI"] and details["emiPlans"]:
                    try:
                        highest_emi = max(
                            (float(plan.get("emi", 0)) for plan in details["emiPlans"] if plan.get("emi")),
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
            
            # Log the complete details dictionary for debugging
            logger.info(f"Extracted bureau decision details: {details}")
            
            return details
        except Exception as e:
            logger.error(f"Error extracting bureau decision details: {e}")
            return {
                "status": None,
                "reason": None,
                "maxEligibleEMI": None,
                "emiPlans": []
            }

    def process_prefill_data_for_basic_details(self, input_data, user_id=None):
        """
        Process prefill data and return a properly formatted JSON string for save_basic_details
        
        Args:
            input_data: Dictionary with prefill data and userId, or just the prefill data
            user_id: Optional user_id parameter that takes precedence if provided
            
        Returns:
            JSON string for save_basic_details
        """
        try:
            prefill_data = {}
            
            # Extract user_id and prefill data from input
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    # Not valid JSON, could be just a user ID
                    if not user_id:  # Only use input as user_id if not explicitly provided
                        user_id = input_data.strip()
            
            # Extract user_id from different possible locations in the input
            if isinstance(input_data, dict):
                # Check if userId is directly in the input
                if not user_id:  # Only get from input if not explicitly provided
                    user_id = input_data.get("userId") or input_data.get("user_id")
                
                # Check if prefill_data field exists and extract data from it
                if "prefill_data" in input_data and isinstance(input_data["prefill_data"], dict):
                    prefill_data = input_data["prefill_data"]
                    # Also check for userId in prefill_data
                    if not user_id:
                        user_id = prefill_data.get("userId") or prefill_data.get("user_id")
                elif "prefillData" in input_data and isinstance(input_data["prefillData"], dict):
                    prefill_data = input_data["prefillData"]
                    # Also check for userId in prefillData
                    if not user_id:
                        user_id = prefill_data.get("userId") or prefill_data.get("user_id")
                else:
                    # If no prefill_data field, the entire input might be the prefill data
                    prefill_data = input_data
                    
            # If user_id still not found, try to get it from session
            if not user_id and hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session:
                    # Try different places where userId might be stored
                    user_id = session.get("user_id")
                    if not user_id and "data" in session:
                        session_data = session["data"]
                        user_id = session_data.get("userId") or session_data.get("user_id")
                        
                        # If still not found, look in nested objects
                        if not user_id and "prefill_data" in session_data:
                            user_id = session_data["prefill_data"].get("userId")
            
            if not user_id:
                return json.dumps({"error": "User ID is required to process prefill data"})
            
            # Now build the data for save_basic_details
            data = {"userId": user_id}
            
            # Set formStatus for the API call
            data["formStatus"] = "Basic"
            
            # Get session data to retrieve name and phone number if available
            session_data = {}
            if hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
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
            
            # Extract fields from prefill_data
            field_mappings = {
                "panCard": ["panCard", "pan"],
                "gender": ["gender"],
                "dateOfBirth": ["dateOfBirth", "dob"],
                "emailId": ["emailId", "email"],
                "firstName": ["firstName", "name"]
            }
            
            for target_field, source_fields in field_mappings.items():
                for source in source_fields:
                    if source in prefill_data and prefill_data[source]:
                        data[target_field] = prefill_data[source]
                        break
            
            # Handle special case for email which might be in different formats
            if "email" in prefill_data and prefill_data["email"] and "emailId" not in data:
                email_data = prefill_data["email"]
                if isinstance(email_data, list) and email_data:
                    if isinstance(email_data[0], dict) and "email" in email_data[0]:
                        data["emailId"] = email_data[0]["email"]
                    else:
                        data["emailId"] = email_data[0]
                else:
                    data["emailId"] = email_data
                    
            # Also extract from response if it exists
            if "response" in prefill_data and isinstance(prefill_data["response"], dict):
                response = prefill_data["response"]
                
                # Handle standard fields
                if "pan" in response and response["pan"] and "panCard" not in data:
                    data["panCard"] = response["pan"]
                if "gender" in response and response["gender"] and "gender" not in data:
                    data["gender"] = response["gender"]
                if "dob" in response and response["dob"] and "dateOfBirth" not in data:
                    data["dateOfBirth"] = response["dob"]
                    
                # Handle name which might be in different formats
                if "name" in response and "firstName" not in data:
                    name_data = response["name"]
                    if isinstance(name_data, dict) and "firstName" in name_data:
                        data["firstName"] = name_data["firstName"]
                    elif isinstance(name_data, str):
                        data["firstName"] = name_data
                
                # Handle email in response
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
            
            # Call the API client to save basic details
            result = self.api_client.save_basic_details(user_id, data)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error processing prefill data: {e}")
            if 'user_id' in locals() and user_id:
                return json.dumps({"userId": user_id, "error": str(e)})
            else:
                return json.dumps({"error": str(e)})

    def process_address_data(self, input_str: str) -> str:
        """
        Extract address information from prefill data and save it using save_address_details.
        Looks for 'Primary' address type and extracts postal code (pincode) and address line.
        
        Args:
            input_str: JSON string with userId or userId and prefill_data
            
        Returns:
            Save result as JSON string
        """
        try:
            # Parse input
            if isinstance(input_str, str):
                if input_str.strip().startswith('{'):
                    # Input is JSON
                    data = json.loads(input_str)
                else:
                    # Input is just userId
                    user_id = input_str.strip()
                    data = {'userId': user_id}
            else:
                data = input_str
                
            # Extract userId
            user_id = data.get('userId')
            if not user_id:
                return json.dumps({"error": "User ID is required"})
            
            logger.info(f"Processing address data for user ID: {user_id}")
            
            # Get prefill data from session if available
            prefill_data = None
            if hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                if session and "data" in session:
                    session_data = session["data"]
                    # Check if the prefill API response is stored in the session
                    if "prefill_api_response" in session_data:
                        prefill_data = session_data["prefill_api_response"]
            
            # If prefill_data not in session, try to get it from the input
            if not prefill_data and 'prefill_data' in data:
                prefill_data = data['prefill_data']
            elif not prefill_data and 'prefillData' in data:
                prefill_data = data['prefillData']
            
            # If still no prefill data, get it from the API
            if not prefill_data:
                logger.info(f"No prefill data found in session or input, getting from API for user ID: {user_id}")
                api_result = self.api_client.get_prefill_data(user_id)
                if api_result.get("status") == 200 and "data" in api_result and "response" in api_result["data"]:
                    prefill_data = api_result["data"]["response"]
                    
                    # Store in session for future use
                    if hasattr(self, '_current_session_id'):
                        session_id = self._current_session_id
                        if session_id in self.sessions:
                            self.sessions[session_id]["data"]["prefill_api_response"] = prefill_data
            
            if not prefill_data:
                return json.dumps({"error": "No prefill data available to extract address"})
            
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
                    
                    logger.info(f"Extracted address data: {address_data}")
                    
                    # Save the address details
                    result = self.api_client.save_address_details(user_id, address_data)
                    
                    return json.dumps(result)
                else:
                    return json.dumps({"error": "No address found in prefill data"})
            else:
                return json.dumps({"error": "Prefill data doesn't contain address information in expected format"})
                
        except Exception as e:
            logger.error(f"Error processing address data: {e}")
            return json.dumps({
                "error": f"Error processing address data: {str(e)}",
                "userId": user_id if 'user_id' in locals() else None
            })

    def save_session_to_db(self, session_id: str) -> None:
        """
        Save session data to the database
        
        Args:
            session_id: Session ID
        """
        try:
            if session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return
            
            session = self.sessions[session_id]
            
            # Convert session_id to UUID if it's a string
            if isinstance(session_id, str):
                session_id = uuid.UUID(session_id)
            
            # Use pre-serialized history if available, otherwise convert
            if 'serializable_history' in session:
                serializable_history = session['serializable_history']
            else:
                serializable_history = []
                if 'history' in session:
                    for msg in session['history']:
                        if hasattr(msg, 'content'):  # Check if it's a Message object (AIMessage or HumanMessage)
                            msg_type = msg.__class__.__name__
                            serializable_history.append({
                                'type': msg_type,
                                'content': msg.content
                            })
                        elif isinstance(msg, dict):
                            serializable_history.append(msg)
                        else:
                            serializable_history.append(str(msg))
                            
                # Store the serialized history for future use
                session['serializable_history'] = serializable_history
            
            # Save to database as JSON-serializable format
            session_data, created = SessionData.objects.update_or_create(
                session_id=session_id,
                defaults={
                    'data': session.get('data', {}),
                    'history': serializable_history,  # Use the serialized history
                    'status': session.get('status', 'active'),
                    'phone_number': session.get('phone_number'),
                }
            )
            
            logger.info(f"Session {session_id} saved to database (created={created})")
        except Exception as e:
            logger.error(f"Error saving session to database: {e}")
            # Log more detailed information for debugging
            if 'history' in session:
                history_types = [type(msg).__name__ for msg in session['history']]
                logger.error(f"History contains types: {history_types}")

    def save_additional_user_details(self, input_str: str) -> str:
        """
        Save additional user details collected after bureau decision
        
        Args:
            input_str: JSON string with additional user details
            
        Returns:
            Confirmation message
        """
        try:
            data = json.loads(input_str)
            session_id = self._current_session_id
            
            if not session_id or session_id not in self.sessions:
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
            if 'additional_details' not in self.sessions[session_id]["data"]:
                self.sessions[session_id]["data"]["additional_details"] = {}
            
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
            elif employment_type == "SELF-EMPLOYED" and business_name:
                additional_details["business_name"] = business_name
                
            # Update session data
            self.sessions[session_id]["data"]["additional_details"].update(additional_details)
            
            # Save updated session to database
            self.save_session_to_db(session_id)
            
            # Try to get user ID from session
            user_id = self.sessions[session_id]["data"].get("userId")
            # doctor_id = self.sessions[session_id]["data"].get("doctorId")
            # doctor_name = self.sessions[session_id]["data"].get("doctorName")
            print(f"User ID: {user_id}")
            
            # If we have a user ID, send employment details to API
            if user_id:
                employment_data = self._process_employment_data_from_additional_details(session_id)
                if employment_data:
                    try:
                        result = self.api_client.save_employment_details(user_id, employment_data)
                        print(f"Successfully saved employment details for user {user_id}: {result}")
                        logger.info(f"Successfully saved employment details for user {user_id}: {employment_data}")
                    except Exception as e:
                        logger.error(f"Error saving employment details for user {user_id}: {e}")

            if user_id:
                loan_data = self._process_loan_data_from_additional_details(session_id)
                if loan_data:
                    try:
                        # Convert loan_data to JSON string for save_loan_details
                        result = self.api_client.save_loan_details_again(user_id, loan_data)
                        print(f"Successfully saved loan details for user {user_id}: {result}")
                        logger.info(f"Successfully saved loan details for user {user_id}: {result}")

                        # logger.info(f"Successfully saved loan details for user {user_id}: {loan_data}")
                    except Exception as e:
                        logger.error(f"Error saving loan details for user {user_id}: {e}")

            if user_id:
                data = self._process_basic_details_from_additional_details(session_id)
                if data:
                    try:
                        result = self.api_client.save_basic_details(user_id, data)
                        print(f"Successfully saved basic details for user {user_id}: {result}")
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
        try:
            session = self.sessions[session_id]
            
            # Ensure additional_details exists in session data
            if "additional_details" not in session["data"]:
                session["data"]["additional_details"] = {}
                
            additional_details = session["data"]["additional_details"]
            
            # Keep track of the current collection step
            # This is stored in session data to remember where we are in the collection flow
            collection_step = session["data"].get("collection_step", "employment_type")
            
            # Function to save the current collection step
            def update_collection_step(new_step):
                session["data"]["collection_step"] = new_step
                self.save_session_to_db(session_id)
            
            # Handle employment type input (first step)
            if collection_step == "employment_type":
                if "1" in message:
                    additional_details["employment_type"] = "SALARIED"
                elif "2" in message:
                    additional_details["employment_type"] = "SELF-EMPLOYED"
                else:
                    return "Please select a valid option for Employment Type: 1. SALARIED or 2. SELF-EMPLOYED"
                
                # Update session data with employment type
                session["data"]["additional_details"] = additional_details
                
                # Update collection step and ask for marital status
                update_collection_step("marital_status")
                return """What is the Marital Status of the patient?
1. Married
2. Unmarried/Single"""
            
            # Handle marital status input
            elif collection_step == "marital_status":
                if "1" in message:
                    additional_details["marital_status"] = "1"
                elif "2" in message:
                    additional_details["marital_status"] = "2"
                else:
                    return "Please select a valid option for Marital Status: 1. Married or 2. Unmarried/Single"
                
                # Update session data with marital status
                session["data"]["additional_details"] = additional_details
                
                # Update collection step and ask for education qualification
                update_collection_step("education_qualification")
                return """What is the Education Qualification of the patient?
1. Less than 10th
2. Passed 10th
3. Passed 12th
4. Diploma
5. Graduation
6. Post graduation
7. P.H.D"""
            
            # Handle education qualification input
            elif collection_step == "education_qualification":
                if message.strip() in ["1", "2", "3", "4", "5", "6", "7"]:
                    additional_details["education_qualification"] = message.strip()
                else:
                    return "Please select a valid option for Education Qualification (1-7)"
                
                # Update session data with education qualification
                session["data"]["additional_details"] = additional_details
                
                # Update collection step and ask for treatment reason
                update_collection_step("treatment_reason")
                return "What is the reason for treatment?"
            
            # Handle treatment reason input
            elif collection_step == "treatment_reason":
                additional_details["treatment_reason"] = message.strip()
                
                # Update session data with treatment reason
                session["data"]["additional_details"] = additional_details
                
                # Update collection step and ask organization/business name based on employment type
                if additional_details.get("employment_type") == "SALARIED":
                    update_collection_step("organization_name")
                    return "What is the Organization Name where the patient works?"
                else:
                    update_collection_step("business_name")
                    return "What is the Business Name of the patient?"
            
            # Handle organization name input (for SALARIED)
            elif collection_step == "organization_name":
                additional_details["organization_name"] = message.strip()
                
                # Update session data
                session["data"]["additional_details"] = additional_details
                
                # Update collection step to ask for workplace pincode
                update_collection_step("workplace_pincode")
                return "Please enter the workplace pincode (6 digits):"
            
            # Handle business name input (for SELF-EMPLOYED)
            elif collection_step == "business_name":
                additional_details["business_name"] = message.strip()
                
                # Update session data
                session["data"]["additional_details"] = additional_details
                
                # Update collection step to ask for workplace pincode
                update_collection_step("workplace_pincode")
                return "Please enter the workplace pincode (6 digits):"

            # Handle workplace pincode input
            elif collection_step == "workplace_pincode":
                # Validate pincode (6 digit number)
                pincode = message.strip()
                if not pincode.isdigit() or len(pincode) != 6:
                    return "Please enter a valid 6-digit pincode."
                
                additional_details["workplacePincode"] = pincode
                
                # Update session data
                session["data"]["additional_details"] = additional_details
                
                # Mark collection as complete
                update_collection_step("complete")
                
                # Save all collected details using the tool
                # Make sure to create a new copy to avoid reference issues
                details_to_save = dict(additional_details)
                result = self.save_additional_user_details(json.dumps(details_to_save))
                
                # Change session status to completed but keep session active
                session["status"] = "additional_details_completed"
                session["data"]["details_collection_timestamp"] = datetime.now().isoformat()
                
                # Get profile link to show to the user
                profile_link = self._get_profile_link(session_id)
                self.save_session_to_db(session_id)
                
                return f"Thank you! Your application is now complete. Please check your application status by visiting the following link:\n\n{profile_link}\n\nYou can track your application progress"
            
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
            Profile completion link URL
        """
        try:
            if session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return "https://app.carepay.co.in"  # Fallback URL if session not found
                
            session = self.sessions[session_id]
            
            # Get doctor ID from session
            doctor_id = session["data"].get("doctorId") or session["data"].get("doctor_id")
            
            # # If doctor_id not found, use a default
            # if not doctor_id:
            #     doctor_id = "e71779851b144d1d9a25a538a03612fc"  # Default doctor ID as fallback
            #     logger.warning(f"Using default doctor_id for profile link: {doctor_id}")
                
            # Call API to get profile completion link
            profile_link_response = self.api_client.get_profile_completion_link(doctor_id)
            logger.info(f"Profile completion link response: {json.dumps(profile_link_response)}")
            
            # Extract link from response
            if isinstance(profile_link_response, dict) and profile_link_response.get("status") == 200:
                profile_link = profile_link_response.get("data", "")
                
                # Store the link in session for future reference
                session["data"]["profile_completion_link"] = profile_link
                
               
                return profile_link
                 # Fallback URL
            else:
                logger.error(f"Error getting profile link: {profile_link_response}")
                return "https://app.carepay.co.in"  # Fallback URL if any error occurs
                
        except Exception as e:
            logger.error(f"Error getting profile completion link: {e}")
            return "https://app.carepay.co.in"  # Fallback URL if any error occurs

    def _process_employment_data_from_additional_details(self, session_id: str) -> Dict[str, Any]:
        """
        Process employment data from additional details collected
        
        Args:
            session_id: Session identifier
            
        Returns:
            Employment data dict ready for API
        """
        try:
            session = self.sessions[session_id]
            additional_details = session["data"].get("additional_details", {})
            
            # Create employment data structure
            employment_data = {}

            if "monthlyIncome" in session["data"]:
                employment_data["monthlyFamilyIncome"] = session["data"]["monthlyIncome"]
            elif "monthlyFamilyIncome" in session["data"]:
                employment_data["monthlyFamilyIncome"] = session["data"]["monthlyFamilyIncome"]

            if "monthlyIncome" in session["data"]:
                employment_data["netTakeHomeSalary"] = session["data"]["monthlyIncome"]
            
            # Map employment type
            if additional_details.get("employment_type"):
                employment_data["employmentType"] = additional_details["employment_type"]
            
            # Map organization or business name
            if employment_data["employmentType"] == "SALARIED" and additional_details.get("organization_name"):
                employment_data["organizationName"] = additional_details["organization_name"]
            elif employment_data["employmentType"] == "SELF-EMPLOYED" and additional_details.get("business_name"):
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
            session = self.sessions[session_id]
            additional_details = session["data"].get("additional_details", {})
            user_id = session["data"].get("userId") or session["data"].get("user_id")
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
            session = self.sessions[session_id]
            additional_details = session["data"].get("additional_details", {})  
            user_id = session["data"].get("userId") or session["data"].get("user_id")
         
            # Create basic details structure with userId
            basic_details = {
                "userId": user_id
            }

            if "name" in session["data"]:
                basic_details["fullName"] = session["data"]["name"]
            elif "fullName" in session["data"]:
                basic_details["fullName"] = session["data"]["fullName"]
            
            if "phoneNumber" in session["data"]:
                basic_details["mobileNumber"] = session["data"]["phoneNumber"]
            
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
