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

        10. Decision Communication:
           - didn't miss this step
           - Format your loan decision response based on the status received from get_bureau_decision:
           
           - For APPROVED status ONLY:
             ```
             ### Loan Application Decision:
             
             🎉 Congratulations, [PATIENT_NAME]! Your loan application has been **APPROVED**.
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
             
             Your application requires **INCOME VERIFICATION**. Please submit additional income documents to proceed.
             ```
             
           - For any other status:
             ```
             ### Loan Application Decision:
             
             Your loan application status is: **[BUREAU_DECISION]**.
             ```
             
           - IMPORTANT: NEVER display "APPROVED" status for any application that has status INCOME_VERIFICATION_REQUIRED or any other status that is not explicitly "APPROVED".
           - IMPORTANT: Strictly use the exact status value received from the get_bureau_decision API response.
           - IMPORTANT: Keep your response simple and clean. Do NOT include any EMI plans, tenure details, or payment schedules in your response.

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
            handle_parsing_errors=True,
        )

        return agent_executor
    
    def create_session(self) -> str:
        """
        Create a new chat session
        
        Returns:
            Session ID
        """
        try:
            # Generate a unique session ID
            session_id = str(uuid.uuid4())
            
            # Create session with initial data
            session = {
                "id": session_id,
                "created_at": datetime.now().isoformat(),
                "status": "initial",  
                "history": [],  
                "data": {},  
                "user_id": None  
            }
            
            # Store session
            self.sessions[session_id] = session
            logger.info(f"Created new session: {session_id}")
            
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
                return "Session expired or not found. Please refresh the page to start a new session."
            
            # Set current session for helper methods
            self._current_session_id = session_id
            
            # Get session data
            session = self.sessions[session_id]
            current_status = session.get("status", "initial")
            phone = session.get("data", {}).get("phone", None)
            
            logger.info(f"Processing message in session {session_id} with status {current_status} and phone {phone}")
            
            # Add user message to history
            chat_history = session.get("history", [])
            
            chat_history.append(HumanMessage(content=message))
            
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
            
            # Check if this is a bureau decision response based on content patterns
            if "bureau decision" in ai_message.lower() and any(plan in ai_message for plan in ["Plan:", "Plan", "EMI:", "/0", "/5", "/1"]):
                try:
                    # Extract EMI plan information
                    emi_plans = []
                    bureau_decision = None
                    decision_reason = None
                    max_eligible_emi = None
                    
                    # Extract bureau decision status if present
                    bureau_match = re.search(r"Bureau Decision:\s*(\w+(?:_\w+)*)", ai_message)
                    if bureau_match:
                        bureau_decision = bureau_match.group(1)
                    
                    # Extract decision reason if present
                    reason_match = re.search(r"Decision Reason:\s*([^•\n]+)", ai_message)
                    if reason_match:
                        decision_reason = reason_match.group(1).strip()
                    
                    # Extract maximum eligible EMI with improved pattern matching
                    # Try multiple patterns for max eligible EMI
                    emi_patterns = [
                        r"Maximum Eligible EMI:\s*₹?([\d,]+(?:\.\d+)?)",
                        r"Max[.]*\s*Eligible EMI:\s*₹?([\d,]+(?:\.\d+)?)",
                        r"Maximum EMI Eligible:\s*₹?([\d,]+(?:\.\d+)?)",
                        r"Eligible EMI:\s*₹?([\d,]+(?:\.\d+)?)",
                        r"Eligible EMI[^:]*:\s*₹?([\d,]+(?:\.\d+)?)",
                        r"EMI Eligibility:\s*₹?([\d,]+(?:\.\d+)?)"
                    ]
                    
                    for pattern in emi_patterns:
                        emi_match = re.search(pattern, ai_message, re.IGNORECASE)
                        if emi_match:
                            max_eligible_emi = emi_match.group(1).replace(',', '')
                            break
                            
                    # If still not found, try to find any number after "eligible" and before "EMI"
                    if not max_eligible_emi:
                        general_emi_match = re.search(r"eligible[^₹\d]*[^\w]₹?([\d,]+(?:\.\d+)?)", ai_message, re.IGNORECASE)
                        if general_emi_match:
                            max_eligible_emi = general_emi_match.group(1).replace(',', '')
                    
                    # Extract all plan details using regex (for internal use, not display)
                    plan_pattern = r"(?:(\d+)/(\d+)(?:\s*([A-Z]+))?\s+Plan:?)\s*(?:Credit Limit:?\s*₹?([\d,]+(?:\.\d+)?))?(?:.*?EMI:?\s*₹?([\d,]+(?:\.\d+)?))?(?:.*?Down Payment:?\s*₹?([\d,]+(?:\.\d+)?))?|(?:(?:EMI:?\s*₹?([\d,]+(?:\.\d+)?))?\s*(?:Credit Limit:?\s*₹?([\d,]+(?:\.\d+)?))?(?:.*?Down Payment:?\s*₹?([\d,]+(?:\.\d+)?))?\s*(\d+)/(\d+)(?:\s*([A-Z]+))?\s+Plan:?)"
                    
                    plans = re.finditer(plan_pattern, ai_message)
                    for plan in plans:
                        groups = plan.groups()
                        
                        # Handle both formats (plan first or EMI first)
                        if groups[0]:  # If first capture group exists, it's in the first format
                            tenure = groups[0]
                            interest = groups[1] 
                            plan_type = groups[2] or ""
                            credit_limit = groups[3].replace(',', '') if groups[3] else None
                            emi = groups[4].replace(',', '') if groups[4] else None
                            down_payment = groups[5].replace(',', '') if groups[5] else None
                        else:  # Otherwise it's in the second format
                            emi = groups[6].replace(',', '') if groups[6] else None
                            credit_limit = groups[7].replace(',', '') if groups[7] else None
                            down_payment = groups[8].replace(',', '') if groups[8] else None
                            tenure = groups[9]
                            interest = groups[10]
                            plan_type = groups[11] or ""
                        
                        emi_plans.append({
                            "planName": f"{tenure}/{interest}{' ' + plan_type if plan_type else ''}",
                            "creditLimit": credit_limit,
                            "emi": emi,
                            "downPayment": down_payment or "0"
                        })
                    
                    # If no plans were found with regex, try alternative approach with line splitting
                    if not emi_plans:
                        for line in ai_message.split('\n'):
                            plan_match = re.search(r"(\d+)/(\d+)\s*([A-Z]*)\s*Plan", line)
                            if plan_match:
                                tenure, interest, plan_type = plan_match.groups()
                                plan_type = plan_type.strip()
                                
                                credit_match = re.search(r"Credit Limit:?\s*₹?([\d,]+(?:\.\d+)?)", line)
                                credit_limit = credit_match.group(1).replace(',', '') if credit_match else None
                                
                                emi_match = re.search(r"EMI:?\s*₹?([\d,]+(?:\.\d+)?)", line)
                                emi = emi_match.group(1).replace(',', '') if emi_match else None
                                
                                down_match = re.search(r"Down Payment:?\s*₹?([\d,]+(?:\.\d+)?)", line)
                                down_payment = down_match.group(1).replace(',', '') if down_match else None
                                
                                emi_plans.append({
                                    "planName": f"{tenure}/{interest}{' ' + plan_type if plan_type else ''}",
                                    "creditLimit": credit_limit,
                                    "emi": emi,
                                    "downPayment": down_payment or "0"
                                })
                    
                    # If we have EMI plans but no max eligible EMI, use the highest EMI from plans
                    if not max_eligible_emi and emi_plans:
                        highest_emi = max(
                            (float(plan["emi"]) for plan in emi_plans if plan["emi"] is not None),
                            default=None
                        )
                        if highest_emi:
                            max_eligible_emi = str(int(highest_emi))
                    
                    # Get user's name from session if available
                    user_name = "Customer"
                    if hasattr(self, '_current_session_id'):
                        session = self.sessions.get(self._current_session_id)
                        if session and "data" in session:
                            session_data = session["data"]
                            if "name" in session_data:
                                user_name = session_data["name"]
                            elif "fullName" in session_data:
                                user_name = session_data["fullName"]
                            # Try to get first name if full name is available
                            if " " in user_name:
                                user_name = user_name.split(" ")[0]
                    
                    # Create simplified decision message with clean format
                    decision_message = ""
                    if bureau_decision:
                        logger.info(f"Processing bureau decision for frontend display: {bureau_decision}")
                        if bureau_decision.upper() == "APPROVED":
                            decision_message = f"### Loan Application Decision:\n\n🎉 Congratulations, {user_name}! Your loan application has been **APPROVED**."
                        elif bureau_decision.upper() == "REJECTED":
                            reason_text = decision_reason if decision_reason else "Insufficient credit score or eligibility"
                            decision_message = f"### Loan Application Decision:\n\nWe regret to inform you that your loan application has been **REJECTED**.\nReason: {reason_text}"
                        elif "INCOME" in bureau_decision.upper() and "VERIFICATION" in bureau_decision.upper():
                            decision_message = f"### Loan Application Decision:\n\nYour application requires **INCOME VERIFICATION**. Please submit additional income documents to proceed."
                        else:
                            decision_message = f"### Loan Application Decision:\n\nYour loan application status is: **{bureau_decision}**."
                    
                    # Create formatted response - now with simpler structure for frontend
                    formatted_response = {
                        "bureauDecision": {
                            "status": bureau_decision,
                            "reason": decision_reason,
                            "maxEligibleEMI": max_eligible_emi,
                            "emiPlans": emi_plans,
                            "customerName": user_name,
                            "decisionMessage": decision_message
                        }
                    }
                    
                    # Check if we need to override the LLM's decision message based on internal knowledge
                    if hasattr(self, '_current_session_id'):
                        session_id = self._current_session_id
                        if session_id in self.sessions and "data" in self.sessions[session_id]:
                            session_data = self.sessions[session_id]["data"]
                            if "bureau_decision_details" in session_data:
                                actual_status = session_data["bureau_decision_details"].get("status")
                                if actual_status and bureau_decision:
                                    # If there's a mismatch, log it and use the stored status from API
                                    if actual_status != bureau_decision:
                                        logger.warning(f"Bureau decision mismatch: API status '{actual_status}' vs LLM extracted '{bureau_decision}'")
                                        
                                        # Override with the correct status and generate proper message
                                        bureau_decision = actual_status
                                        formatted_response["bureauDecision"]["status"] = bureau_decision
                                        
                                        # Generate the correct decision message based on the actual status
                                        if bureau_decision.upper() == "APPROVED":
                                            formatted_response["bureauDecision"]["decisionMessage"] = f"### Loan Application Decision:\n\n🎉 Congratulations, {user_name}! Your loan application has been **APPROVED**."
                                        elif bureau_decision.upper() == "REJECTED":
                                            reason_text = session_data["bureau_decision_details"].get("reason") or "Insufficient credit score or eligibility"
                                            formatted_response["bureauDecision"]["decisionMessage"] = f"### Loan Application Decision:\n\nWe regret to inform you that your loan application has been **REJECTED**.\nReason: {reason_text}"
                                        elif "INCOME" in bureau_decision.upper() and "VERIFICATION" in bureau_decision.upper():
                                            formatted_response["bureauDecision"]["decisionMessage"] = f"### Loan Application Decision:\n\nYour application requires **INCOME VERIFICATION**. Please submit additional income documents to proceed."
                                        else:
                                            formatted_response["bureauDecision"]["decisionMessage"] = f"### Loan Application Decision:\n\nYour loan application status is: **{bureau_decision}**."
                    
                    # Convert the formatted response to JSON and log it
                    formatted_json = json.dumps(formatted_response)
                    logger.info(f"Formatted bureau decision response: {formatted_json}")
                    
                    # Store the formatted response in the session
                    self.sessions[session_id]["data"]["bureau_decision"] = formatted_response["bureauDecision"]
                    
                    # Return the formatted response
                    return formatted_json
                    
                except Exception as e:
                    logger.error(f"Error formatting bureau decision: {e}")
                    # Fall back to standard response if formatting fails
            
            # Save updated history to session
            session["history"] = chat_history
            self.sessions[session_id] = session
            
            return ai_message
        
        except Exception as e:
            logger.error(f"Error in agent run: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

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
        return json.dumps(session)
    
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
                # Store user_id at the session level
                self.sessions[session_id]['user_id'] = user_id
            
            # Preserve doctor details if they exist in the session
            doctor_id = None
            doctor_name = None
            
            if "data" in self.sessions[session_id]:
                doctor_id = self.sessions[session_id]["data"].get("doctor_id")
                doctor_name = self.sessions[session_id]["data"].get("doctor_name")
                
            # Update session data
            self.sessions[session_id]["data"].update(data)
            
            # Restore doctor details if they were present
            if doctor_id and "doctor_id" not in data:
                self.sessions[session_id]["data"]["doctor_id"] = doctor_id
                logger.info(f"Preserved doctor_id {doctor_id} in session {session_id}")
                
            if doctor_name and "doctor_name" not in data:
                self.sessions[session_id]["data"]["doctor_name"] = doctor_name
                logger.info(f"Preserved doctor_name {doctor_name} in session {session_id}")
                
            logger.info(f"Session data updated: {self.sessions[session_id]['data']}")
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
                if session_id in self.sessions:
                    user_id = None
                    # Try to extract userId from the response
                    data = result.get("data", {})
                    if isinstance(data, dict) and "userId" in data: # ??
                        user_id = data["userId"]
                    
                    # Store userId in session if found
                    if user_id:
                        self.sessions[session_id]["user_id"] = user_id
                        logger.info(f"Stored user_id {user_id} in session {session_id}")
                        
                        # Also store in session data for completeness
                        self.sessions[session_id]["data"]["userId"] = user_id
            
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
                    
            # Use doctor details from input if provided
            if "doctorId" in data:
                doctor_id = data.get("doctorId")
            if "doctorName" in data:
                doctor_name = data.get("doctorName")
            
            if not user_id or not name or not loan_amount:
                return "User ID, name, and loan amount are required"
            
            if not doctor_id or not doctor_name:
                error_msg = "Doctor ID and doctor name are required for loan details. Please ensure they are available in the session."
                logger.error(error_msg)
                return json.dumps({"error": error_msg, "status": 400})
            
            # Update the API client's doctor details
            self.api_client.doctor_id = doctor_id
            self.api_client.doctor_name = doctor_name
            self.api_client.has_doctor_details = True
            logger.info(f"Using doctor details from session: doctor_id={doctor_id}, doctor_name={doctor_name}")
                
            result = self.api_client.save_loan_details(user_id, name, loan_amount)
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
                doctor_id = None
                regenerate_param = 0
            else:
                # Try to parse as JSON
                data = json.loads(input_str)
                loan_id = data.get("loan_id") or data.get("loanId")
                doctor_id = data.get("doctor_id") or data.get("doctorId")
                regenerate_param = data.get("regenerate_param", 0) or data.get("regenerateParam", 0)
            
            # If loan_id is not provided, try to get from session
            session = None
            if hasattr(self, '_current_session_id'):
                session = self.sessions.get(self._current_session_id)
                
            if not loan_id and session and "data" in session and "loanId" in session["data"]:
                loan_id = session["data"]["loanId"]
            
            # Get doctor_id from session if not provided
            if not doctor_id and session and "data" in session and "doctor_id" in session["data"]:
                doctor_id = session["data"]["doctor_id"]
            
            # Require doctor_id to be provided from session or input
            if not doctor_id:
                error_msg = "Doctor ID is required for bureau decision. Please ensure it is available in the session."
                logger.error(error_msg)
                return json.dumps({"error": error_msg, "status": 400})
            
            if not loan_id:
                return "Loan ID is required"
                
            result = self.api_client.get_bureau_decision(doctor_id, loan_id, regenerate_param)
            
            # Log the raw API response for debugging
            logger.info(f"Bureau decision API response for loan ID {loan_id} with doctor ID {doctor_id}: {json.dumps(result)}")
            
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