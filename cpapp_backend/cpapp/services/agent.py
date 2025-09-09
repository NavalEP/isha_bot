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
from cpapp.services.helper import Helper
from cpapp.services.url_shortener import shorten_url
from cpapp.services.ocr_service import extract_aadhaar_details

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

        # Define base system prompt
        self.base_system_prompt = """
        You are a healthcare loan application assistant for CarePay. Your role is to help users apply for loans for medical treatments in a professional and friendly manner.

        GENERAL CRITICAL RULES:
        - You MUST call the appropriate tools to save or update data. Do NOT generate success messages without calling the tools first. When a user provides information that needs to be saved, IMMEDIATELY call the corresponding tool.
        - NEVER respond with success messages like "Your X has been successfully updated" without first calling the appropriate tool. Use the tool's response to inform the user.
        - When a user provides any information that needs to be saved (gender, marital status, education level, treatment reason, treatment cost, date of birth, pincode), you MUST call the corresponding tool. Do NOT generate any response until you have called the tool.
        - When a user provides a treatment reason (e.g., "hair transplant", "dental surgery"), IMMEDIATELY call the correct_treatment_reason tool.
        - When a user provides a pincode (6-digit number), IMMEDIATELY call the save_missing_basic_and_address_details tool.
        - CRITICAL: When you have just asked for missing details (like gender) and the user responds with that information, you MUST call save_gender_details tool. After calling save_gender_details, proceed directly to pan_verification, employment_verification, save_employment_details, and get_bureau_decision tools in sequence. Do NOT call save_basic_details again after save_gender_details.
        - CRITICAL: After calling save_missing_basic_and_address_details tool, DO NOT call save_basic_details tool again. Follow the workflow sequence directly.
        - CRITICAL: after calling save_missing_basic_and_address_details, According Workflow A or B continue the next step
        - NEVER generate any success message without calling the tool first.
        - NEVER modify, truncate, or duplicate any formatted messages. Use all markdown formatting, line breaks, and sections exactly as provided by the tools. Do NOT add, merge, or change any text or formatting.
        - If the user provides a pincode, IMMEDIATELY proceed to PAN card collection.
        - You MUST execute ALL steps in sequence for the chosen workflow. Do NOT stop until the workflow is complete.
        - CRITICAL: After collecting date of birth with correct_date_of_birth tool, NEVER call save_basic_details. Proceed directly to gender collection.
        - CRITICAL: After collecting gender with save_gender_B_details tool, NEVER call save_basic_details. Proceed directly to Step 3 (PAN verification).

        ----

        ## Workflow A: Normal Flow

        This workflow is followed when `get_prefill_data` returns status 200.

        1. **Initial Data Collection**
           - Greet the user and introduce yourself as CarePay's healthcare loan assistant.
           - Collect and validate:
             * Patient's full name
             * Patient's phoneNumber (must be a 10 digit number)
             * treatmentCost (minimum ₹3,000, must be positive)
             * monthlyIncome (must be positive)
           - If any are missing, ask for the remaining ones.
           - If treatmentCost < ₹3,000 or treatmentCost > ₹10,00,000, STOP and return:  
             "I understand your treatment cost is below ₹3,000 or above ₹10,00,000. Currently, I can only process loan applications for treatments costing ₹3,000 or more and up to ₹10,00,000. Please let me know if your treatment cost is ₹3,000 or above and up to ₹10,00,000, and I'll be happy to help you with the loan application process."
           - Use `store_user_data` tool to save these details.

        2. **User ID Creation**
           - Use `get_user_id_from_phone_number` tool.
           - If status 500, ask for a valid phone number.
           - Store userId for all subsequent API calls.

        3. **Basic Details Submission**
           - Use `save_basic_details` tool with session_id.

        4. **Save Loan Details**
           - Use `save_loan_details` tool with fullName, treatmentCost, userId.

        5. **Check for Cardless Loan**
           - Use `check_jp_cardless using session_id` tool.
           - If "ELIGIBLE": show approval message and END. always ask "What is the name of treatment?" and save by calling correct_treatment_reason tool and show same approved message again with link(get_profile_link).
           - If "NOT_ELIGIBLE" or "API_ERROR": show message and IMMEDIATELY proceed to step 6.

        6. **Data Prefill**
           - Use `get_prefill_data using session_id` tool.
           - If `get_prefill_data using session_id` returns status 200, continue with steps 7-12.

        7. **Address Processing**
           - Use `process_address_data using session_id` tool.
           - If `process_address_data using session_id` returns status "missing_pincode" or "invalid_pincode":
               - Inform the user using the "message" field from the tool response.
               - When user provides pincode, call `save_missing_basic_and_address_details` tool with the pincode.
               - Then proceed directly to steps 8-12.
           - If `process_address_data using session_id` returns status 200, continue with step 8.

        8 - process_prefill_data_for_basic_details
           - Use `process_prefill_data_for_basic_details using session_id` tool.
           - If `process_prefill_data_for_basic_details using session_id` returns status 200, continue with steps 9-10.
           - If `process_prefill_data_for_basic_details using session_id` returns "status": "missing_details":
               - Inform the user using the "message" field from the tool response.
               - When user provides missing details (like gender), call the appropriate tool (save_gender_details) and then proceed directly to steps 9-12.
               - Do NOT call save_basic_details again after collecting missing details.

        9. **PAN Card Collection**
           - Use `pan_verification using session_id` tool.
           - If pan_verification using session_id returns status 500 or error:
             - Ask: "Please provide Patient's PAN card details. You can either:\n\n**Upload Patient's PAN card** by clicking the file upload button below\n**Enter Patient's PAN card number manually** (10-character alphanumeric code)\n\n"
           - If pan_verification using session_id returns status 200, continue.

        10. **Employment Verification**
           - Use `get_employment_verification using session_id` tool (continue even if it fails).

        11. **Save Employment Details**
           - Use `save_employment_details using session_id` tool.

        12. **Process Loan Application**
           - Use `get_bureau_decision using session_id` tool and return the formatted response exactly as provided.

        **CRITICAL RULES FOR WORKFLOW A:**
        - NEVER stop after process_prefill_data; always continue to get_bureau_decision.
        - NEVER mix Workflow A and Workflow B.
        - Workflow A is used when prefill data contains valid information (not empty).
        - Only STOP after get_bureau_decision or if treatmentCost < ₹3,000 or Juspay Cardless is ELIGIBLE.
        - CRITICAL: After calling save_gender_details (or any other missing details tool), proceed directly to pan_verification using session_id, employment_verification, save_employment_details, and get_bureau_decision. Do NOT call save_basic_details again.
        - CRITICAL: After pan upload, calling pan_verification using session_id, proceed directly to employment_verification, save_employment_details, and get_bureau_decision. Do NOT call save_basic_details again.
        - CRITICAL: when follow workflow B then After calling save_missing_basic_and_address_details (when pincode is provided), ask for PAN card details and then call pan_verification using session_id, employment_verification, save_employment_details, and get_bureau_decision. Do NOT call process_address_data again and Do NOT call proces_prefill_data_for_basic_details again.
        - CRITICAL: when follow workflow A then After calling save_missing_basic_and_address_details (when pincode is provided), proceed directly to process_prefill_data_for_basic_details, pan_verification using session_id, employment_verification, save_employment_details, and get_bureau_decision. Do NOT call process_address_data again.
        

        ----

        ## Workflow B: Direct Pincode Collection Flow (STRICT CHECKLIST)

        Trigger:
        - Use this ONLY when `get_prefill_data` returns status 500 with error "phoneToPrefill_failed" OR when prefill data is empty (all important fields like pan, name, income, gender, age, dob are empty).
        - NEVER mix with Workflow A.
        - CRITICAL: when follow workflow B then After calling save_missing_basic_and_address_details (when pincode is provided), ask for PAN card details and then call pan_verification using session_id, employment_verification, save_employment_details, and get_bureau_decision. Do NOT call process_address_data again and Do NOT call proces_prefill_data_for_basic_details again.
        

        State Flags (internal):
        - pan_verified = false
        - employment_verified = false
        - employment_details_submitted = false
        - bureau_decision_processed = false

        Step 1: Current Address Pincode Collection
        - Ask ONLY "Please provide 6-digit pincode of Patient's Current address:" and wait for response
        - After collecting pincode, call save_missing_basic_and_address_details
        - Then proceed directly to Step 2 (PAN collection)

        Step 2: PAN Card Collection
        - Ask: "Please provide Patient's PAN card details. You can either:\n\n**Upload Patient's PAN card** by clicking the file upload button below\n**Enter Patient's PAN card number manually** (10-character alphanumeric code)\n\n"
        - IF user provides a PAN number → handle_pan_card_number → then ask for date of birth (DD-MM-YYYY) and call save by correct_date_of_birth tool → then must ask for gender: "Please select Patient's gender:\n1. Male\n2. Female\n" and wait for user response, then call save_gender_B_details
        - IF user uploads PAN card → wait for upload confirmation message(PAN card processed successfully) → then ask: "Please select Patient's gender:\n1. Male\n2. Female\n" and wait for user response, then call save_gender_B_details
        - After PAN is saved and additional details collected, continue with Step 3
      
        Step 3: PAN Verification
        - Call pan_verification using session_id
        - IF it fails → return to Step 2 (re-collect PAN) and retry
        - IF it succeeds → set pan_verified = true → go to Step 4

        Step 4: Employment Verification
        - Call get_employment_verification (continue even if it fails)
        - If status 200 → set employment_verified = true
        - Proceed to Step 5

        Step 5: Save Employment Details
        - Call save_employment_details
        - If status 200 → set employment_details_submitted = true
        - Proceed to Step 6

        Step 6: Process Loan Application
        - Call get_bureau_decision
        - Use its formatted response exactly as provided; set bureau_decision_processed = true
        - END

        Do Not Repeat Rules:
        - Never mix Workflow A and Workflow B.

        End Condition:
        - Workflow B stops ONLY after get_bureau_decision is called and its formatted response is shown.

        Auto-Chain Rule:
        - If any tool output includes continue_chain = true, immediately execute Steps 4-7 without asking for more user input.

        ----

        ## Additional CRITICAL Rules (Apply to Both Workflows)

        - When process_prefill_data returns "status": "missing_details", inform the user and use handle_missing_details_collection until all details are collected, then continue.
        - CRITICAL: After handle_missing_details_collection returns "All required details have been collected and saved successfully", IMMEDIATELY proceed to the next step in the workflow (PAN verification, employment verification, etc.). DO NOT restart the workflow or call basic workflow steps again.
        - NEVER assume or guess user's gender, marital status, or education level.
        - NEVER use name, age, or any other data to determine gender.
        - The ONLY way to get gender is to ask: "Please select Patient's gender:\n1. Male\n2. Female\n"
        - NEVER assume gender from PAN card data or any other source - ALWAYS ask the user explicitly
        - When get_bureau_decision tool returns a formatted response, use it EXACTLY as provided.
        - NEVER duplicate or modify any part of the tool's formatted response.
        - NEVER concatenate or merge multiple formatted responses.
        - NEVER add, modify, or skip any steps in the workflow.
        - If any step fails, continue to the next step unless otherwise specified.
        - Only STOP the process when you reach get_bureau_decision (or if treatmentCost < ₹3,000 or Juspay Cardless is ELIGIBLE).

        ----

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
                "Hello! I'm **Careena 👩🏻‍⚕️**. I'm here to help you check Patient's eligibility for a medical loan.\n\n"
                "To get started, please share the following details:\n\n"
                "1. Patient's full name\n"
                "2. Patient's phone number (linked to their PAN)\n"
                "3. Expected cost of the treatment\n"
                "4. Patient's monthly income\n"
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
    
    def _create_context_aware_system_prompt(self, session_id: str) -> str:
        """
        Create a context-aware system prompt that includes conversation history and session data
        
        Args:
            session_id: Session identifier
            
        Returns:
            Enhanced system prompt with context
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return self.base_system_prompt
            
            # Get conversation history
            history = session.get("history", [])
            data = session.get("data", {})
            
            # Create conversation context
            conversation_context = self._extract_conversation_context(history, data, session_id)
            
            # Create enhanced system prompt
            enhanced_prompt = f"""{self.base_system_prompt}

CONVERSATION CONTEXT AND MEMORY:
{conversation_context}

CRITICAL CONTEXT AWARENESS RULES:
1. ALWAYS refer to the conversation history above to understand the current state
2. DO NOT repeat questions or steps that have already been completed
3. Use the stored data to provide personalized responses
4. If the user asks about previously provided information, refer to the stored data
5. Maintain consistency with previous responses and collected information
6. If the user wants to change something, use the appropriate correction tools
7. Remember the current workflow (A or B) and application status
8. NEVER ask for Aadhaar upload in Workflow A
9. NEVER mix Workflow A and Workflow B - stick to the current workflow
10. Continue from where you left off based on the current step in the workflow
11. CRITICAL: After missing details collection is complete, proceed to the next workflow step (PAN verification, employment verification, etc.) - DO NOT restart the workflow
12. Workflow B is triggered when prefill data is empty (all important fields are empty) or when phoneToPrefill API fails

"""
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error creating context-aware system prompt: {e}")
            return self.base_system_prompt
    
    def _extract_conversation_context(self, history: List[Dict[str, Any]], data: Dict[str, Any], session_id: str = None) -> str:
        """
        Extract meaningful context from conversation history and session data
        
        Args:
            history: Conversation history
            data: Session data
            session_id: Session identifier (optional, for conversation summary)
            
        Returns:
            Formatted conversation context
        """
        try:
            context_parts = []
            
            # Determine current workflow
            current_workflow = self._determine_current_workflow(data, history)
            context_parts.append(f"CURRENT WORKFLOW: {current_workflow}")
            
            # Add session status and progress
            if data:
                context_parts.append("\nCURRENT SESSION STATUS:")
                if data.get("userId"):
                    context_parts.append(f"- User ID: {data.get('userId')}")
                if data.get("fullName"):
                    context_parts.append(f"- Patient Name: {data.get('fullName')}")
                if data.get("phoneNumber"):
                    context_parts.append(f"- Phone Number: {data.get('phoneNumber')}")
                if data.get("treatmentCost"):
                    context_parts.append(f"- Treatment Cost: ₹{data.get('treatmentCost')}")
                if data.get("monthlyIncome"):
                    context_parts.append(f"- Monthly Income: ₹{data.get('monthlyIncome')}")
                if data.get("treatmentReason"):
                    context_parts.append(f"- Treatment Reason: {data.get('treatmentReason')}")
                if data.get("dateOfBirth"):
                    context_parts.append(f"- Date of Birth: {data.get('dateOfBirth')}")
                if data.get("gender"):
                    context_parts.append(f"- Gender: {data.get('gender')}")
                if data.get("maritalStatus"):
                    context_parts.append(f"- Marital Status: {data.get('maritalStatus')}")
                if data.get("educationLevel"):
                    context_parts.append(f"- Education Level: {data.get('educationLevel')}")
                if data.get("panNumber"):
                    context_parts.append(f"- PAN Number: {data.get('panNumber')}")
                
                # Add collection status
                if data.get("additional_details"):
                    context_parts.append(f"- Additional Details Collection: In Progress")
                    collection_step = data.get("additional_details", {}).get("collection_step")
                    if collection_step:
                        context_parts.append(f"- Current Collection Step: {collection_step}")
            
            # Add current workflow step
            current_step = self._determine_current_workflow_step(data, history, current_workflow)
            if current_step:
                context_parts.append(f"\nCURRENT WORKFLOW STEP: {current_step}")
            
            # Add conversation summary for long sessions
            if history and len(history) > 10:
                conversation_summary = self._create_conversation_summary(session_id)
                if conversation_summary:
                    context_parts.append(f"\n{conversation_summary}")
            else:
                # Add recent conversation summary (last 6 messages) for shorter sessions
                if history and len(history) > 1:
                    context_parts.append("\nRECENT CONVERSATION SUMMARY:")
                    recent_messages = history[-6:]  # Last 6 messages
                    for i, msg in enumerate(recent_messages):
                        msg_type = msg.get("type", "HumanMessage")
                        content = msg.get("content", "")
                        # Truncate long messages for context
                        if len(content) > 200:
                            content = content[:200] + "..."
                        context_parts.append(f"- {msg_type}: {content}")
            
            # Add application progress indicators based on workflow
            progress_indicators = self._get_workflow_progress_indicators(data, current_workflow)
            if progress_indicators:
                context_parts.append("\nAPPLICATION PROGRESS:")
                context_parts.extend(progress_indicators)
            
            return "\n".join(context_parts) if context_parts else "No previous context available."
            
        except Exception as e:
            logger.error(f"Error extracting conversation context: {e}")
            return "Error extracting conversation context."
    
    def _determine_current_workflow(self, data: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        """
        Determine which workflow (A or B) is currently active
        
        Args:
            data: Session data
            history: Conversation history
            
        Returns:
            "Workflow A" or "Workflow B"
        """
        try:
            # Check if get_prefill_data returned status 500 with phoneToPrefill_failed or empty data
            for msg in history:
                content = msg.get("content", "")
                if "phoneToPrefill_failed" in content or "phoneToPrefill_empty_data" in content:
                    return "Workflow B"
            
            # Default to Workflow A
            return "Workflow A"
            
        except Exception as e:
            logger.error(f"Error determining current workflow: {e}")
            return "Workflow A"
    
    def _determine_current_workflow_step(self, data: Dict[str, Any], history: List[Dict[str, Any]], workflow: str) -> str:
        """
        Determine the current step in the workflow
        
        Args:
            data: Session data
            history: Conversation history
            workflow: Current workflow (A or B)
            
        Returns:
            Current step description
        """
        try:
            if workflow == "Workflow A":
                return self._determine_workflow_a_step(data, history)
            else:
                return self._determine_workflow_b_step(data, history)
                
        except Exception as e:
            logger.error(f"Error determining current workflow step: {e}")
            return "Unknown step"
    
    def _determine_workflow_a_step(self, data: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        """
        Determine current step in Workflow A
        """
        try:
            # Check if treatment cost is below minimum or above maximum
            treatment_cost = data.get("treatmentCost")
            if treatment_cost:
                if treatment_cost < 3000:
                    return "Treatment cost below minimum (₹3,000) - Application stopped"
                elif treatment_cost > 1000000:
                    return "Treatment cost above maximum (₹10,00,000) - Application stopped"
            
            # Check if Juspay Cardless is eligible
            for msg in history:
                content = msg.get("content", "")
                if "ELIGIBLE" in content and "Juspay Cardless" in content:
                    return "Juspay Cardless approved - Application completed"
            
            # Check workflow steps
            if not data.get("fullName") or not data.get("phoneNumber") or not data.get("treatmentCost") or not data.get("monthlyIncome"):
                return "Step 1: Initial Data Collection"
            
            if not data.get("userId"):
                return "Step 2: User ID Creation"
            
            if not data.get("basic_details_submitted"):
                return "Step 3: Basic Details Submission"
            
            if not data.get("loan_details_submitted"):
                return "Step 4: Save Loan Details"
            
            # Check if Juspay Cardless check is done
            jp_cardless_checked = any("check_jp_cardless" in str(msg.get("content", "")) for msg in history)
            if not jp_cardless_checked:
                return "Step 5: Check Juspay Cardless"
            
            if not data.get("prefill_data_processed"):
                return "Step 6: Data Prefill"
            
            if not data.get("address_processed"):
                return "Step 7: Address Processing"
            
            if not data.get("basic_details_prefill_processed"):
                return "Step 8: Process Prefill Data for Basic Details"
            
            # Check for missing details collection
            if data.get("additional_details", {}).get("collection_step"):
                return f"Step 8: Collecting Missing Details - {data.get('additional_details', {}).get('collection_step')}"
            
            # Check if missing details collection is complete but basic details prefill is not processed
            if data.get("all_details_collected") and not data.get("basic_details_prefill_processed"):
                return "Step 8: Processing Basic Details After Missing Details Collection"
            
            if not data.get("pan_verified"):
                return "Step 9: PAN Card Collection"
            
            if not data.get("employment_verified"):
                return "Step 10: Employment Verification"
            
            if not data.get("employment_details_submitted"):
                return "Step 11: Save Employment Details"
            
            if not data.get("bureau_decision_processed"):
                return "Step 12: Process Loan Application (Bureau Decision)"
            
            return "Application completed - Bureau decision processed"
            
        except Exception as e:
            logger.error(f"Error determining Workflow A step: {e}")
            return "Unknown step"
    
    def _determine_workflow_b_step(self, data: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        """
        Determine current step in Workflow B
        """
        try:
            # Check if pincode is collected
            if not data.get("pincode_collected"):
                return "Step 1: Current Address Pincode Collection"
            
            # Check if PAN is provided
            if not data.get("panNumber"):
                return "Step 2: PAN Card Collection"
            
            # Check if PAN verification is done
            if not data.get("pan_verified"):
                return "Step 3: PAN Verification"
            
            # Check if employment verification is done
            if not data.get("employment_verified"):
                return "Step 4: Employment Verification"
            
            # Check if employment details are saved
            if not data.get("employment_details_submitted"):
                return "Step 5: Save Employment Details"
            
            # Check if bureau decision is processed
            if not data.get("bureau_decision_processed"):
                return "Step 6: Process Loan Application (Bureau Decision)"
            
            return "Application completed - Bureau decision processed"
            
        except Exception as e:
            logger.error(f"Error determining Workflow B step: {e}")
            return "Unknown step"
    
    def _get_workflow_progress_indicators(self, data: Dict[str, Any], workflow: str) -> List[str]:
        """
        Get progress indicators based on the current workflow
        
        Args:
            data: Session data
            workflow: Current workflow (A or B)
            
        Returns:
            List of progress indicators
        """
        try:
            indicators = []
            
            if workflow == "Workflow A":
                # Workflow A progress indicators
                if data.get("userId"):
                    indicators.append("✓ User ID created")
                if data.get("fullName") and data.get("phoneNumber") and data.get("treatmentCost") and data.get("monthlyIncome"):
                    indicators.append("✓ Initial data collected")
                if data.get("basic_details_submitted"):
                    indicators.append("✓ Basic details submitted")
                if data.get("loan_details_submitted"):
                    indicators.append("✓ Loan details submitted")
                if data.get("prefill_data_processed"):
                    indicators.append("✓ Data prefill processed")
                if data.get("address_processed"):
                    indicators.append("✓ Address processed")
                if data.get("basic_details_prefill_processed"):
                    indicators.append("✓ Basic details prefill processed")
                if data.get("all_details_collected"):
                    indicators.append("✓ All missing details collected")
                if data.get("pan_verified"):
                    indicators.append("✓ PAN verified")
                if data.get("employment_verified"):
                    indicators.append("✓ Employment verified")
                if data.get("employment_details_submitted"):
                    indicators.append("✓ Employment details submitted")
                if data.get("bureau_decision_processed"):
                    indicators.append("✓ Bureau decision processed")
            else:
                # Workflow B progress indicators
                if data.get("pincode_collected"):
                    indicators.append("✓ Pincode collected")
                if data.get("panNumber"):
                    indicators.append("✓ PAN number provided")
                if data.get("pan_verified"):
                    indicators.append("✓ PAN verified")
                if data.get("employment_verified"):
                    indicators.append("✓ Employment verified")
                if data.get("employment_details_submitted"):
                    indicators.append("✓ Employment details submitted")
                if data.get("bureau_decision_processed"):
                    indicators.append("✓ Bureau decision processed")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error getting workflow progress indicators: {e}")
            return []
    
    def _validate_context_consistency(self, session_id: str, message: str, ai_message: str) -> bool:
        """
        Validate that the AI response is consistent with the conversation context
        
        Args:
            session_id: Session identifier
            message: User message
            ai_message: AI response
            
        Returns:
            True if context is consistent, False if inconsistencies detected
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return True
            
            data = session.get("data", {})
            message_lower = message.lower()
            ai_message_lower = ai_message.lower()
            
            # Check for common hallucination patterns
            inconsistencies = []
            
            # Check if AI is asking for information that's already been provided
            if data.get("fullName") and "name" in ai_message_lower and "what is" in ai_message_lower:
                inconsistencies.append("AI asking for name when already provided")
            
            if data.get("phoneNumber") and "phone" in ai_message_lower and "what is" in ai_message_lower:
                inconsistencies.append("AI asking for phone when already provided")
            
            if data.get("treatmentCost") and "cost" in ai_message_lower and "what is" in ai_message_lower:
                inconsistencies.append("AI asking for treatment cost when already provided")
            
            if data.get("monthlyIncome") and "income" in ai_message_lower and "what is" in ai_message_lower:
                inconsistencies.append("AI asking for income when already provided")
            
            # Check workflow-specific inconsistencies
            current_workflow = self._determine_current_workflow(data, session.get("history", []))
            
            if current_workflow == "Workflow A":
                # In Workflow A, never ask for Aadhaar upload
                if "aadhaar" in ai_message_lower and "upload" in ai_message_lower:
                    inconsistencies.append("AI asking for Aadhaar upload in Workflow A")
            
            # Check if AI is asking for PAN upload when PAN number is already provided
            if data.get("panNumber") and "pan" in ai_message_lower and "upload" in ai_message_lower:
                inconsistencies.append("AI asking for PAN upload when PAN number already provided")
            
            # Check if AI is asking for employment type when it's already been collected
            if data.get("employment_type_collected") and "employment type" in ai_message_lower:
                inconsistencies.append("AI asking for employment type when already collected")
            
            # Log inconsistencies for debugging
            if inconsistencies:
                logger.warning(f"Context inconsistencies detected in session {session_id}: {inconsistencies}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating context consistency: {e}")
            return True  # Default to allowing the response if validation fails
    
    def get_conversation_context(self, session_id: str) -> str:
        """
        Get the current conversation context for debugging and monitoring
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted conversation context
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return "Session not found"
            
            history = session.get("history", [])
            data = session.get("data", {})
            
            return self._extract_conversation_context(history, data, session_id)
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return f"Error retrieving context: {e}"
    
    def _create_conversation_summary(self, session_id: str) -> str:
        """
        Create a summary of the conversation for long sessions to maintain context
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation summary
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return ""
            
            history = session.get("history", [])
            data = session.get("data", {})
            
            if len(history) < 10:  # No need for summary for short conversations
                return ""
            
            summary_parts = []
            summary_parts.append("CONVERSATION SUMMARY:")
            
            # Determine workflow
            current_workflow = self._determine_current_workflow(data, history)
            summary_parts.append(f"Workflow: {current_workflow}")
            
            # Key milestones in the conversation
            milestones = []
            
            # Check for key events in history
            for i, msg in enumerate(history):
                content = msg.get("content", "").lower()
                msg_type = msg.get("type", "")
                
                if msg_type == "HumanMessage":
                    if any(keyword in content for keyword in ["name:", "phone:", "cost:", "income:"]):
                        milestones.append(f"Step {i//2 + 1}: User provided initial data")
                    elif "pincode" in content and len(content.strip()) == 6:
                        milestones.append(f"Step {i//2 + 1}: User provided pincode")
                    elif "pan" in content and ("upload" in content or len(content.strip()) == 10):
                        milestones.append(f"Step {i//2 + 1}: User provided PAN")
                    elif any(keyword in content for keyword in ["1", "2", "salaried", "self-employed"]):
                        milestones.append(f"Step {i//2 + 1}: User selected employment type")
                    elif "@" in content and "." in content:
                        milestones.append(f"Step {i//2 + 1}: User provided email")
                
                elif msg_type == "AIMessage":
                    if "employment type" in content and "1. SALARIED" in content:
                        milestones.append(f"Step {i//2 + 1}: AI asked for employment type")
                    elif "pincode" in content and "current address" in content:
                        milestones.append(f"Step {i//2 + 1}: AI requested pincode")
                    elif "pan" in content and "upload" in content:
                        milestones.append(f"Step {i//2 + 1}: AI requested PAN upload")
                    elif "email" in content and "address" in content:
                        milestones.append(f"Step {i//2 + 1}: AI requested email")
            
            # Add unique milestones
            unique_milestones = []
            for milestone in milestones:
                if milestone not in unique_milestones:
                    unique_milestones.append(milestone)
            
            summary_parts.extend(unique_milestones[-5:])  # Last 5 milestones
            
            # Add current application status
            if data:
                summary_parts.append("\nCURRENT STATUS:")
                if data.get("userId"):
                    summary_parts.append("- User ID created and stored")
                if data.get("fullName") and data.get("phoneNumber"):
                    summary_parts.append("- Patient details collected")
                if data.get("treatmentCost"):
                    summary_parts.append(f"- Treatment cost: ₹{data.get('treatmentCost')}")
                if data.get("pincode_collected"):
                    summary_parts.append("- Pincode collected")
                if data.get("panNumber"):
                    summary_parts.append("- PAN number provided")
                if data.get("pan_verified"):
                    summary_parts.append("- PAN verification completed")
                if data.get("email"):
                    summary_parts.append("- Email provided")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating conversation summary: {e}")
            return ""
    
    def _get_optimized_chat_history(self, session_id: str, max_messages: int = 10) -> List:
        """
        Get optimized chat history for the agent with context preservation
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Optimized chat history as LangChain messages
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return []
            
            history = session.get("history", [])
            
            # If history is short, return all messages
            if len(history) <= max_messages:
                return self._convert_to_langchain_messages(history)
            
            # For longer histories, keep the first message (greeting) and recent messages
            optimized_history = []
            
            # Always keep the first message (initial greeting)
            if history:
                optimized_history.append(history[0])
            
            # Add recent messages
            recent_messages = history[-(max_messages-1):] if len(history) > 1 else []
            optimized_history.extend(recent_messages)
            
            return self._convert_to_langchain_messages(optimized_history)
            
        except Exception as e:
            logger.error(f"Error getting optimized chat history: {e}")
            return []
    
    def _update_conversation_progress(self, session_id: str, message: str, ai_message: str) -> None:
        """
        Update conversation progress indicators based on message content and session data
        
        Args:
            session_id: Session identifier
            message: User message
            ai_message: AI response
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return
            
            data = session.get("data", {})
            message_lower = message.lower()
            ai_message_lower = ai_message.lower()
            
            # Track progress based on message content and AI responses
            progress_updates = {}
            
            # Check for initial data collection completion
            if (data.get("fullName") and data.get("phoneNumber") and 
                data.get("treatmentCost") and data.get("monthlyIncome") and 
                not data.get("initial_data_completed")):
                progress_updates["initial_data_completed"] = True
            
            # Check for basic details submission
            if "basic details" in ai_message_lower and "submitted" in ai_message_lower:
                progress_updates["basic_details_submitted"] = True
            
            # Check for loan details submission
            if "loan details" in ai_message_lower and "submitted" in ai_message_lower:
                progress_updates["loan_details_submitted"] = True
            
            # Check for pincode collection
            if "pincode" in message_lower and len(message_lower.strip()) == 6:
                progress_updates["pincode_collected"] = True
            
            # Check for PAN upload
            if ("pan" in message_lower and "upload" in message_lower) or "pan card" in ai_message_lower:
                progress_updates["pan_uploaded"] = True
            
            # Check for employment type collection
            if "employment type" in ai_message_lower and "1. SALARIED" in ai_message:
                progress_updates["employment_type_collected"] = True
            
            # Update progress in session data
            for key, value in progress_updates.items():
                SessionManager.update_session_data_field(session_id, f"data.{key}", value)
                logger.info(f"Updated progress for session {session_id}: {key} = {value}")
                
        except Exception as e:
            logger.error(f"Error updating conversation progress: {e}")
    
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
            logger.info(f"Session {session_id} current status: {current_status}")

            # Helper: detect employment type prompt in a string
            def is_employment_type_prompt(text: str) -> bool:
                return (
                    "Patient's employment type:" in text
                    and "1. SALARIED" in text
                    and "2. SELF_EMPLOYED" in text
                    and "Please Enter input 1 or 2 only" in text
                )

            # Helper: detect limit options prompt in a string
            def is_limit_options_prompt(text: str) -> bool:
                return (
                    "Continue with this limit" in text
                    and "Continue with limit enhancement" in text
                    and ("1." in text and "2." in text)
                )

            # If already collecting additional details, use the sequential handler,
            # but allow status to be changed if agent message contains employment type question
            if current_status == "collecting_additional_details":
                logger.info(f"Session {session_id}: Entering additional details collection mode")
                ai_message = self._handle_additional_details_collection(session_id, message)
                # If the AI message contains the employment type prompt, update status accordingly
                if is_employment_type_prompt(ai_message):
                    logger.info(f"Employment type prompt detected in collecting_additional_details mode, updating session status for {session_id}")
                    SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                    SessionManager.update_session_data_field(session_id, "data.collection_step", "employment_type")
                    updated_session = SessionManager.get_session_from_db(session_id)
                    if not updated_session.get("data", {}).get("additional_details"):
                        SessionManager.update_session_data_field(session_id, "data.additional_details", {})
                    logger.info(f"Session {session_id} marked as collecting_additional_details (from collecting_additional_details branch)")
                self._update_session_history(session_id, message, ai_message)
                return ai_message

            # Handle post-approval address details flow when additional_details_completed
            if current_status == "additional_details_completed":
                logger.info(f"Session {session_id}: Handling post-approval address details flow")
                ai_message = self._handle_post_approval_address_details(session_id, message)
                self._update_session_history(session_id, message, ai_message)
                return ai_message

            # Handle post-approval address details completion
            if current_status == "post_approval_address_details":
                logger.info(f"Session {session_id}: Handling post-approval address details completion")
                ai_message = self._handle_address_details_completion(session_id, message)
                self._update_session_history(session_id, message, ai_message)
                return ai_message

            # Handle KYC completed status
            if current_status == "kyc_completed":
                logger.info(f"Session {session_id}: Handling KYC completed status")
                ai_message = self._handle_kyc_completed_status(session_id, message)
                self._update_session_history(session_id, message, ai_message)
                return ai_message

            # Handle loan disbursal ready status
            if current_status == "loan_disbursal_ready":
                logger.info(f"Session {session_id}: Handling loan disbursal ready status")
                ai_message = self._handle_loan_disbursal_ready_status(session_id, message)
                self._update_session_history(session_id, message, ai_message)
                return ai_message

            logger.info(f"Session {session_id}: Using full agent executor (status: {current_status})")
            session_tools = self._create_session_aware_tools(session_id)

            # Create context-aware system prompt with conversation history and session data
            context_aware_system_prompt = self._create_context_aware_system_prompt(session_id)
            
            # Get optimized chat history for better context management
            optimized_chat_history = self._get_optimized_chat_history(session_id, max_messages=12)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", context_aware_system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            agent = create_openai_functions_agent(self.llm, session_tools, prompt)
            session_agent_executor = AgentExecutor(
                agent=agent,
                tools=session_tools,
                verbose=True,
                max_iterations=50,
                handle_parsing_errors=True,
                return_intermediate_steps=True,  # Ensure we get intermediate steps
            )

            # Use optimized chat history instead of full history
            chat_history = optimized_chat_history.copy()
            chat_history.append(HumanMessage(content=message))

            # Check if the message might trigger employment type questions
            message_triggers_employment_check = any(keyword in message.lower() for keyword in [
                "continue", "next", "proceed", "what", "employment", "type", "salaried", "self-employed"
            ])
            
            response = session_agent_executor.invoke({
                "input": message,
                "chat_history": chat_history
            })

            logger.info(f"Agent executor response keys: {list(response.keys())}")
            logger.info(f"Agent executor output: {response.get('output', 'No output')}")
            
            # Check if get_bureau_decision was called by looking at intermediate steps
            bureau_decision_response = None
            if "intermediate_steps" in response:
                for step in response.get("intermediate_steps", []):
                    if len(step) >= 2 and hasattr(step[0], 'tool') and step[0].tool == "get_bureau_decision":
                        tool_output = step[1]
                        logger.info(f"Found get_bureau_decision in intermediate steps with output: {tool_output}")
                        if "Patient's employment type:" in str(tool_output):
                            bureau_decision_response = str(tool_output)
                            logger.info(f"Using bureau decision tool output as final response")
                            break
            
            # If bureau decision was called and has employment type prompt, use its response
            if bureau_decision_response:
                ai_message = bureau_decision_response
            else:
                ai_message = response.get("output", "I'm processing your request. Please wait.")
                
            # Additional check: if the response is just "1. SALARIED" or similar, it's wrong
            if ai_message.strip() in ["1. SALARIED", "2. SELF_EMPLOYED", "1", "2", "SALARIED", "SELF_EMPLOYED"]:
                logger.error(f"Agent returned incorrect simplified response: {ai_message}")
                # Try to get the bureau decision directly
                try:
                    bureau_result = self.get_bureau_decision(session_id)
                    if bureau_result and "Patient's employment type:" in bureau_result:
                        ai_message = bureau_result
                        logger.info(f"Forced bureau decision call to get correct response: {ai_message}")
                except Exception as e:
                    logger.error(f"Error forcing bureau decision: {e}")

            # Check if the response came from get_bureau_decision tool and use it directly
            bureau_decision_tool_used = False
            bureau_decision_tool_output = None
            if "intermediate_steps" in response:
                logger.info(f"Checking intermediate steps for bureau decision tool: {len(response['intermediate_steps'])} steps")
                for i, step in enumerate(response["intermediate_steps"]):
                    logger.info(f"Step {i}: tool={step[0].tool if len(step) > 0 else 'None'}")
                    if len(step) >= 2 and step[0].tool == "get_bureau_decision":
                        tool_output = step[1]
                        logger.info(f"Found get_bureau_decision tool, output: {tool_output}")
                        if is_employment_type_prompt(str(tool_output)):
                            bureau_decision_tool_output = str(tool_output)
                            # Remove duplicate lines
                            lines = bureau_decision_tool_output.split('\n')
                            unique_lines = []
                            seen_lines = set()
                            for line in lines:
                                line = line.strip()
                                if line and line not in seen_lines:
                                    unique_lines.append(line)
                                    seen_lines.add(line)
                            bureau_decision_tool_output = '\n'.join(unique_lines)
                            bureau_decision_tool_used = True
                            logger.info(f"Found get_bureau_decision tool output with employment type prompt: {bureau_decision_tool_output}")
                            break
                        elif is_limit_options_prompt(str(tool_output)):
                            bureau_decision_tool_output = str(tool_output)
                            # Remove duplicate lines
                            lines = bureau_decision_tool_output.split('\n')
                            unique_lines = []
                            seen_lines = set()
                            for line in lines:
                                line = line.strip()
                                if line and line not in seen_lines:
                                    unique_lines.append(line)
                                    seen_lines.add(line)
                            bureau_decision_tool_output = '\n'.join(unique_lines)
                            bureau_decision_tool_used = True
                            logger.info(f"Found get_bureau_decision tool output with limit options prompt: {bureau_decision_tool_output}")
                            break
            else:
                logger.info("No intermediate_steps found in response - agent executor may not have called any tools")

            # If bureau decision tool was used and prompt is present, update status and return
            if bureau_decision_tool_used and bureau_decision_tool_output:
                if is_employment_type_prompt(bureau_decision_tool_output):
                    logger.info(f"Employment type prompt detected, updating session status for {session_id}")
                    SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                    SessionManager.update_session_data_field(session_id, "data.collection_step", "employment_type")
                    updated_session = SessionManager.get_session_from_db(session_id)
                    if not updated_session.get("data", {}).get("additional_details"):
                        SessionManager.update_session_data_field(session_id, "data.additional_details", {})
                    logger.info(f"Session {session_id} marked as collecting_additional_details (from bureau_decision_tool branch)")
                    
                    # Force verify the status was updated
                    final_session = SessionManager.get_session_from_db(session_id)
                    if final_session:
                        logger.info(f"Final session status after update: {final_session.get('status')}")
                        if final_session.get('status') != "collecting_additional_details":
                            logger.error(f"Status update failed! Current status: {final_session.get('status')}")
                            # Force update again
                            SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                            logger.info("Forced status update again")
                elif is_limit_options_prompt(bureau_decision_tool_output):
                    logger.info(f"Limit options prompt detected, updating session status for {session_id}")
                    SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                    SessionManager.update_session_data_field(session_id, "data.collection_step", "limit_options")
                    updated_session = SessionManager.get_session_from_db(session_id)
                    if not updated_session.get("data", {}).get("additional_details"):
                        SessionManager.update_session_data_field(session_id, "data.additional_details", {})
                    logger.info(f"Session {session_id} marked as collecting_additional_details (from limit_options branch)")
                
                self._update_session_history(session_id, message, bureau_decision_tool_output)
                return bureau_decision_tool_output

            # Check if the agent executor output contains employment type prompt (even if tool wasn't called directly)
            employment_type_prompt_in_output = is_employment_type_prompt(ai_message)
            
            # Check if the agent executor output contains limit options prompt
            limit_options_prompt_in_output = is_limit_options_prompt(ai_message)

            # Check if agent should have called bureau decision tool but didn't
            should_have_called_bureau_tool = (
                employment_type_prompt_in_output
                and not bureau_decision_tool_used
                and "intermediate_steps" in response
                and len(response["intermediate_steps"]) > 0
            )

            # ALWAYS force the bureau decision tool call if employment type prompt is detected
            if employment_type_prompt_in_output and not bureau_decision_tool_used:
                logger.warning(f"Employment type prompt detected but get_bureau_decision tool not used. Forcing tool call.")
                try:
                    bureau_result = self.get_bureau_decision(session_id)
                    if bureau_result and is_employment_type_prompt(bureau_result):
                        ai_message = bureau_result
                        logger.info(f"Forced bureau decision tool call successful: {ai_message}")
                        # Update the response to indicate tool was used
                        bureau_decision_tool_used = True
                        bureau_decision_tool_output = bureau_result
                    else:
                        logger.error(f"Forced bureau decision tool call returned invalid result: {bureau_result}")
                except Exception as e:
                    logger.error(f"Error forcing bureau decision tool call: {e}")

            # If employment type prompt is present in output, update status and collection step
            if employment_type_prompt_in_output:
                logger.info(f"Employment type prompt detected in agent output, updating session status for {session_id}")
                SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                SessionManager.update_session_data_field(session_id, "data.collection_step", "employment_type")
                updated_session = SessionManager.get_session_from_db(session_id)
                if not updated_session.get("data", {}).get("additional_details"):
                    SessionManager.update_session_data_field(session_id, "data.additional_details", {})
                logger.info(f"Session {session_id} marked as collecting_additional_details (from agent output branch)")
                
                # Force verify the status was updated
                final_session = SessionManager.get_session_from_db(session_id)
                if final_session:
                    logger.info(f"Final session status after update: {final_session.get('status')}")
                    if final_session.get('status') != "collecting_additional_details":
                        logger.error(f"Status update failed! Current status: {final_session.get('status')}")
                        # Force update again
                        SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                        logger.info("Forced status update again")
                
                self._update_session_history(session_id, message, ai_message)
                logger.info(f"Final response to user: {ai_message}")
                return ai_message

            # If limit options prompt is present in output, update status and collection step
            if limit_options_prompt_in_output:
                logger.info(f"Limit options prompt detected in agent output, updating session status for {session_id}")
                SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                SessionManager.update_session_data_field(session_id, "data.collection_step", "limit_options")
                updated_session = SessionManager.get_session_from_db(session_id)
                if not updated_session.get("data", {}).get("additional_details"):
                    SessionManager.update_session_data_field(session_id, "data.additional_details", {})
                logger.info(f"Session {session_id} marked as collecting_additional_details (from limit_options output branch)")
                
                self._update_session_history(session_id, message, ai_message)
                logger.info(f"Final response to user: {ai_message}")
                return ai_message

            # Final check: if the response contains employment type prompt, ensure status is updated
            if is_employment_type_prompt(ai_message) and current_status != "collecting_additional_details":
                logger.warning(f"Employment type prompt in final response but status not updated. Forcing update.")
                SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                SessionManager.update_session_data_field(session_id, "data.collection_step", "employment_type")
                if not session.get("data", {}).get("additional_details"):
                    SessionManager.update_session_data_field(session_id, "data.additional_details", {})
                logger.info(f"Forced status update to collecting_additional_details")
            
            # Check if user is trying to make corrections before application completion
            correction_keywords = ["change", "correct", "update", "modify", "edit", "wrong", "mistake"]
            is_correction_request = any(keyword in message.lower() for keyword in correction_keywords)
            
            
            # Otherwise, just update the conversation history and return
            self._update_session_history(session_id, message, ai_message)
            logger.info(f"Final response to user: {ai_message}")
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
        Efficiently update session history with new messages and track progress
        
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
            
            # Update conversation progress tracking
            self._update_conversation_progress(session_id, user_message, ai_message)
            
            # Validate context consistency to prevent hallucination
            is_consistent = self._validate_context_consistency(session_id, user_message, ai_message)
            if not is_consistent:
                logger.warning(f"Context inconsistency detected in session {session_id}. AI response may need review.")
            
        except Exception as e:
            logger.error(f"Error updating session history: {e}")

    # def get_session_data(self, session_id: str = None) -> str:
    #     """
    #     Get session data for the current session
        
    #     Args:
    #         session_id: Session ID (required)
            
    #     Returns:
    #         Session data as JSON string with comprehensive data view
    #     """
    #     if not session_id:
    #         return "Session ID not found"
        
    #     session = SessionManager.get_session_from_db(session_id)
    #     if not session:
    #         return "Session ID not found"
        
    #     # Create a comprehensive view of session data
    #     comprehensive_session = {
    #         "session_info": {
    #             "id": session.get("id"),
    #             "status": session.get("status"),
    #             "created_at": session.get("created_at"),
    #             "phone_number": session.get("phone_number")
    #         },
    #         "user_data": session.get("data", {}),
    #         "conversation_history": []
    #     }
        
    #     # History is already in serializable format
    #     if "history" in session:
    #         comprehensive_session["conversation_history"] = session["history"]
        
    #     # Add summary of stored data
    #     data = session.get("data", {})
    #     data_summary = {
    #         "core_user_data": {
    #             "userId": data.get("userId"),
    #             "fullName": data.get("fullName") or data.get("name"),
    #             "phoneNumber": data.get("phoneNumber") or data.get("phone"),
    #             "treatmentCost": data.get("treatmentCost"),
    #             "monthlyIncome": data.get("monthlyIncome"),
    #         },
    #         "api_responses_count": len(data.get("api_responses", {})),
    #         "api_requests_count": len(data.get("api_requests", {})),
    #         "prefill_data_available": bool(data.get("prefill_data")),
    #         "employment_data_available": bool(data.get("employment_data")),
    #         "bureau_decision_available": bool(data.get("bureau_decision_details")),
    #         "additional_details_available": bool(data.get("additional_details")),
    #     }
        
    #     comprehensive_session["data_summary"] = data_summary
        
    #     return json.dumps(comprehensive_session, indent=2)
    
    # Tool implementations
    
    def store_user_data_structured(self, fullName: str, phoneNumber: str, treatmentCost: int, monthlyIncome: int, session_id: str) -> str:
        """
        Store user data in the session using structured input
        
        Args:
            fullName: Patient's full name
            phoneNumber: Patient's phone number
            treatmentCost: Treatment cost amount
            monthlyIncome: Monthly income amount
            session_id: Session identifier
            
        Returns:
            Confirmation message
        """
        try:
            # Convert structured input to JSON string format
            data = {
                "fullName": fullName,
                "phoneNumber": phoneNumber,
                "treatmentCost": treatmentCost,
                "monthlyIncome": monthlyIncome
            }
            
            # Convert to JSON string and call the original method
            input_str = json.dumps(data)
            return self.store_user_data(input_str, session_id)
            
        except Exception as e:
            logger.error(f"Error in store_user_data_structured: {e}")
            return f"Error storing data: {str(e)}"

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
                        return f"I understand your treatment cost is ₹{cost_value:,.0f}. Currently, I can only process loan applications for treatments costing ₹3,000 or more. Please let me know if your treatment cost is ₹3,000 or above, and I'll be happy to help you with the loan application process."
                    elif cost_value > 1000000:
                        return f"I understand your treatment cost is ₹{cost_value:,.0f}. Currently, I can only process loan applications for treatments costing up to ₹10,00,000. Please let me know if your treatment cost is ₹10,00,000 or below, and I'll be happy to help you with the loan application process."
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
            
            return session_id
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
            
            # Check if the API call failed with 500 error
            if result.get("status") == 500:
                logger.warning(f"phoneToPrefill API failed with 500 error for user_id: {user_id}")
                # Return a specific message asking for Aadhaar upload
                return json.dumps({
                    "status": 500,
                    "error": "phoneToPrefill_failed",
                    "message": "Follow workflow B. Please provide 6-digit pincode of Patient's Current address: ",
                    "requires_pincode_collection": True
                })
            
            # Check if the API call was successful but returned empty data
            if result.get("status") == 200:
                data = result.get("data", {})
                response_data = data.get("response", {})
                
                # Check if all important fields are empty
                is_empty = True
                if response_data:
                    # Check if any of the important fields have data
                    important_fields = ["pan", "gender", "dob", "email"]
                    for field in important_fields:
                        field_value = response_data.get(field, "")
                        if field_value and str(field_value).strip():
                            # For nested name object, check if any name field has data
                            if field == "name" and isinstance(field_value, dict):
                                name_fields = ["fullName", "firstName", "lastName"]
                                for name_field in name_fields:
                                    if field_value.get(name_field, "").strip():
                                        is_empty = False
                                        break
                            else:
                                is_empty = False
                                break
                
                if is_empty:
                    logger.warning(f"phoneToPrefill API returned empty data for user_id: {user_id}")
                    # Return a specific message asking for Aadhaar upload
                    return json.dumps({
                        "status": 500,
                        "error": "phoneToPrefill_empty_data",
                        "message": "Follow workflow B. Please provide 6-digit pincode of Patient's Current address:",
                        "requires_pincode_collection": True
                    })
            
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
    
    def save_loan_details_structured(self, fullName: str, treatmentCost: int, userId: str, session_id: str) -> str:
        """
        Save loan details using structured input
        
        Args:
            fullName: Patient's full name
            treatmentCost: Treatment cost amount
            userId: User ID
            session_id: Session identifier
            
        Returns:
            Save result as JSON string
        """
        try:
            # Convert structured input to JSON string format
            data = {
                "fullName": fullName,
                "treatmentCost": treatmentCost,
                "userId": userId
            }
            
            # Convert to JSON string and call the original method
            input_str = json.dumps(data)
            return self.save_loan_details(input_str, session_id)
            
        except Exception as e:
            logger.error(f"Error in save_loan_details_structured: {e}")
            return f"Error saving loan details: {str(e)}"

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
                logger.info(f"Bureau result: {bureau_result}")
                
                # Save the extracted bureau decision details to session data
                if session_id:
                    SessionManager.update_session_data_field(session_id, "data.bureau_decision_details", bureau_result)
                    logger.info(f"Session {session_id}: Saved bureau decision details to session data")
                
                # Format the response using the new function
                formatted_response = self._format_bureau_decision_response(bureau_result, session_id)
                logger.info(f"Formatted response: {formatted_response}")
                
                # Ensure we always return a string
                if formatted_response is None:
                    logger.error("Formatted response is None, returning default message")
                    return "There was an error processing the loan decision. Please try again."
                
                return formatted_response
            
            # Save the raw result as bureau decision details even if it's not a successful response
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.bureau_decision_details", result)
                logger.info(f"Session {session_id}: Saved raw bureau decision result to session data")
            
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error getting bureau decision: {e}")
            error_result = {
                "status": 500,
                "error": f"Error getting bureau decision: {str(e)}"
            }
            
            # Save error information to session data
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.bureau_decision_details", error_result)
                logger.info(f"Session {session_id}: Saved bureau decision error to session data")
            
            return json.dumps(error_result)

    def extract_bureau_decision_details(self, bureau_result: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Extract and format eligible EMI details from a bureau decision result
        
        Args:
            bureau_result: Bureau decision API response
            session_id: Session identifier
            
        Returns:
            Dictionary with formatted bureau decision details
        """
        try:
            details = {
                "status": None,
                "reason": None,
                "maxEligibleEMI": None,
                "emiPlans": [],
                "creditLimitCalculated": None,
                "loanAmount": None,
                "maxTreatmentAmount": None
            }
            
            if not isinstance(bureau_result, dict) or bureau_result.get("status") != 200:
                return details
                
            data = bureau_result.get("data", {})
            
            # Handle nested structure where actual data is in data.data
            if "data" in data and isinstance(data["data"], dict):
                data = data["data"]
            
            # Extract status from multiple possible locations
            if "finalDecision" in data:
                details["status"] = data["finalDecision"]
            elif "status" in data:
                details["status"] = data["status"]
            elif "bureauDecision" in data:
                details["status"] = data["bureauDecision"]
            
            # Log the extracted status for debugging
            logger.info(f"Extracted bureau decision status: {details['status']}")
            
            # Extract loan amount
            if "loanAmount" in data:
                details["loanAmount"] = data["loanAmount"]
            
            # Extract max eligible EMI
            if "maxEligibleEmi" in data:
                details["maxEligibleEMI"] = data["maxEligibleEmi"]
            elif "maxEligibleEMI" in data:
                details["maxEligibleEMI"] = data["maxEligibleEMI"]
            elif "eligibleEMI" in data:
                details["maxEligibleEMI"] = data["eligibleEMI"]
            
            # Extract reason from rejection reasons or other sources
            if "rejectionReasons" in data and isinstance(data["rejectionReasons"], list) and len(data["rejectionReasons"]) > 0:
                details["reason"] = ", ".join(data["rejectionReasons"])
            elif "reason" in data:
                details["reason"] = data["reason"]
            elif "decisionReason" in data:
                details["reason"] = data["decisionReason"]
            elif "bureauChecks" in data and isinstance(data["bureauChecks"], list):
                for check in data["bureauChecks"]:
                    if isinstance(check, dict) and check.get("autoDecision") == "FAILED":
                        if "policyCheck" in check:
                            details["reason"] = f"Failed {check['policyCheck']} check"
                            break
            
            # Extract EMI plans and find max credit limit and treatment amount
            emi_plans_data = None
            if "emiPlanList" in data and isinstance(data["emiPlanList"], list):
                emi_plans_data = data["emiPlanList"]
            elif "emiPlans" in data and isinstance(data["emiPlans"], list):
                emi_plans_data = data["emiPlans"]
            
            if emi_plans_data:
                details["emiPlans"] = emi_plans_data
                
                # Find maximum creditLimit from all plans
                try:
                    max_credit_limit = max(
                        (float(plan.get("creditLimitCalculated", 0)) for plan in emi_plans_data if plan.get("creditLimitCalculated")),
                        default=None
                    )
                    if max_credit_limit:
                        details["creditLimitCalculated"] = str(int(max_credit_limit))
                except (ValueError, TypeError):
                    pass
                
                # Find maximum treatment amount by getting the highest grossTreatmentAmount from all plans
                try:
                    max_treatment_amount = max(
                        (float(plan.get("grossTreatmentAmount", 0)) for plan in emi_plans_data if plan.get("grossTreatmentAmount")),
                        default=None
                    )
                    if max_treatment_amount:
                        details["maxTreatmentAmount"] = str(int(max_treatment_amount))
                except (ValueError, TypeError):
                    pass

                print(f"max_treatment_amount: {max_treatment_amount}")
                
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
            
            # Ensure credit limit, EMI, down payment, and net loan amount values are strings
            for plan in details["emiPlans"]:
                for key in ["creditLimitCalculated", "emi", "downPayment", "netLoanAmount"]:
                    if key in plan and plan[key] is not None and not isinstance(plan[key], str):
                        plan[key] = str(plan[key])
                
                # Ensure grossTreatmentAmount is available and is a string
                if "grossTreatmentAmount" in plan and plan["grossTreatmentAmount"] is not None and not isinstance(plan["grossTreatmentAmount"], str):
                    plan["grossTreatmentAmount"] = str(plan["grossTreatmentAmount"])
            
            # Log the complete details dictionary for debugging
            logger.info(f"Extracted bureau decision details: {details}")

            
            return details
        except Exception as e:
            logger.error(f"Error extracting bureau decision details: {e}")
            return {
                "status": None,
                "reason": None,
                "maxEligibleEMI": None,
                "emiPlans": [],
                "creditLimitCalculated": None,
                "loanAmount": None,
                "maxTreatmentAmount": None
            }

    def process_prefill_data_for_basic_details(self, session_id: str) -> str:
        """
        Process prefill data and check for missing details. If any required details are missing,
        always save the available details using the API client, and return a message asking the user to provide the missing ones.

        Args:
            session_id: Session identifier.

        Returns:
            JSON string for save_basic_details or message asking for missing details
        """
        try:
            # 1. Get user_id if not provided
            session = None
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
            user_id = session.get("data", {}).get("userId") if session else None
            if not user_id:
                return "User ID is required to process prefill data"

            # 2. Get prefill data from API response in session
            prefill_data = {}
            session_data = {}
            if session_id:
                session = SessionManager.get_session_from_db(session_id)
                if session and "data" in session:
                    session_data = session["data"]
                    api_responses = session_data.get("api_responses", {})
                    prefill_api_result = api_responses.get("get_prefill_data")
                    if prefill_api_result and isinstance(prefill_api_result, dict):
                        prefill_data = prefill_api_result.get("data", {}).get("response", {})
                    if not prefill_data and "prefill_api_response" in session_data:
                        prefill_data = session_data["prefill_api_response"]

            # 3. Build the data for save_basic_details
            data = {"userId": user_id, "formStatus": "Basic"}

            # 4. Get name and phone from session if available
            if "name" in session_data and session_data["name"] is not None:
                data["firstName"] = session_data["name"]
            elif "fullName" in session_data and session_data["fullName"] is not None:
                data["firstName"] = session_data["fullName"]

            if "phone" in session_data and session_data["phone"] is not None:
                data["mobileNumber"] = session_data["phone"]
            elif "phoneNumber" in session_data and session_data["phoneNumber"] is not None:
                data["mobileNumber"] = session_data["phoneNumber"]
            elif "mobileNumber" in session_data and session_data["mobileNumber"] is not None:
                data["mobileNumber"] = session_data["mobileNumber"]

            # 5. Extract fields from prefill_data (from API response)
            field_mappings = {
                "panCard": ["pan"],
                "gender": ["gender"],
                "dateOfBirth": ["dob"],
                "emailId": ["email"],
            }

            for target_field, source_fields in field_mappings.items():
                for source in source_fields:
                    if source in prefill_data and prefill_data[source] is not None:
                        value = prefill_data[source]
                        if isinstance(value, (dict, list)):
                            continue
                        else:
                            data[target_field] = str(value)
                        break

            # Special handling for email if it's a list or dict
            if "email" in prefill_data and prefill_data["email"] is not None and "emailId" not in data:
                email_data = prefill_data["email"]
                if isinstance(email_data, list) and email_data:
                    if isinstance(email_data[0], dict) and "email" in email_data[0] and email_data[0]["email"] is not None:
                        data["emailId"] = email_data[0]["email"]
                    else:
                        data["emailId"] = email_data[0]
                elif isinstance(email_data, str) and email_data.strip():
                    data["emailId"] = email_data

            # Also extract from nested "response" if it exists
            if "response" in prefill_data and isinstance(prefill_data["response"], dict):
                response = prefill_data["response"]
                for target_field, source_fields in field_mappings.items():
                    for source in source_fields:
                        if source in response and response[source] is not None and target_field not in data:
                            value = response[source]
                            if isinstance(value, (dict, list)):
                                continue
                            else:
                                data[target_field] = str(value)
                            break
                # Special handling for email in nested response
                if "email" in response and response["email"] is not None and "emailId" not in data:
                    email_data = response["email"]
                    if isinstance(email_data, list):
                        if email_data:
                            if isinstance(email_data[0], dict) and "email" in email_data[0] and email_data[0]["email"] is not None:
                                data["emailId"] = str(email_data[0]["email"])
                            else:
                                data["emailId"] = str(email_data[0])
                    elif isinstance(email_data, dict) and "email" in email_data and email_data["email"] is not None:
                        data["emailId"] = str(email_data["email"])
                    elif isinstance(email_data, str) and email_data.strip():
                        data["emailId"] = str(email_data)
                # Handle phone number in response if needed
                if "mobile" in response and response["mobile"] is not None and "mobileNumber" not in data:
                    data["mobileNumber"] = response["mobile"]

            # 6. Check for missing required details
            missing_details = []
            required_fields = ["panCard", "gender", "dateOfBirth"]
            
            for field in required_fields:
                if field not in data or not data[field] or data[field].strip() == "":
                    missing_details.append(field)

            # Always save the available details, even if some are missing
            logger.info(f"Saving available prefill details: user_id={user_id}, data={data}")
            result = self.api_client.save_prefill_details(user_id, data)
            logger.info(f"Saved (partial) prefill details: {result}")
            if session_id:
                SessionManager.update_session_data_field(session_id, "data.api_responses.save_prefill_details", result)
                # Only mark as completed if nothing is missing
                if not missing_details:
                    SessionManager.update_session_data_field(session_id, "data.basic_details_completed", True)

            # If there are missing details, ask the user to provide them
            if missing_details:
                missing_messages = []
                for field in missing_details:
                    if field == "panCard":
                        missing_messages.append("PAN card number")
                    elif field == "gender":
                        missing_messages.append("gender (Male/Female/Other)")
                    elif field == "dateOfBirth":
                        missing_messages.append("date of birth (YYYY-MM-DD format)")
                    # elif field == "emailId":
                    #     missing_messages.append("email address")
                
                missing_text = ", ".join(missing_messages)
                response_message = f"I need some additional information to complete Patient's application. Please provide Patient's {missing_text}."
                
                # Store the missing details in session for the agent to handle
                if session_id:
                    SessionManager.update_session_data_field(session_id, "data.missing_details", missing_details)
                    SessionManager.update_session_data_field(session_id, "data.prefill_data_processed", data)
                    logger.info(f"Missing details detected: {missing_details}")
                
                return json.dumps({
                    "status": "missing_details",
                    "message": response_message,
                    "missing_details": missing_details,
                    "available_data": data,
                    "save_result": result
                })

            # All details are available, return the save result
            logger.info(f"All basic details present and saved for user_id={user_id}")
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
        If pincode is missing, returns a special status to ask user for pincode.
        If state is not found from API, use state from prefill data, but if it's a code, crosswalk to real state name using pincode.

        Args:
            session_id: Session identifier

        Returns:
            Save result as JSON string
        """
        user_id = None  # Ensure user_id is always defined

        # State code to state name mapping with pincode ranges
        PINCODE_STATE_MAP = [
            ("Andaman & Nicobar Islands", ["744"]),
            ("Andhra Pradesh", [str(i) for i in range(500, 536)]),
            ("Arunachal Pradesh", [str(i) for i in range(790, 793)]),
            ("Assam", [str(i) for i in range(781, 789)]),
            ("Bihar", [str(i) for i in range(800, 856)]),
            ("Chhattisgarh", [str(i) for i in range(490, 498)]),
            ("Chandigarh", ["160"]),
            ("Daman & Diu", ["362", "396"]),
            ("Delhi", ["110"]),
            ("Dadra & Nagar Haveli", ["396"]),
            ("Goa", ["403"]),
            ("Gujarat", [str(i) for i in range(360, 397)]),
            ("Himachal Pradesh", [str(i) for i in range(171, 178)]),
            ("Haryana", [str(i) for i in range(121, 137)]),
            ("Jharkhand", [str(i) for i in range(813, 836)]),
            ("Jammu & Kashmir", [str(i) for i in range(180, 195)]),
            ("Karnataka", [str(i) for i in range(560, 592)]),
            ("Kerala", [str(i) for i in range(670, 696)]),
            ("Lakshadweep", ["682"]),
            ("Maharashtra", [str(i) for i in range(400, 446)]),
            ("Meghalaya", [str(i) for i in range(793, 795)]),
            ("Manipur", ["795"]),
            ("Madhya Pradesh", [str(i) for i in range(450, 489)]),
            ("Mizoram", ["796"]),
            ("Nagaland", ["797", "798"]),
            ("Odisha", [str(i) for i in range(751, 771)]),
            ("Punjab", [str(i) for i in range(140, 161)]),
            ("Pondicherry/Puducherry", ["533", "605", "607", "609"]),
            ("Rajasthan", [str(i) for i in range(301, 346)]),
            ("Sikkim", ["737"]),
            ("Telangana", [str(i) for i in range(500, 510)]),
            ("Tamil Nadu", [str(i) for i in range(600, 644)]),
            ("Tripura", ["799"]),
            ("Uttarakhand", [str(i) for i in range(244, 264)]),
            ("Uttar Pradesh", [str(i) for i in range(201, 286)]),
            ("West Bengal", [str(i) for i in range(700, 744)]),
        ]

        STATE_CODE_TO_NAME = {
            "AN": "Andaman & Nicobar Islands",
            "AP": "Andhra Pradesh",
            "AR": "Arunachal Pradesh",
            "AS": "Assam",
            "BR": "Bihar",
            "CG": "Chhattisgarh",
            "CT": "Chhattisgarh",
            "CH": "Chandigarh",
            "DD": "Daman & Diu",
            "DL": "Delhi",
            "DN": "Dadra & Nagar Haveli",
            "GA": "Goa",
            "GJ": "Gujarat",
            "HP": "Himachal Pradesh",
            "HR": "Haryana",
            "JH": "Jharkhand",
            "JK": "Jammu & Kashmir",
            "KA": "Karnataka",
            "KL": "Kerala",
            "LD": "Lakshadweep",
            "MH": "Maharashtra",
            "ML": "Meghalaya",
            "MN": "Manipur",
            "MP": "Madhya Pradesh",
            "MZ": "Mizoram",
            "NL": "Nagaland",
            "OR": "Odisha",
            "PB": "Punjab",
            "PY": "Pondicherry/Puducherry",
            "RJ": "Rajasthan",
            "SK": "Sikkim",
            "TG": "Telangana",
            "TS": "Telangana",
            "TN": "Tamil Nadu",
            "TR": "Tripura",
            "UL": "Uttarakhand",
            "UP": "Uttar Pradesh",
            "WB": "West Bengal",
        }

        def get_state_from_pincode(pincode):
            if not pincode or len(pincode) < 3:
                return None
            prefix = pincode[:3]
            for state_name, ranges in PINCODE_STATE_MAP:
                for rng in ranges:
                    if "-" in rng:
                        start, end = rng.split("-")
                        if start.strip().isdigit() and end.strip().isdigit():
                            if int(start) <= int(prefix) <= int(end):
                                return state_name
                    else:
                        if prefix == rng:
                            return state_name
            return None

        def is_valid_pincode(pincode):
            """Check if a string is a valid 6-digit pincode"""
            if not pincode:
                return False
            # Clean the pincode string
            clean_pincode = ''.join(filter(str.isdigit, str(pincode)))
            return len(clean_pincode) == 6 and clean_pincode.isdigit()

        def extract_pincode_from_postal(postal):
            """Extract valid pincode from postal field"""
            if not postal:
                return None
            # Clean the postal string and extract digits
            clean_postal = ''.join(filter(str.isdigit, str(postal)))
            if len(clean_postal) == 6:
                return clean_postal
            # If we have more than 6 digits, try to find 6-digit sequence
            if len(clean_postal) > 6:
                for i in range(len(clean_postal) - 5):
                    potential_pincode = clean_postal[i:i+6]
                    if potential_pincode.isdigit():
                        return potential_pincode
            return None

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
                valid_pincode = None

                # First, try to find address with Type "Primary" or "Permanent"
                for addr in address_list:
                    addr_type = addr.get("Type", "").lower()
                    if addr_type in ["primary", "permanent"]:
                        primary_address = addr
                        # Check if this address has a valid pincode
                        postal = addr.get("Postal", "")
                        extracted_pincode = extract_pincode_from_postal(postal)
                        if is_valid_pincode(extracted_pincode):
                            valid_pincode = extracted_pincode
                            break

                # If no primary address with valid pincode found, search all addresses for valid pincode
                if not valid_pincode:
                    for addr in address_list:
                        postal = addr.get("Postal", "")
                        extracted_pincode = extract_pincode_from_postal(postal)
                        if is_valid_pincode(extracted_pincode):
                            primary_address = addr
                            valid_pincode = extracted_pincode
                            break

                # If still no valid pincode found, use the first address in the list
                if not primary_address and address_list:
                    primary_address = address_list[0]

                if primary_address:
                    # Extract address details
                    address_data["address"] = primary_address.get("Address", "")
                    
                    # Use the valid pincode we found, or try to extract from current address
                    if valid_pincode:
                        address_data["pincode"] = valid_pincode
                    else:
                        postal = primary_address.get("Postal", "")
                        extracted_pincode = extract_pincode_from_postal(postal)
                        address_data["pincode"] = extracted_pincode if extracted_pincode else ""

                    address_data["state"] = primary_address.get("State", "")

                    # Check if pincode is missing or invalid
                    if not address_data["pincode"] or not is_valid_pincode(address_data["pincode"]):
                        # Return special status to ask for pincode
                        return json.dumps({
                            "status": "missing_pincode",
                            "message": "Please provide your 6-digit pincode to continue with the loan application process. Follow workflow A",
                            "extracted_address_data": address_data
                        })

                    # Clean up the pincode
                    pincode = address_data["pincode"].strip()
                    address_data["pincode"] = pincode

                    # If we have a valid pincode, get city and state from API
                    try:
                        pincode_data = self.api_client.state_and_city_by_pincode(address_data["pincode"])
                        logger.info(f"Pincode API response for pincode {address_data['pincode']}: {pincode_data}")
                        city_set = False
                        state_set = False
                        if pincode_data and pincode_data.get("status") == "success":
                            # Only update if we get valid non-null data
                            if pincode_data.get("city") and pincode_data["city"] is not None:
                                address_data["city"] = pincode_data["city"]
                                city_set = True
                            if pincode_data.get("state") and pincode_data["state"] is not None:
                                address_data["state"] = pincode_data["state"]
                                state_set = True
                        # If state is not set from API, use state from prefill data, but crosswalk code if needed
                        if not state_set:
                            prefill_state = primary_address.get("State", "")
                            # If prefill_state is a code, map to name
                            state_name = STATE_CODE_TO_NAME.get(prefill_state.strip().upper())
                            if state_name:
                                address_data["state"] = state_name
                            else:
                                # If not a code, try to use as is, but if still not a valid state, use pincode mapping
                                if prefill_state and len(prefill_state) <= 3:
                                    # Try pincode mapping
                                    state_from_pin = get_state_from_pincode(address_data["pincode"])
                                    if state_from_pin:
                                        address_data["state"] = state_from_pin
                                    else:
                                        address_data["state"] = prefill_state
                                elif prefill_state:
                                    address_data["state"] = prefill_state
                                else:
                                    # As last resort, use pincode mapping
                                    state_from_pin = get_state_from_pincode(address_data["pincode"])
                                    if state_from_pin:
                                        address_data["state"] = state_from_pin
                        # If city is not set from API, use last word of address as city
                        if not city_set:
                            address_str = address_data.get("address", "")
                            if address_str:
                                # Split address by whitespace and take last word
                                address_words = address_str.strip().split()
                                if address_words:
                                    # Save city in title case
                                    address_data["city"] = address_words[-1].title()
                    except Exception as e:
                        logger.warning(f"Failed to get city/state from pincode API: {e}")
                        # If API call fails, try to set city from address as fallback
                        address_str = address_data.get("address", "")
                        if address_str:
                            address_words = address_str.strip().split()
                            if address_words:
                                address_data["city"] = address_words[-1].title()
                        # For state, use prefill state or pincode mapping
                        prefill_state = primary_address.get("State", "")
                        state_name = STATE_CODE_TO_NAME.get(prefill_state.strip().upper())
                        if state_name:
                            address_data["state"] = state_name
                        else:
                            state_from_pin = get_state_from_pincode(address_data["pincode"])
                            if state_from_pin:
                                address_data["state"] = state_from_pin
                            elif prefill_state:
                                address_data["state"] = prefill_state

                logger.info(f"Extracted address data: {address_data}")

                # Store the extracted address data in session
                if session_id:
                    SessionManager.update_session_data_field(session_id, "data.extracted_address_data", address_data)

                # Save the address details
                result = self.api_client.save_address_details(user_id, address_data)
                permanent_result = self.api_client.save_permanent_address_details(user_id, address_data)
                logger.info(f"Permanent address details saved: {permanent_result}")

                # Store the API response
                if session_id:
                    SessionManager.update_session_data_field(session_id, "data.api_responses.process_address_data", result)

                return json.dumps(result)
            else:
                # No address found in prefill data, ask for pincode
                return json.dumps({
                    "status": "missing_pincode",
                    "message": "Please provide your 6-digit pincode to continue with the loan application process. Follow workflow A",
                    "extracted_address_data": {}
                })

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
            # Helper function to detect limit options prompt
            def is_limit_options_prompt(text: str) -> bool:
                return (
                    "Continue with this limit" in text
                    and "Continue with limit enhancement" in text
                    and ("1." in text and "2." in text)
                )
            
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
            
            # If no collection step is set, check if we should start with limit options
            if not collection_step:
                # Check if the session has limit options in the history
                session_history = session.get("history", [])
                for hist_item in reversed(session_history[-5:]):  # Check last 5 messages
                    if isinstance(hist_item, dict) and hist_item.get("type") == "AIMessage":
                        content = hist_item.get("content", "")
                        if is_limit_options_prompt(content):
                            collection_step = "limit_options"
                            SessionManager.update_session_data_field(session_id, "data.collection_step", "limit_options")
                            logger.info(f"Session {session_id}: Detected limit options in history, setting collection step to limit_options")
                            break
            
            # Log current step for debugging
            logger.info(f"Session {session_id}: Processing step '{collection_step}' with message: {message.strip()}")
            logger.info(f"Session {session_id}: Current collection step from session data: {session['data'].get('collection_step', 'not_set')}")
            
            # Function to save the current collection step and refresh session
            def update_collection_step(new_step):
                # Use update_session_data_field to preserve existing data
                SessionManager.update_session_data_field(session_id, "data.collection_step", new_step)
                SessionManager.update_session_data_field(session_id, "status", "collecting_additional_details")
                logger.info(f"Session {session_id}: Updated collection step to '{new_step}'")
            
            # Handle limit options input (first step when limit options are presented)
            if collection_step == "limit_options":
                # Check for both number and word inputs
                message_lower = message.lower().strip()
                message_stripped = message.strip()
                
                if (message_stripped == "1" or 
                    "continue with this limit" in message_lower or 
                    "this limit" in message_lower):
                    additional_details["limit_choice"] = "continue_with_limit"
                    selected_option = "Continue with this limit"
                    logger.info(f"Limit choice input: message='{message}', stored_value='continue_with_limit', selected_option='{selected_option}'")
                elif (message_stripped == "2" or 
                      "continue with limit enhancement" in message_lower or 
                      "limit enhancement" in message_lower or 
                      "enhancement" in message_lower):
                    additional_details["limit_choice"] = "continue_with_enhancement"
                    selected_option = "Continue with limit enhancement"
                    logger.info(f"Limit choice input: message='{message}', stored_value='continue_with_enhancement', selected_option='{selected_option}'")
                else:
                    return "Please select a valid option: 1. Continue with this limit or 2. Continue with limit enhancement"
                
                # Update session data with limit choice using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step and ask for employment type
                update_collection_step("employment_type")
                return f"""

To proceed, please help me with a few more details.

Patient's employment type:   
1. SALARIED
2. SELF_EMPLOYED
Please Enter input 1 or 2 only"""

            # Handle employment type input (first step)
            elif collection_step == "employment_type":
                # Check for both number and word inputs
                if "1" in message or "salaried" in message.lower():
                    additional_details["employment_type"] = "SALARIED"
                    selected_option = "SALARIED"
                elif "2" in message or "self" in message.lower() and "employed" in message.lower():
                    additional_details["employment_type"] = "SELF_EMPLOYED"
                    selected_option = "SELF_EMPLOYED"
                else:
                    return "Please select a valid option for Employment Type: 1. SALARIED or 2. SELF_EMPLOYED"
                
                # Update session data with employment type using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step and ask for marital status
                update_collection_step("marital_status")
                return f"""

Patient's marital status:
1. Married
2. Unmarried/Single\n
Please Enter input 1 or 2 only"""
            
            # Handle marital status input
            elif collection_step == "marital_status":
                # Check for both number and word inputs
                message_lower = message.lower().strip()
                
                # Check for exact number matches first
                if message.strip() == "1" or message_lower == "married":
                    additional_details["marital_status"] = "1"
                    selected_option = "Married"
                    logger.info(f"Marital status input: message='{message}', stored_value='1', selected_option='{selected_option}'")
                elif message.strip() == "2" or message_lower in ["unmarried", "single", "unmarried/single"]:
                    additional_details["marital_status"] = "2"
                    selected_option = "Unmarried/Single"
                    logger.info(f"Marital status input: message='{message}', stored_value='2', selected_option='{selected_option}'")
                else:
                    return "Please select a valid option for Marital Status: 1. Married or 2. Unmarried/Single"
                
                # Update session data with marital status using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step and ask for education qualification
                update_collection_step("education_qualification")
                return f"""
Patient's education qualification: 
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
                
                # Check for both number and word inputs
                selected_key = None
                message_lower = message.lower().strip()
                
                # First check if it's a number
                if message.strip() in education_options:
                    selected_key = message.strip()
                # Then check for word matches
                elif "less" in message_lower and "10th" in message_lower:
                    selected_key = "1"
                elif "passed 10th" in message_lower or "10th" in message_lower:
                    selected_key = "2"
                elif "passed 12th" in message_lower or "12th" in message_lower:
                    selected_key = "3"
                elif "diploma" in message_lower:
                    selected_key = "4"
                elif "graduation" in message_lower and "post" not in message_lower:
                    selected_key = "5"
                elif "post graduation" in message_lower or "postgraduation" in message_lower:
                    selected_key = "6"
                elif "phd" in message_lower or "p.h.d" in message_lower:
                    selected_key = "7"
                
                if selected_key:
                    additional_details["education_qualification"] = selected_key
                    selected_option = education_options[selected_key]
                else:
                    return "Please select a valid option for Education Qualification (1-7)"
                
                # Update session data with education qualification using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step and ask for treatment reason
                update_collection_step("treatment_reason")
                return f"""

What is the name of treatment?"""
            
            # Handle treatment reason input
            elif collection_step == "treatment_reason":
                additional_details["treatment_reason"] = message.strip()
                
                # Update session data with treatment reason using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)

                # Check if email was already saved during prefill data processing
                session = SessionManager.get_session_from_db(session_id)
                session_data = session.get("data", {}) if session else {}
                api_responses = session_data.get("api_responses", {})
                
                # Check if email was saved in prefill data processing
                prefill_save_result = api_responses.get("save_prefill_details")
                email_already_saved = False
                
                if prefill_save_result and isinstance(prefill_save_result, dict):
                    # Check if email was successfully saved in prefill processing
                    if prefill_save_result.get("status") == 200:
                        # Check if emailId is present in the saved data
                        saved_data = prefill_save_result.get("data", {})
                        email_value = saved_data.get("emailId")
                        if email_value and "@" in str(email_value):
                            email_already_saved = True
                            logger.info(f"Email already saved during prefill processing: {email_value}")
                
                if email_already_saved:
                    # Skip email collection, proceed directly to employment type check
                    logger.info("Email already saved during prefill processing, skipping email collection")
                    
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
                            return f"""

Patient's 6-digit workplace/office pincode"""
                        else:
                            # If not found, ask for organization name as usual
                            additional_details["organization_name"] = ""  # Initialize organization name
                            update_collection_step("organization_name")
                            return f"""

Organization Name where the patient works?"""
                    else:
                        additional_details["business_name"] = ""  # Initialize business name
                        update_collection_step("business_name")
                        return f"""

Business Name where the patient works?"""
                else:
                    # Email not saved during prefill, ask for it now
                    update_collection_step("email_address")
                    return f"""

Patient's email address"""
            
            # Handle email address input
            elif collection_step == "email_address":
                # Validate email format
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, message.strip()):
                    return "Please provide a valid email address."
                
                # Save email address using handle_email_address
                email_result = self.handle_email_address(message.strip(), session_id)
                
                # Parse the result
                if isinstance(email_result, str):
                    try:
                        email_result_data = json.loads(email_result)
                    except json.JSONDecodeError:
                        email_result_data = {"status": "error", "message": "Invalid response from email handler"}
                else:
                    email_result_data = email_result
                
                if email_result_data.get('status') == 'error':
                    return email_result_data.get('message', 'Failed to save email address. Please try again.')
                
                # Store email in additional details
                additional_details["email_address"] = message.strip()
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
                        return f"""

Patient's 6-digit workplace/office pincode"""
                    else:
                        # If not found, ask for organization name as usual
                        additional_details["organization_name"] = ""  # Initialize organization name
                        update_collection_step("organization_name")
                        return f"""

Organization Name where the patient works?"""
                else:
                    additional_details["business_name"] = ""  # Initialize business name
                    update_collection_step("business_name")
                    return f"""

Business Name where the patient works?"""
            
            # Handle organization name input (for SALARIED)
            elif collection_step == "organization_name":
                additional_details["organization_name"] = message.strip()
                
                # Update session data using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step to ask for workplace pincode
                update_collection_step("workplace_pincode")
                return f"""

Patient's 6-digit workplace/office pincode"""
            
            # Handle business name input (for SELF_EMPLOYED)
            elif collection_step == "business_name":
                additional_details["business_name"] = message.strip()
                
                # Update session data using update_session_data_field
                SessionManager.update_session_data_field(session_id, "data.additional_details", additional_details)
                
                # Update collection step to ask for workplace pincode
                update_collection_step("workplace_pincode")
                return f"""

Patient's 6-digit business location pincode"""

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

                # Get user ID from session data
                session = SessionManager.get_session_from_db(session_id)
                user_id = session.get("data", {}).get("userId", "") if session else ""
                
               
                link_to_display = fibe_link_to_display if fibe_link_to_display else profile_link


                # Use the new centralized decision logic
                decision_result = self._determine_loan_decision(session_id, profile_link, fibe_link_to_display)
                decision_status = decision_result["status"]
                link_to_display = decision_result["link"]
                is_bureau_approved = decision_result.get("is_bureau_approved", False)
                
                # Get patient name from session data
                patient_name = session.get("data", {}).get("fullName", "")
                if not patient_name:
                    # Try to get from ocr_result if fullName is not available
                    ocr_result = session.get("data", {}).get("ocr_result", {})
                    patient_name = ocr_result.get("name", "")
                
                if decision_status == "INCOME_VERIFICATION_REQUIRED":
                    return f"""Patient {patient_name} has a fair chance of approval, we need their last 3 months' bank statement to assess their application.

Upload bank statement by clicking on the link below.

{link_to_display}"""
                elif decision_status == "APPROVED":
                    if is_bureau_approved:
                        return f"""Great news! 🥳 Patient {patient_name} is **APPROVED** ✅ for a no-cost EMI payment plan.

You are just 4 steps away from the disbursal.

Continue with payment plan selection."""
                    else:
                        return f"""Great news! 🥳 Patient {patient_name} is **APPROVED** ✅ for a no-cost EMI payment plan.

You are just 4 steps away from the disbursal.

Continue on the link to complete the process and get the loan disbursed.

{link_to_display}"""
                else:
                    return f"""We regret to inform you that Patient {patient_name} is not eligible for the proposed loan amount.

{patient_name} can try financing their treatment via No-Cost Credit & Debit Card EMI or someone from their immediate family can apply on their behalf.

CTA -

No-cost Credit & Debit Card EMI

Re-enquire with your family member's details."""
                
        except Exception as e:
            logger.error(f"Error handling additional details collection: {e}")
            return "There was an error processing Patient's information. Please try again."

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

    def _handle_post_approval_address_details(self, session_id: str, message: str) -> str:
        """
        Handle post-approval address details flow and KYC status transition
        
        Args:
            session_id: Session identifier
            message: User message
            
        Returns:
            Response message with payment plan details and KYC link
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return "Session not found. Please start a new conversation."
            
            
            response_message = f"""
Kindly confirm patient's address details by clicking below buttom.

"""
            
            # Update status to KYC pending
            SessionManager.update_session_data_field(session_id, "status", "post_approval_address_details")
            SessionManager.update_session_data_field(session_id, "data.post_approval_address_details", datetime.now().isoformat())
            
            logger.info(f"Session {session_id}: Updated status to post_approval_address_details_completed and provided post-approval address details link")
            
            return response_message
            
        except Exception as e:
            logger.error(f"Error handling post-approval address details: {e}")
            return "There was an error processing your request. Please try again."

    def _handle_address_details_completion(self, session_id: str, message: str) -> str:
        """
        Handle address details completion and provide next steps with URLs
        
        Args:
            session_id: Session identifier
            message: User message
            
        Returns:
            Response message with face verification, EMI auto-pay, and agreement e-signing links
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return "Session not found. Please start a new conversation."
            
            # Check if user message indicates address details are complete
            if message.lower().strip() == "address details complete":
                # Get session data to construct URLs with session ID
                session_data = session.get("data", {})
                user_id = session_data.get("userId", "")
                
                # Get loan ID from save_loan_details in session data
                loan_id = ""
                if "api_responses" in session_data and "save_loan_details" in session_data["api_responses"]:
                    save_loan_response = session_data["api_responses"]["save_loan_details"]
                    logger.info(f"save_loan_details response: {save_loan_response}")
                    if isinstance(save_loan_response, dict) and save_loan_response.get("status") == 200:
                        if "data" in save_loan_response and isinstance(save_loan_response["data"], dict):
                            loan_id = save_loan_response["data"].get("loanId")
                            logger.info(f"Found loan_id in save_loan_details response: {loan_id}")
                
                logger.info(f"Session {session_id}: Retrieved loanId: {loan_id}, userId: {user_id}")

                digilocker_response = self.api_client.create_digilocker_url(loan_id)
                
                # Extract DigiLocker URL from response
                adhaar_verification_url = ""
                if digilocker_response and digilocker_response.get("status") == 200:
                    adhaar_verification_url = digilocker_response.get("data", "")
                    logger.info(f"Session {session_id}: Retrieved DigiLocker URL: {adhaar_verification_url}")
                else:
                    logger.error(f"Session {session_id}: Failed to get DigiLocker URL. Response: {digilocker_response}")
                
                # Construct the URLs with proper loan ID and user ID - ensure loanId is not empty
                face_verification_url = f"https://carepay.money/patient/faceverification/{user_id}" if user_id else "https://carepay.money/patient/faceverification/"
                emi_autopay_url = f"https://carepay.money/patient/emiautopayintro/{loan_id}" if loan_id else "https://carepay.money/patient/emiautopayintro/"
                agreement_esigning_url = f"https://carepay.money/patient/agreementesigning/{loan_id}" if loan_id else "https://carepay.money/patient/agreementesigning/"
                
                logger.info(f"Session {session_id}: Constructed URLs - Face: {face_verification_url}, EMI: {emi_autopay_url}, Agreement: {agreement_esigning_url}")
                
                # Create response with three different messages and URLs
                response_message = f"""Payment is now just 4 steps away.

• Adhaar verification.
• Face verification.
• EMI auto payment approval.
• Agreement e-signing.

Now, let's complete Adhaar verification.

[Adhaar Verification]{adhaar_verification_url}

Now, let's complete face verification.

[Face Verification]{face_verification_url}

Approve the EMI auto-pay setup.

[EMI Auto-pay Setup]{emi_autopay_url}

E-sign agreement using this link.

[Agreement E-signing]{agreement_esigning_url}"""
                
                # Update status to kyc_step
                SessionManager.update_session_data_field(session_id, "status", "kyc_step")
                SessionManager.update_session_data_field(session_id, "data.address_details_completed", datetime.now().isoformat())
                
                logger.info(f"Session {session_id}: Address details completed, status updated to kyc_step")
                
                return response_message
            else:
                # If message is not "address details complete", provide guidance
                return "Please confirm that address details are complete by typing 'address details complete'."
                
        except Exception as e:
            logger.error(f"Error handling address details completion: {e}")
            return "There was an error processing your request. Please try again."


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
                logger.info(f"Processing marital status: raw_value='{additional_details['marital_status']}', mapped_value='{marital_status_map.get(additional_details['marital_status'], additional_details['marital_status'])}'")
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
        logger.info(f"Creating session-aware tools for session_id: {session_id}")
        tools = [
            StructuredTool.from_function(
                func=lambda fullName, phoneNumber, treatmentCost, monthlyIncome: self.store_user_data_structured(fullName, phoneNumber, treatmentCost, monthlyIncome, session_id),
                name="store_user_data",
                description="Store user data in session with the four parameters: fullName, phoneNumber, treatmentCost, monthlyIncome",
            ),
            Tool(
                name="get_user_id_from_phone_number",
                func=lambda phone_number: self.get_user_id_from_phone_number(phone_number, session_id),
                description="Get userId from response of API call get_user_id_from_phone_number",
            ),
            Tool(
                name="save_basic_details",
                func=lambda _: self.save_basic_details(session_id),
                description="Save user's basic personal details. Call this tool using session_id ",
            ),
            StructuredTool.from_function(
                func=lambda fullName, treatmentCost, userId: self.save_loan_details_structured(fullName, treatmentCost, userId, session_id),
                name="save_loan_details",
                description="Save user's loan details with fullName, treatmentCost, and userId parameters.",
            ),
            Tool(
                name="check_jp_cardless",
                func=lambda _: self.check_jp_cardless(session_id),
                description="Check eligibility for Juspay Cardless",
            ),
            Tool(
                name="get_prefill_data",
                func=lambda user_id=None: self.get_prefill_data(user_id, session_id),
                description="Get prefilled user data from user ID",
            ),
            
             Tool(
                name="process_prefill_data",
                func=lambda _: self.process_prefill_data_for_basic_details(session_id),
                description="Convert prefill data from get_prefill_data_for_basic_details to a properly formatted JSON for save_basic_details. Call this tool using session_id.",
            ),
            Tool(
                name="process_address_data",
                func=lambda _: self.process_address_data(session_id),
                description="Extract address information from prefill data and save it using save_address_details. Call this after process_prefill_data. Must include session_id parameter.",
            ),
            Tool(
                name="pan_verification",
                func=lambda _: self.pan_verification(session_id),
                description="Verify PAN details for a user using session_id. Call this tool using session_id",
            ),
            Tool(
                name="get_employment_verification",
                func=lambda _: self.get_employment_verification(session_id),
                description="Get employment verification data using session_id",
            ),
           
            Tool(
                name="save_employment_details",
                func=lambda _: self.save_employment_details(session_id),
                description="Save user's employment details using session_id",
            ),
            

            Tool(
                name="get_bureau_decision",
                func=lambda _: self.get_bureau_decision(session_id),
                description="Get bureau decision for loan application using session_id. CRITICAL: The response from this tool is the FINAL formatted message that MUST be returned to the user EXACTLY as provided without any modifications.   ",
            ),
           
            
            Tool(
                name="get_profile_link",
                func=lambda _: self._get_profile_link(session_id),
                description="Get profile link for a user using session_id",
            ),
            Tool(
                name="handle_pan_card_number",
                func=lambda pan_number: self.handle_pan_card_number(pan_number, session_id),
                description="Handle PAN card number input and save it to the system. Use this when user provides their PAN card number.",
            ),
            
            StructuredTool.from_function(
                func=lambda pincode: self.save_missing_basic_and_address_details(
                      pincode, session_id
                ),
                name="save_missing_basic_and_address_details",
                description=(
                    "Save address details when pincode is provided. Use this in two scenarios: "
                    "1. When user needs to provide pincode (Workflow B) - collect 6-digit pincode. "
                    "2. When process_address_data returns missing/invalid pincode (Workflow A) - user provides 6-digit pincode. "
                    "Call this tool immediately when user provides a valid 6-digit pincode."
                ),
            ),
            Tool(
                name="handle_email_address",
                func=lambda email_address: self.handle_email_address(email_address, session_id),
                description="Handle email address input and save it to the system. Use this when user provides their email address.",
            ),
            Tool(
                name="save_gender_details",
                func=lambda gender: self.save_gender_details(gender, session_id),
                description="Save user's gender details. Use this when user provides their gender information like 'Male', 'Female', '1', or '2'. Call this tool immediately when user provides gender selection.",
            ),
            Tool(
                name="save_marital_status_details",
                func=lambda marital_status: self.save_marital_status_details(marital_status, session_id),
                description="Save user's marital status details. Use this when user provides their marital status information like 'Married', 'Unmarried/Single', 'Yes', 'No', '1', or '2'. The system will automatically format it to the correct API format (Yes/No). Call this tool immediately when user provides marital status selection.",
            ),
            Tool(
                name="save_education_level_details",
                func=lambda education_level: self.save_education_level_details(education_level, session_id),
                description="Save user's education level details. Use this when user provides their education level information like 'P.H.D', 'Graduation', 'Post graduation', 'Diploma', 'Passed 12th', 'Passed 10th', 'Less than 10th', or numbers 1-7. The system will automatically format it to the correct API format (LESS THAN 10TH, PASSED 10TH, etc.). Call this tool immediately when user provides education level selection.",
            ),
            Tool(
                name="correct_treatment_reason",
                func=lambda new_treatment_reason: self.correct_treatment_name(new_treatment_reason, session_id),
                description="Correct/update the treatment reason in the loan application. Use this when user provides a new treatment reason like 'hair transplant', 'dental surgery', etc. Call this tool immediately when user provides a treatment reason.",
            ),
            Tool(
                name="correct_treatment_cost",
                func=lambda new_treatment_cost: self.correct_treatment_cost(new_treatment_cost, session_id),
                description="Correct/update the treatment cost in the loan application. Use this when user provides a new treatment cost like '5000', '10000', '90000', etc. (must be >= ₹3,000 and <= ₹10,00,000). Call this tool immediately when user provides a numeric treatment cost amount.",
            ),
            Tool(
                name="correct_date_of_birth",
                func=lambda new_date_of_birth: self.correct_date_of_birth(new_date_of_birth, session_id),
                description="Correct/update the date of birth in the user profile. Use this when user wants to change their date of birth (format: DD-MM-YYYY).",
            ),
            Tool(
                name="save_gender_B_details",
                func=lambda gender: self.save_gender_B_details(gender, session_id),
                description="Save user's gender details. Use this when user provides their gender information like 'Male', 'Female', '1', or '2'. Call this tool immediately when user provides gender selection.",
            ),
        ]
        logger.info(f"Created {len(tools)} tools for session {session_id}")
        return tools

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
            fibe_lead_status = None
            bureau_status = None
            
            # Check for profile ingestion 500 error
            if profile_ingestion and profile_ingestion.get("status") == 500:
                logger.info(f"Session {session_id}: Profile ingestion returned 500 error - treating as RED status")
                fibe_status = "RED"
            # Extract Fibe status if no 500 error
            elif check_fibe_flow and check_fibe_flow.get("status") == 200:
                fibe_status = check_fibe_flow.get("data")
                logger.info(f"Session {session_id}: Fibe status: {fibe_status}")
            
            # Extract FIBE lead status from profile ingestion response
            if profile_ingestion and profile_ingestion.get("status") == 200:
                ingestion_data = profile_ingestion.get("data", {})
                if isinstance(ingestion_data, dict):
                    fibe_lead_status = ingestion_data.get("leadStatus")
                    logger.info(f"Session {session_id}: FIBE lead status from profile ingestion: {fibe_lead_status}")
            
            # Extract bureau status
            if bureau_decision:
                bureau_status = bureau_decision.get("status")
                logger.info(f"Session {session_id}: Bureau status: {bureau_status} (type: {type(bureau_status)})")
                logger.info(f"Session {session_id}: Full bureau decision: {bureau_decision}")
                
                # Debug: Check exact string matching
                if bureau_status:
                    logger.info(f"Session {session_id}: Bureau status checks:")
                    logger.info(f"  - Exact match 'INCOME_VERIFICATION_REQUIRED': {bureau_status == 'INCOME_VERIFICATION_REQUIRED'}")
                    logger.info(f"  - Upper case match: {bureau_status.upper() == 'INCOME_VERIFICATION_REQUIRED'}")
                    logger.info(f"  - Contains 'income verification required': {'income verification required' in bureau_status.lower()}")
                    logger.info(f"  - Raw status value: '{bureau_status}'")
            else:
                logger.warning(f"Session {session_id}: No bureau decision found in session data")
                logger.info(f"Session {session_id}: Available session data keys: {list(session['data'].keys()) if 'data' in session else 'No data'}")
                logger.info(f"Session {session_id}: API responses keys: {list(api_responses.keys()) if api_responses else 'No API responses'}")
                
                # Check if bureau decision is stored in api_responses
                api_bureau_decision = api_responses.get("get_bureau_decision")
                if api_bureau_decision:
                    logger.info(f"Session {session_id}: Found bureau decision in api_responses: {api_bureau_decision}")
                    # Try to extract and save it
                    if isinstance(api_bureau_decision, dict) and api_bureau_decision.get("status") == 200:
                        extracted_bureau = self.extract_bureau_decision_details(api_bureau_decision, session_id)
                        SessionManager.update_session_data_field(session_id, "data.bureau_decision_details", extracted_bureau)
                        logger.info(f"Session {session_id}: Extracted and saved bureau decision from api_responses")
                        bureau_decision = extracted_bureau
                        bureau_status = bureau_decision.get("status")
            
            # Apply decision flow logic
            decision_status = None
            link_to_use = profile_link
            is_bureau_approved = False  # Track if approval came from bureau decision
            
            # 0. If both FIBE lead status and Bureau are REJECTED -> REJECTED
            if (fibe_lead_status and fibe_lead_status.upper() == "REJECTED" and 
                bureau_status and (bureau_status.upper() == "REJECTED" or "rejected" in bureau_status.lower())):
                decision_status = "REJECTED"
                link_to_use = profile_link
                logger.info(f"Session {session_id}: FIBE lead status REJECTED + Bureau REJECTED -> REJECTED")
            
            # 1. If Fibe GREEN -> APPROVED with Fibe link
            elif fibe_status == "GREEN":
                decision_status = "APPROVED"
                link_to_use = fibe_link if fibe_link else profile_link
                is_bureau_approved = False  # This is FIBE approval, not bureau
                logger.info(f"Session {session_id}: Fibe GREEN -> APPROVED with Fibe link")
            
            # 2. If Fibe AMBER
            elif fibe_status == "AMBER":
                # If bureau APPROVED -> APPROVED with profile link
                if bureau_status and (bureau_status.upper() == "APPROVED" or "approved" in bureau_status.lower()):
                    decision_status = "APPROVED"
                    link_to_use = profile_link
                    is_bureau_approved = True  # This approval came from bureau decision
                    logger.info(f"Session {session_id}: Fibe AMBER + Bureau APPROVED -> APPROVED with profile link")
                # If bureau INCOME_VERIFICATION_REQUIRED -> INCOME_VERIFICATION_REQUIRED with Fibe link
                elif bureau_status and (bureau_status.upper() == "INCOME_VERIFICATION_REQUIRED" or "income verification required" in bureau_status.lower()):
                    decision_status = "INCOME_VERIFICATION_REQUIRED"
                    link_to_use = fibe_link if fibe_link else profile_link
                    logger.info(f"Session {session_id}: Fibe AMBER + Bureau INCOME_VERIFICATION_REQUIRED -> INCOME_VERIFICATION_REQUIRED with Fibe link")
                    logger.info(f"Session {session_id}: Matched INCOME_VERIFICATION_REQUIRED condition")
                # If bureau REJECTED -> INCOME_VERIFICATION_REQUIRED with Fibe link (only if FIBE lead status is not REJECTED)
                elif bureau_status and (bureau_status.upper() == "REJECTED" or "rejected" in bureau_status.lower()):
                    # Only apply this rule if FIBE lead status is not REJECTED
                    if not fibe_lead_status or fibe_lead_status.upper() != "REJECTED":
                        decision_status = "INCOME_VERIFICATION_REQUIRED"
                        link_to_use = fibe_link if fibe_link else profile_link
                        logger.info(f"Session {session_id}: Fibe AMBER + Bureau REJECTED -> INCOME_VERIFICATION_REQUIRED with Fibe link (FIBE lead status not REJECTED)")
                    else:
                        # If FIBE lead status is also REJECTED, this case should have been handled by the first condition
                        decision_status = "REJECTED"
                        link_to_use = profile_link
                        logger.info(f"Session {session_id}: Fibe AMBER + Bureau REJECTED + FIBE lead REJECTED -> REJECTED")
                # Otherwise -> INCOME_VERIFICATION_REQUIRED with Fibe link
                else:
                    decision_status = "INCOME_VERIFICATION_REQUIRED"
                    link_to_use = fibe_link if fibe_link else profile_link
                    logger.info(f"Session {session_id}: Fibe AMBER + Bureau not APPROVED -> INCOME_VERIFICATION_REQUIRED with Fibe link")
                    logger.info(f"Session {session_id}: Fell through to else condition - bureau_status: '{bureau_status}'")
            # 3. If Fibe RED or profile ingestion 500 error -> Fall back to bureau decision with profile link
            elif fibe_status == "RED":
                if bureau_status and (bureau_status.upper() == "APPROVED" or "approved" in bureau_status.lower()):
                    decision_status = "APPROVED"
                    is_bureau_approved = True  # This approval came from bureau decision
                elif bureau_status and (bureau_status.upper() == "REJECTED" or "rejected" in bureau_status.lower()):
                    decision_status = "REJECTED"
                elif bureau_status and (bureau_status.upper() == "INCOME_VERIFICATION_REQUIRED" or "income verification required" in bureau_status.lower()):
                    decision_status = "INCOME_VERIFICATION_REQUIRED"
                else:
                    decision_status = "PENDING"
                link_to_use = profile_link
                logger.info(f"Session {session_id}: Fibe RED or profile ingestion 500 error -> Using bureau decision ({bureau_status}) with profile link")
            
            # 4. If no Fibe status -> Use bureau decision with profile link
            elif fibe_status is None:
                if bureau_status and (bureau_status.upper() == "APPROVED" or "approved" in bureau_status.lower()):
                    decision_status = "APPROVED"
                    is_bureau_approved = True  # This approval came from bureau decision
                elif bureau_status and (bureau_status.upper() == "REJECTED" or "rejected" in bureau_status.lower()):
                    decision_status = "REJECTED"
                elif bureau_status and (bureau_status.upper() == "INCOME_VERIFICATION_REQUIRED" or "income verification required" in bureau_status.lower()):
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
                logger.info(f"Session {session_id}: Fell through to final PENDING condition - fibe_status: '{fibe_status}', bureau_status: '{bureau_status}'")
            
            logger.info(f"Session {session_id}: Final decision - Status: {decision_status}, Link: {link_to_use}, Bureau Approved: {is_bureau_approved}")
            logger.info(f"Session {session_id}: Decision logic summary - Fibe: {fibe_status}, FIBE Lead Status: {fibe_lead_status}, Bureau: {bureau_status}, Final: {decision_status}")
            
            return {
                "status": decision_status,
                "link": link_to_use,
                "is_bureau_approved": is_bureau_approved
            }
            
        except Exception as e:
            logger.error(f"Error determining loan decision for session {session_id}: {e}")
            return {"status": "PENDING", "link": profile_link, "is_bureau_approved": False}

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
                            formatted_response = f"""
🎉 Congratulations, {patient_name}! Patient's loan application has been **APPROVED** for Cardless EMI.\n\n

Continue your journey with the link here:\n\n
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
                    show_detailed_approval = cost_value > 100000
                except (ValueError, TypeError):
                    show_detailed_approval = False
            
            # Get status from bureau decision
            status = bureau_decision.get("status")
            logger.info(f"Bureau decision status: '{status}' (type: {type(status)})")
            
            # Format response based on status (case-insensitive)
            if status and status.upper() == "APPROVED":
                
                    max_treatment_amount = bureau_decision.get("maxTreatmentAmount", 0)
                    try:
                        max_treatment_amount = float(str(max_treatment_amount).replace(',', '').replace('₹', '')) if max_treatment_amount else 0
                    except (ValueError, TypeError):
                        max_treatment_amount = 0
                    
                    # Check if max_treatment_amount is greater than or equal to treatment_cost
                    if max_treatment_amount >= treatment_cost:
                        return f"""
🎉 Congratulations {patient_name} is eligible ✅ for a no-cost EMI
payment plan for amount up to ₹{max_treatment_amount:,.0f}

To proceed, please help me with a few more details.

Patient's employment type:   
1. SALARIED
2. SELF_EMPLOYED
Please Enter input 1 or 2 only"""
                    else:
                        return f"""
We were only able to approve payment plans
for a treatment amount up to
₹{max_treatment_amount:,.0f}

1. Continue with this limit
2. Continue with limit enhancement"""
            elif status and status.upper() == "REJECTED":
                # Check if doctor is mapped with FIBE
                doctor_id = session["data"].get("doctorId") or session["data"].get("doctor_id")
                doctor_mapped_with_fibe = False
                
                if doctor_id:
                    try:
                        if hasattr(self.api_client, 'check_doctor_mapped_by_nbfc'):
                            check_doctor_mapped_by_nbfc_response = self.api_client.check_doctor_mapped_by_nbfc(doctor_id)
                            logger.info(f"Session {session_id}: Check doctor mapped by FIBE response for REJECTED status - doctor_id {doctor_id}: {json.dumps(check_doctor_mapped_by_nbfc_response)}")
                            
                            if check_doctor_mapped_by_nbfc_response.get("status") == 200:
                                doctor_mapped_by_nbfc = check_doctor_mapped_by_nbfc_response.get("data")
                                doctor_mapped_with_fibe = (doctor_mapped_by_nbfc == "true")
                                logger.info(f"Session {session_id}: Doctor {doctor_id} mapped with FIBE: {doctor_mapped_with_fibe}")
                    except Exception as e:
                        logger.error(f"Session {session_id}: Exception during doctor mapping check for REJECTED status - doctor_id {doctor_id}: {e}", exc_info=True)
                
                if not doctor_mapped_with_fibe:
                    return f"""
We regret to inform you that Patient {patient_name} is not eligible for the proposed loan amount.\n\n

{patient_name} can try financing their treatment via No-Cost Credit & Debit Card EMI or someone from their immediate family can apply on their behalf.\n\n

CTA - \n\n

No-cost Credit & Debit Card EMI\n\n

Re-enquire with your family member's details."""
                else:
                    return f"""
We need a few more details to better assess patient {patient_name}'s application.

Patient's employment type:
1. SALARIED
2. SELF_EMPLOYED
Please Enter input 1 or 2 only"""
            
            elif status and "income verification" in status.lower():
                return f"""
We need a few more details to better assess patient {patient_name}'s application.

Patient's employment type:
1. SALARIED
2. SELF_EMPLOYED
Please Enter input 1 or 2 only"""
            
            else:
                # Default case for unknown status
                logger.warning(f"Unknown bureau decision status: '{status}'")
                return f"""Dear {patient_name}! We are processing Patient's loan application. Please wait while we check Patient's eligibility.
Patient's employment type:
1. SALARIED
2. SELF_EMPLOYED
Please Enter input 1 or 2 only"""
                
        except Exception as e:
            logger.error(f"Error formatting bureau decision response: {e}")
            return "There was an error processing the loan decision. Please try again."
        

    def handle_pan_card_number(self, pan_number: str, session_id: str) -> dict:
        """
        Handle PAN card number input and save it to the system, with PAN validation
        
        Args:
            pan_number: PAN card number provided by user
            session_id: Session identifier
            
        Returns:
            Dictionary with status and message
        """
        try:
            # Validate PAN card number format before processing
            import re
            pan_pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]$'
            if not pan_number or not re.match(pan_pattern, pan_number.strip().upper()):
                return {
                    'status': 'error',
                    'message': "Please provide a valid PAN card number (e.g., ABCDE1234F)."
                }

            # Get session data
            session_data = SessionManager.get_session_from_db(session_id)
            if not session_data:
                return {
                    'status': 'error',
                    'message': "Session not found. Please try again."
                }
            
            # Extract user ID from session data
            user_id = None
            if isinstance(session_data.get('data'), str):
                try:
                    data = json.loads(session_data['data'])
                    user_id = data.get('userId')
                except json.JSONDecodeError:
                    user_id = session_data.get('data')
            else:
                user_id = session_data.get('data', {}).get('userId')
            
            if not user_id:
                return {
                    'status': 'error',
                    'message': "User ID not found in session. Please try again."
                }
            
            # Get mobile number from session data
            mobile_number = session_data.get('data', {}).get('mobileNumber') or session_data.get('data', {}).get('phoneNumber') or session_data.get('phone_number')
            
            # Prepare PAN card details
            pan_details = {
                "userId": user_id,
                "panCard": pan_number,
                "mobileNumber": mobile_number
            }
            
            # Save PAN card details using API client
            logger.info(f"Saving PAN card details for user {user_id}: {pan_details}")
            try:
                # Check if method exists
                if hasattr(self.api_client, 'save_panCard_details'):
                    logger.info("Method save_panCard_details found")
                    save_result = self.api_client.save_panCard_details(user_id, pan_details)
                    logger.info(f"PAN save result: {save_result}")
                else:
                    logger.warning("Method save_panCard_details not found, using save_basic_details")
                    save_result = self.api_client.save_basic_details(user_id, pan_details)
                    logger.info(f"Fallback save result: {save_result}")
            except Exception as e:
                logger.error(f"Error calling save method: {e}")
                # Try fallback
                try:
                    save_result = self.api_client.save_basic_details(user_id, pan_details)
                    logger.info(f"Fallback save result: {save_result}")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise
            
            # Parse the save result
            if isinstance(save_result, str):
                try:
                    save_result_data = json.loads(save_result)
                except json.JSONDecodeError:
                    save_result_data = {"status": 500, "message": "Invalid response from save_panCard_details"}
            else:
                save_result_data = save_result
            
            if save_result_data.get('status') != 200:
                return {
                    'status': 'error',
                    'message': "Failed to save PAN card details. Please try again."
                }
            
            return {
                'status': 'success',
                'session_id': session_id,
                'message': "PAN card number saved successfully. Ask for date of birth and call correct_date_of_birth tool and then ask for gender and call save_gender_B_details",
                'data': {'panCard': pan_number},
            }
            
        except Exception as e:
            logger.error(f"Error handling PAN card number: {e}")
            return {
                'status': 'error',
                'message': f"Error processing PAN card number: {str(e)}"
            }

    def save_missing_basic_and_address_details(
        self,
        pincode: str,
        session_id: str,
    ) -> dict:
        """
        Save address details when pincode is provided, then continue workflow steps.
        This can be called in two scenarios:
        1. When user needs to provide pincode (Workflow B)
        2. When process_address_data returns missing/invalid pincode (Workflow A)

        - Validates: 6-digit pincode
        - Saves address details (and permanent address if API available)
        - Continues with: process_prefill_data_for_basic_details -> pan_verification using session_id -> get_employment_verification -> save_employment_details -> get_bureau_decision
        - CRITICAL: when follow workflow A then After calling save_missing_basic_and_address_details (when pincode is provided), proceed directly to process_prefill_data_for_basic_details, pan_verification using session_id, employment_verification, save_employment_details, and get_bureau_decision. Do NOT call process_address_data again.
        - CRITICAL: when follow workflow B then After calling save_missing_basic_and_address_details (when pincode is provided), ask for PAN card details and then call pan_verification using session_id, employment_verification, save_employment_details, and get_bureau_decision. Do NOT call process_address_data again.
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return {"status": "error", "message": "Session not found"}

            user_id = session.get("data", {}).get("userId")
            if not user_id:
                return {"status": "error", "message": "User ID missing in session"}

            # Validate pincode
            if not pincode or not re.match(r"^\d{6}$", pincode.strip()):
                return {"status": "error", "message": "Pincode must be a 6-digit number."}

            # Check if we have extracted address data from process_address_data
            extracted_address_data = session.get("data", {}).get("extracted_address_data", {})
            
            # Prepare and enrich address data
            address_data = {
                "pincode": pincode.strip(),
                "formStatus": "Address",
            }
            
            # If we have extracted address data, merge it with the pincode
            if extracted_address_data:
                address_data.update(extracted_address_data)
                # Ensure pincode from user input takes precedence
                address_data["pincode"] = pincode.strip()
            
            # Enrich address data with city and state from pincode API
            try:
                if len(address_data["pincode"]) == 6:
                    pincode_info = self.api_client.state_and_city_by_pincode(address_data["pincode"]) or {}
                    if pincode_info.get("status") == "success":
                        if pincode_info.get("city"):
                            address_data["city"] = pincode_info["city"]
                        if pincode_info.get("state"):
                            address_data["state"] = pincode_info["state"]
            except Exception:
                pass

            # Save address details
            addr_resp = self.api_client.save_address_details(user_id, address_data)
            if isinstance(addr_resp, str):
                try:
                    addr_resp = json.loads(addr_resp)
                except json.JSONDecodeError:
                    addr_resp = {"status": 500}
            if addr_resp.get("status") != 200:
                return {"status": "error", "message": "Failed to save address details."}

            # Also save permanent address if supported
            try:
                if hasattr(self.api_client, "save_permanent_address_details"):
                    _ = self.api_client.save_permanent_address_details(user_id, address_data)
            except Exception:
                pass

            # Store the API response
            SessionManager.update_session_data_field(session_id, "data.api_responses.save_missing_address_details", addr_resp)

            return {
                "status": "success",
                "message": "Address details saved successfully. according to workflow continue the next step",
            }
        except Exception as e:
            logger.error(f"Error in save_missing_basic_and_address_details: {e}", exc_info=True)
            return {"status": "error", "message": "Failed to save details. Please try again."}

    

    def handle_email_address(self, email_address: str, session_id: str) -> dict:
        """
        Handle email address input and save it to the system
        
        Args:
            email_address: Email address provided by user
            session_id: Session identifier
            
        Returns:
            Dictionary with status and message
        """
        try:
            # Basic email validation
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email_address):
                return {
                    'status': 'error',
                    'message': "Please provide a valid email address."
                }
            
            # Get session data
            session_data = SessionManager.get_session_from_db(session_id)
            if not session_data:
                return {
                    'status': 'error',
                    'message': "Session not found. Please try again."
                }
            
            # Extract user ID from session data
            user_id = None
            if isinstance(session_data.get('data'), str):
                try:
                    data = json.loads(session_data['data'])
                    user_id = data.get('userId')
                except json.JSONDecodeError:
                    user_id = session_data.get('data')
            else:
                user_id = session_data.get('data', {}).get('userId')
            
            if not user_id:
                return {
                    'status': 'error',
                    'message': "User ID not found in session. Please try again."
                }
            
            # Get mobile number from session data
            mobile_number = session_data.get('data', {}).get('mobileNumber') or session_data.get('data', {}).get('phoneNumber') or session_data.get('phone_number')
            
            # Prepare email details
            email_details = {
                "userId": user_id,
                "emailId": email_address,
                "mobileNumber": mobile_number
            }
            
            # Save email details using API client
            logger.info(f"Saving email details for user {user_id}: {email_details}")
            try:
                # Check if method exists
                if hasattr(self.api_client, 'save_emailaddress_details'):
                    logger.info("Method save_emailaddress_details found")
                    save_result = self.api_client.save_emailaddress_details(user_id, email_details)
                    logger.info(f"Email save result: {save_result}")
                else:
                    logger.warning("Method save_emailaddress_details not found, using save_basic_details")
                    save_result = self.api_client.save_basic_details(user_id, email_details)
                    logger.info(f"Fallback save result: {save_result}")
            except Exception as e:
                logger.error(f"Error calling save method: {e}")
                # Try fallback
                try:
                    save_result = self.api_client.save_basic_details(user_id, email_details)
                    logger.info(f"Fallback save result: {save_result}")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise
            
            # Parse the save result
            if isinstance(save_result, str):
                try:
                    save_result_data = json.loads(save_result)
                except json.JSONDecodeError:
                    save_result_data = {"status": 500, "message": "Invalid response from save_emailaddress_details"}
            else:
                save_result_data = save_result
            
            if save_result_data.get('status') != 200:
                return {
                    'status': 'error',
                    'message': "Failed to save email address. Please try again."
                }
            
            import json
            return json.dumps({
                'status': 'success',
                'message': "Email address saved successfully. Now continuing with the remaining verification steps automatically...",
                'data': {'emailId': email_address},
                'continue_chain': True,
                'session_id': session_id,
                'next_step': 'pan_verification using session_id, employment_verification, save_employment_details, get_bureau_decision'
            })
            
        except Exception as e:
            logger.error(f"Error handling email address: {e}")
            import json
            return json.dumps({
                'status': 'error',
                'message': f"Error processing email address: {str(e)}"
            })

    def handle_aadhaar_upload(self, document_path: str, session_id: str) -> dict:
        try:
            logger.info(f"Starting Aadhaar upload processing for session {session_id}")
            
            # Extract Aadhaar details using OCR service
            result = extract_aadhaar_details(document_path)
            logger.info(f"OCR extraction result: {result}")
            
            # Store the OCR result in session data
            SessionManager.update_session_data_field(session_id, 'ocr_result', result)
            
            # Validate extracted data
            if not result.get('name') or not result.get('aadhaar_number'):
                return {
                    'status': 'error',
                    'message': "Could not extract name or Aadhaar number from the document. Please ensure the document is clear and try again.",
                    'data': result
                }
            
            # Get session data
            session_data = SessionManager.get_session_from_db(session_id)
            if not session_data:
                return {
                    'status': 'error',
                    'message': "Session not found. Please try again."
                }
            
            # Extract user ID from session data
            user_id = None
            if isinstance(session_data.get('data'), str):
                try:
                    data = json.loads(session_data['data'])
                    user_id = data.get('userId')
                except json.JSONDecodeError:
                    user_id = session_data.get('data')
            else:
                user_id = session_data.get('data', {}).get('userId')
            
            if not user_id:
                return {
                    'status': 'error',
                    'message': "User ID not found in session. Please try again."
                }
            
            # Get mobile number from session data
            mobile_number = session_data.get('data', {}).get('mobileNumber') or session_data.get('data', {}).get('phoneNumber') or session_data.get('phone_number')
            
            # Prepare basic details from OCR data
            basic_details = {
                "userId": user_id,
                # "firstName": result.get('name', ''),
                "dateOfBirth": result.get('dob', ''),
                "gender": result.get('gender', ''),
                "aadhaarNo": result.get('aadhaar_number', ''),
                "fatherName": result.get('father_name', ''),
                "mobileNumber": mobile_number,
                "formStatus": "Basic"
            }
            
            # Save basic details using API client
            logger.info(f"Saving Aadhaar details for user {user_id}: {basic_details}")
            try:
                # Check if method exists
                if hasattr(self.api_client, 'save_aadhaar_details'):
                    logger.info("Method save_aadhaar_details found")
                    save_result = self.api_client.save_aadhaar_details(user_id, basic_details)
                    logger.info(f"Save result: {save_result}")
                else:
                    logger.warning("Method save_aadhaar_details not found, using save_basic_details")
                    save_result = self.api_client.save_basic_details(user_id, basic_details)
                    logger.info(f"Fallback save result: {save_result}")
            except Exception as e:
                logger.error(f"Error calling save method: {e}")
                # Try fallback
                try:
                    save_result = self.api_client.save_basic_details(user_id, basic_details)
                    logger.info(f"Fallback save result: {save_result}")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise
            
            # Parse the save result
            if isinstance(save_result, str):
                try:
                    save_result_data = json.loads(save_result)
                except json.JSONDecodeError:
                    save_result_data = {"status": 500, "message": "Invalid response from save_basic_details"}
            else:
                save_result_data = save_result
            
            if save_result_data.get('status') != 200:
                return {
                    'status': 'error',
                    'message': "Failed to save basic details. Please try again."
                }
            
            # Process address data if available
            if result.get('address') or result.get('pincode'):
                address_data = {
                    "address": result.get('address', ''),
                    "pincode": result.get('pincode', ''),
                    "formStatus": "Address"
                }
                
                # If we have a valid pincode, get city and state from API
                if address_data.get('pincode') and len(address_data['pincode']) == 6:
                    try:
                        pincode_data = self.api_client.state_and_city_by_pincode(address_data['pincode'])
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
                
                logger.info(f"Final address data to save: {address_data}")
                
                # Save address details
                address_result = self.api_client.save_address_details(user_id, address_data)
                address_permanent_result = self.api_client.save_permanent_address_details(user_id, address_data)
                if isinstance(address_result, str):
                    try:
                        address_result_data = json.loads(address_result)
                    except json.JSONDecodeError:
                        address_result_data = {"status": 500, "message": "Invalid response from save_address_details"}
                else:
                    address_result_data = address_result

                logger.info(f"Address result: {address_result} and address_permanent_result: {address_permanent_result}")
                
                if address_result_data.get('status') != 200:
                    return {
                        'status': 'warning',
                        'message': "Basic details saved but failed to save address. Please try again.",
                        'data': result
                    }
            
            return {
                'status': 'success',
                'message': "Successfully processed Aadhaar document and saved details.",
                'data': result
            }
            
        except Exception as e:
            logger.error(f"Error handling Aadhaar upload: {e}")
            return {
                'status': 'error',
                'message': f"Error processing Aadhaar document: {str(e)}"
            }

    def save_gender_details(self, gender: str, session_id: str) -> str:
        """
        Save user's gender details

        Args:
            gender: User's gender (Male/Female/Other)
            session_id: Session identifier

        Returns:
            Save result as JSON string
        """
        logger.info(f"save_gender_details called with: gender='{gender}', session_id='{session_id}'")
        try:
            # Get user ID from session
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return "Session not found"

            user_id = session.get("data", {}).get("userId")
            if not user_id:
                return "User ID not found in session"

            # Get mobile number from session
            mobile_number = session.get("data", {}).get("mobileNumber") or session.get("data", {}).get("phoneNumber")

            # Prepare data for API
            details = {
                "gender": gender,
                "mobileNumber": mobile_number,
                "userId": user_id
            }

            # Store the data being sent to the API
            SessionManager.update_session_data_field(session_id, "data.api_requests.save_gender_details", {
                "user_id": user_id,
                "details": details.copy()
            })

            # Call API
            result = self.api_client.save_gender_details(user_id, details)

            # Store the API response
            SessionManager.update_session_data_field(session_id, "data.api_responses.save_gender_details", result)

            import json
            return json.dumps({
                'status': 'success',
                'message': "Gender saved successfully. Now proceeding to PAN verification and employment verification steps. Please wait while I process the next steps automatically.",
                'data': result,
                'session_id': session_id
            })

        except Exception as e:
            logger.error(f"Error saving gender details: {e}")
            import json
            return json.dumps({
                'status': 'error',
                'message': f"Error saving gender details: {str(e)}"
            })

    def save_marital_status_details(self, marital_status: str, session_id: str) -> str:
        """
        Save user's marital status details

        Args:
            marital_status: User's marital status (Yes/No)
            session_id: Session identifier

        Returns:
            Save result as JSON string
        """
        logger.info(f"save_marital_status_details called with: marital_status='{marital_status}', session_id='{session_id}'")
        try:
            # Get user ID from session
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return "Session not found"

            user_id = session.get("data", {}).get("userId")
            if not user_id:
                return "User ID not found in session"

            # Get mobile number from session
            mobile_number = session.get("data", {}).get("mobileNumber") or session.get("data", {}).get("phoneNumber")

            # Format marital status to correct API format
            formatted_marital_status = self._format_marital_status(marital_status)
            logger.info(f"Formatted marital status: '{marital_status}' -> '{formatted_marital_status}'")
            logger.info(f"Input type: {type(marital_status)}, Input value: '{marital_status}'")

            # Prepare data for API
            details = {
                "maritalStatus": formatted_marital_status,
                "mobileNumber": mobile_number,
                "userId": user_id
            }

            # Store the data being sent to the API
            SessionManager.update_session_data_field(session_id, "data.api_requests.save_marital_status_details", {
                "user_id": user_id,
                "details": details.copy()
            })

            # Call API
            result = self.api_client.save_marital_status_details(user_id, details)

            # Store the API response
            SessionManager.update_session_data_field(session_id, "data.api_responses.save_marital_status_details", result)

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error saving marital status details: {e}")
            return f"Error saving marital status details: {str(e)}"

    def save_education_level_details(self, education_level: str, session_id: str) -> str:
        """
        Save user's education level details

        Args:
            education_level: User's education level (LESS THAN 10TH, PASSED 10TH, etc.)
            session_id: Session identifier

        Returns:
            Save result as JSON string
        """
        logger.info(f"save_education_level_details called with: education_level='{education_level}', session_id='{session_id}'")
        try:
            # Get user ID from session
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return "Session not found"

            user_id = session.get("data", {}).get("userId")
            if not user_id:
                return "User ID not found in session"

            # Get mobile number from session
            mobile_number = session.get("data", {}).get("mobileNumber") or session.get("data", {}).get("phoneNumber")

            # Format education level to correct API format
            formatted_education_level = self._format_education_level(education_level)
            logger.info(f"Formatted education level: '{education_level}' -> '{formatted_education_level}'")

            # Prepare data for API
            details = {
                "educationLevel": formatted_education_level,
                "mobileNumber": mobile_number,
                "userId": user_id
            }

            # Store the data being sent to the API
            SessionManager.update_session_data_field(session_id, "data.api_requests.save_education_level_details", {
                "user_id": user_id,
                "details": details.copy()
            })

            # Call API
            result = self.api_client.save_education_level_details(user_id, details)

            # Store the API response
            SessionManager.update_session_data_field(session_id, "data.api_responses.save_education_level_details", result)

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error saving education level details: {e}")
            return f"Error saving education level details: {str(e)}"

    def correct_treatment_name(self, new_treatment_reason: str, session_id: str) -> str:
        """
        Correct/update the treatment reason in the loan application
        
        Args:
            new_treatment_reason: The new/corrected treatment reason
            session_id: Session ID to get user data from
            
        Returns:
            Success or error message
        """
        logger.info(f"correct_treatment_name called with: new_treatment_reason='{new_treatment_reason}', session_id='{session_id}'")
        try:
            # Get session data
            session_data = SessionManager.get_session_from_db(session_id)
            if not session_data:
                return "❌ Error: Session not found. Please start a new conversation."
            
            user_data = session_data.get('data', {})
            user_id = user_data.get('userId')
            
            if not user_id:
                return "❌ Error: User ID not found in session. Please complete the initial setup first."
            
            # Try to get doctor_id and doctor_name from session
            doctor_id = user_data.get('doctorId') or user_data.get('doctor_id')
            doctor_name = user_data.get('doctorName') or user_data.get('doctor_name')
            
            # Get existing loan data from session
            loan_data = {
                "doctorId": doctor_id,
                "doctorName": doctor_name,
                "treatmentCost": user_data.get('treatmentCost'),
                "loanReason": new_treatment_reason,
                "fullName": user_data.get('fullName')
            }
            
            # Call API to update treatment name
            response = self.api_client.save_change_treatment_name_details(user_id, loan_data)
            
            if response.get("status") == 200:
                # Update session with new treatment reason
                SessionManager.update_session_data_field(session_id, "data.treatmentReason", new_treatment_reason)
                
                return f"✅ Treatment reason has been successfully updated to '{new_treatment_reason}'!"
            else:
                error_msg = response.get("error", "Unknown error occurred")
                logger.error(f"Failed to update treatment name: {error_msg}")
                return f"❌ Error updating treatment name: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error in correct_treatment_name: {e}")
            return f"❌ Error: {str(e)}"

    def correct_treatment_cost(self, new_treatment_cost, session_id: str) -> str:
        """
        Correct/update the treatment cost in the loan application
        
        Args:
            new_treatment_cost: The new/corrected treatment cost (must be >= 3000 and <= 1000000)
            session_id: Session ID to get user data from
            
        Returns:
            Success or error message
        """
        logger.info(f"correct_treatment_cost called with: new_treatment_cost='{new_treatment_cost}', session_id='{session_id}'")
        try:
            # Convert to integer if it's a string
            try:
                new_treatment_cost = int(new_treatment_cost)
            except (ValueError, TypeError):
                return "❌ Error: Please enter a valid numeric amount for the treatment cost."
            
            # Validate treatment cost
            if new_treatment_cost < 3000:
                return "❌ Error: Treatment cost must be ₹3,000 or more. Please enter a valid amount."
            elif new_treatment_cost > 1000000:
                return "❌ Error: Treatment cost cannot exceed ₹10,00,000. Please enter a valid amount."
            
            # Get session data
            session_data = SessionManager.get_session_from_db(session_id)
            if not session_data:
                return "❌ Error: Session not found. Please start a new conversation."
            
            user_data = session_data.get('data', {})
            user_id = user_data.get('userId')
            
            if not user_id:
                return "❌ Error: User ID not found in session. Please complete the initial setup first."
            
            # Get doctor_id and doctor_name from session (handle both possible keys)
            doctor_id = user_data.get('doctorId') or user_data.get('doctor_id')
            doctor_name = user_data.get('doctorName') or user_data.get('doctor_name')

            # Get treatment_reason from additional_details if available
            additional_details = user_data.get('additional_details', {})
            treatment_reason = additional_details.get('treatment_reason', '')

            # Get existing loan data from session, using treatment_reason from additional_details
            loan_data = {
                "doctorId": doctor_id,
                "doctorName": doctor_name,
                "treatmentCost": new_treatment_cost,  # Use the new treatment cost
                "loanReason": treatment_reason,
                "fullName": user_data.get('fullName')
            }
            
            # Call API to update treatment cost
            response = self.api_client.save_change_treatment_cost_details(user_id, loan_data)
            
            if response.get("status") == 200:
                # Update session with new treatment cost
                SessionManager.update_session_data_field(session_id, "data.treatmentCost", new_treatment_cost)
                
                return f"✅ Treatment cost has been successfully updated to ₹{new_treatment_cost:,}!"
            else:
                error_msg = response.get("error", "Unknown error occurred")
                logger.error(f"Failed to update treatment cost: {error_msg}")
                return f"❌ Error updating treatment cost: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error in correct_treatment_cost: {e}")
            return f"❌ Error: {str(e)}"

    def correct_date_of_birth(self, new_date_of_birth: str, session_id: str) -> str:
        """
        Correct/update the date of birth in the user profile
        
        Args:
            new_date_of_birth: The new/corrected date of birth (format: DD-MM-YYYY)
            session_id: Session ID to get user data from
            
        Returns:
            Success or error message
        """
        try:
            # Validate and convert date format from DD-MM-YYYY to YYYY-MM-DD
            try:
                # Parse DD-MM-YYYY format
                date_obj = datetime.strptime(new_date_of_birth, '%d-%m-%Y')
                # Convert to YYYY-MM-DD format for saving
                formatted_date = date_obj.strftime('%Y-%m-%d')
            except ValueError:
                return "❌ Error: Please enter the date in DD-MM-YYYY format (e.g., 15-01-1990)."
            
            # Get session data
            session_data = SessionManager.get_session_from_db(session_id)
            if not session_data:
                return "❌ Error: Session not found. Please start a new conversation."
            
            
            user_data = session_data.get('data', {})
            user_id = user_data.get('userId')
            phone_number = user_data.get('phoneNumber')
            
            if not user_id:
                return "❌ Error: User ID not found in session. Please complete the initial setup first."
            
            if not phone_number:
                return "❌ Error: Phone number not found in session. Please complete the initial setup first."
            
            # Prepare details for API
            details = {
                "dateOfBirth": formatted_date,
                "mobileNumber": phone_number,
                "userId": user_id
            }
            
            # Call API to update date of birth
            response = self.api_client.save_change_date_of_birth_details(user_id, details)
            
            if response.get("status") == 200:
                # Update session with new date of birth (in YYYY-MM-DD format)
                SessionManager.update_session_data_field(session_id, "data.dateOfBirth", formatted_date)
                
                return f"✅ Date of birth has been successfully updated to {new_date_of_birth}!"
            else:
                error_msg = response.get("error", "Unknown error occurred")
                logger.error(f"Failed to update date of birth: {error_msg}")
                return f"❌ Error updating date of birth: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error in correct_date_of_birth: {e}")
            return f"❌ Error: {str(e)}"

    def handle_pan_card_upload(self, document_path: str, session_id: str, ocr_result: dict = None) -> dict:
        """
        Handle PAN card upload and save extracted details
        
        Args:
            document_path: Path to the uploaded PAN card document
            session_id: Session ID to get user data from
            ocr_result: Pre-extracted OCR result (optional)
            
        Returns:
            Dictionary with status and message
        """
        try:
            # Get session data
            session_data = SessionManager.get_session_from_db(session_id)
            if not session_data:
                return {
                    'status': 'error',
                    'message': 'Session not found. Please start a new conversation.'
                }
            
            user_data = session_data.get('data', {})
            user_id = user_data.get('userId')
            phone_number = user_data.get('phoneNumber')
            
            if not user_id:
                return {
                    'status': 'error',
                    'message': 'User ID not found in session. Please complete the initial setup first.'
                }
            
            if not phone_number:
                return {
                    'status': 'error',
                    'message': 'Phone number not found in session. Please complete the initial setup first.'
                }
            
            # Extract PAN details if not provided
            if not ocr_result:
                from cpapp.services.ocr_service import extract_pan_details
                ocr_result = extract_pan_details(document_path)
            
            # Validate OCR result
            if not ocr_result or not any(ocr_result.values()):
                return {
                    'status': 'error',
                    'message': 'Failed to extract PAN card details from the uploaded document.'
                }
            
            pan_card_number = ocr_result.get('pan_card_number', '')
            person_name = ocr_result.get('person_name', '')
            date_of_birth = ocr_result.get('date_of_birth', '')
            father_name = ocr_result.get('father_name', '')
            
            # Validate PAN card number
            if not pan_card_number:
                return {
                    'status': 'error',
                    'message': 'Could not extract PAN card number from the document.'
                }
            
            # Save PAN card number to API
            pan_details = {
                "panCard": pan_card_number,
                "mobileNumber": phone_number
            }
            
            pan_response = self.api_client.save_panCard_details(user_id, pan_details)
            
            if pan_response.get("status") != 200:
                logger.error(f"Failed to save PAN card number: {pan_response}")
                return {
                    'status': 'error',
                    'message': f'Failed to save PAN card number: {pan_response.get("error", "Unknown error")}'
                }
            
            # Save other details if available
            details_to_save = {}
            
            if person_name:
                details_to_save["fullName"] = person_name
            
            if date_of_birth:
                details_to_save["dateOfBirth"] = date_of_birth
            
            
            if father_name:
                details_to_save["fatherName"] = father_name
            
            # Save additional details if any
            if details_to_save:
                details_to_save["mobileNumber"] = phone_number
                basic_response = self.api_client.save_basic_details(user_id, details_to_save)
                
                if basic_response.get("status") != 200:
                    logger.warning(f"Failed to save additional PAN details: {basic_response}")
                    # Continue anyway as PAN card number was saved successfully
            
            # Update session with extracted data
            SessionManager.update_session_data_field(session_id, "data.panCard", pan_card_number)
            if person_name:
                SessionManager.update_session_data_field(session_id, "data.fullName", person_name)
            if date_of_birth:
                SessionManager.update_session_data_field(session_id, "data.dateOfBirth", date_of_birth)

            if father_name:
                SessionManager.update_session_data_field(session_id, "data.fatherName", father_name)
            
            # Store OCR result in session
            SessionManager.update_session_data_field(session_id, "data.pan_ocr_result", ocr_result)
            
            # Prepare success message
            success_parts = [f"✅ PAN card number: {pan_card_number}"]
            if person_name:
                success_parts.append(f"Name: {person_name}")
            if date_of_birth:
                success_parts.append(f"Date of Birth: {date_of_birth}")
            if father_name:
                success_parts.append(f"Father's Name: {father_name}")
            
            success_message = " | ".join(success_parts)
            
            return {
                'status': 'success',
                'message': f'PAN card processed successfully! {success_message}',
                'data': ocr_result
            }
            
        except Exception as e:
            logger.error(f"Error in handle_pan_card_upload: {e}")
            return {
                'status': 'error',
                'message': f'Error processing PAN card: {str(e)}'
            }

    def _format_marital_status(self, marital_status: str) -> str:
        """
        Format marital status to the correct API format
        
        Args:
            marital_status: Raw marital status input
            
        Returns:
            Formatted marital status for API
        """
        logger.info(f"_format_marital_status called with: '{marital_status}'")
        
        if not marital_status:
            logger.info("Empty marital status, returning 'No'")
            return "No"
        
        # Convert to lowercase for easier comparison
        status_lower = marital_status.lower().strip()
        logger.info(f"Lowercase status: '{status_lower}'")
        
        # Map various inputs to correct API format
        married_variants = ["married", "yes", "1", "marriage"]
        unmarried_variants = ["unmarried", "single", "no", "2", "unmarried/single", "unmarried or single"]
        
        if status_lower in married_variants:
            logger.info(f"Matched married variant: '{status_lower}' -> 'Yes'")
            return "Yes"
        elif status_lower in unmarried_variants:
            logger.info(f"Matched unmarried variant: '{status_lower}' -> 'No'")
            return "No"
        else:
            # If it's already in correct format, return as-is
            if marital_status in ["Yes", "No"]:
                logger.info(f"Already in correct format: '{marital_status}'")
                return marital_status
            # Default to "No" for unrecognized values
            logger.warning(f"Unrecognized marital status: '{marital_status}', defaulting to 'No'")
            return "No"

    def _format_education_level(self, education_level: str) -> str:
        """
        Format education level to the correct API format
        
        Args:
            education_level: Raw education level input
            
        Returns:
            Formatted education level for API
        """
        if not education_level:
            return "LESS THAN 10TH"
        
        # Convert to lowercase for easier comparison
        level_lower = education_level.lower().strip()
        
        # Map various inputs to correct API format
        education_mapping = {
            # Number mappings
            "1": "LESS THAN 10TH",
            "2": "PASSED 10TH", 
            "3": "PASSED 12TH",
            "4": "DIPLOMA",
            "5": "GRADUATION",
            "6": "POST GRADUATION",
            "7": "P.H.D.",
            
            # Text mappings
            "less than 10th": "LESS THAN 10TH",
            "less than 10": "LESS THAN 10TH",
            "below 10th": "LESS THAN 10TH",
            "below 10": "LESS THAN 10TH",
            "under 10th": "LESS THAN 10TH",
            "under 10": "LESS THAN 10TH",
            
            "passed 10th": "PASSED 10TH",
            "10th": "PASSED 10TH",
            "10th standard": "PASSED 10TH",
            "sslc": "PASSED 10TH",
            
            "passed 12th": "PASSED 12TH",
            "12th": "PASSED 12TH",
            "12th standard": "PASSED 12TH",
            "hsc": "PASSED 12TH",
            "higher secondary": "PASSED 12TH",
            
            "diploma": "DIPLOMA",
            "diploma course": "DIPLOMA",
            
            "graduation": "GRADUATION",
            "graduate": "GRADUATION",
            "bachelor": "GRADUATION",
            "bachelor's": "GRADUATION",
            "bachelors": "GRADUATION",
            "b.tech": "GRADUATION",
            "b.e": "GRADUATION",
            "b.com": "GRADUATION",
            "b.sc": "GRADUATION",
            "b.a": "GRADUATION",
            "b.b.a": "GRADUATION",
            "b.c.a": "GRADUATION",
            
            "post graduation": "POST GRADUATION",
            "post graduate": "POST GRADUATION",
            "postgraduate": "POST GRADUATION",
            "master": "POST GRADUATION",
            "master's": "POST GRADUATION",
            "masters": "POST GRADUATION",
            "m.tech": "POST GRADUATION",
            "m.e": "POST GRADUATION",
            "m.com": "POST GRADUATION",
            "m.sc": "POST GRADUATION",
            "m.a": "POST GRADUATION",
            "m.b.a": "POST GRADUATION",
            "m.c.a": "POST GRADUATION",
            
            "p.h.d": "P.H.D.",
            "phd": "P.H.D.",
            "doctorate": "P.H.D.",
            "doctor of philosophy": "P.H.D.",
            "ph.d": "P.H.D.",
            "ph.d.": "P.H.D.",
        }
        
        # Check if it's already in correct format
        if education_level in ["LESS THAN 10TH", "PASSED 10TH", "PASSED 12TH", "DIPLOMA", "GRADUATION", "POST GRADUATION", "P.H.D."]:
            return education_level
        
        # Try to find a match
        for key, value in education_mapping.items():
            if level_lower == key.lower():
                return value
        
        # If no exact match, try partial matching
        for key, value in education_mapping.items():
            if key.lower() in level_lower or level_lower in key.lower():
                logger.info(f"Partial match found for education level: '{education_level}' -> '{value}'")
                return value
        
        # Default to "LESS THAN 10TH" for unrecognized values
        logger.warning(f"Unrecognized education level: '{education_level}', defaulting to 'LESS THAN 10TH'")
        return "LESS THAN 10TH"

    def save_gender_B_details(self, gender: str, session_id: str) -> str:
        """
        Save user's gender details

        Args:
            gender: User's gender (Male/Female/Other)
            session_id: Session identifier

        Returns:
            Save result as JSON string
        """
        logger.info(f"save_gender_B_details called with: gender='{gender}', session_id='{session_id}'")
        try:
            # Get user ID from session
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                return "Session not found"

            user_id = session.get("data", {}).get("userId")
            if not user_id:
                return "User ID not found in session"

            # Get mobile number from session
            mobile_number = session.get("data", {}).get("mobileNumber") or session.get("data", {}).get("phoneNumber")

            # Prepare data for API
            details = {
                "gender": gender,
                "mobileNumber": mobile_number,
                "userId": user_id
            }

            # Store the data being sent to the API
            SessionManager.update_session_data_field(session_id, "data.api_requests.save_gender_B_details", {
                "user_id": user_id,
                "details": details.copy()
            })

            # Call API
            result = self.api_client.save_gender_details(user_id, details)

            # Store the API response
            SessionManager.update_session_data_field(session_id, "data.api_responses.save_gender_B_details", result)

            import json
            return json.dumps({
                'status': 'success',
                'message': "Gender saved successfully. process next steps(step 3)",
                'data': result,
                'session_id': session_id
            })

        except Exception as e:
            logger.error(f"Error saving gender details: {e}")
            import json
            return json.dumps({
                'status': 'error',
                'message': f"Error saving gender details: {str(e)}"
            })