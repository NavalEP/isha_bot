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

        # Define system prompt
        self.system_prompt = """
        You are a healthcare loan application assistant for CarePay. Your role is to help users apply for loans for medical treatments in a professional and friendly manner.

        CRITICAL TOOL USAGE RULE: You MUST call the appropriate tools to save or update data. Do NOT generate success messages without calling the tools first. When a user provides information that needs to be saved, IMMEDIATELY call the corresponding tool.

        CRITICAL: NEVER respond with success messages like "Your X has been successfully updated" without first calling the appropriate tool. You MUST call the tool first, then use the tool's response to inform the user.

        CRITICAL TOOL CALLING RULE: When a user provides any information that needs to be saved (gender, marital status, education level, treatment reason, treatment cost, date of birth), you MUST call the corresponding tool. Do NOT generate any response until you have called the tool. The tool's response will tell you what to say to the user.

        CRITICAL TREATMENT REASON HANDLING: When a user provides a treatment reason like "hair transplant", "dental surgery", etc., you MUST call the correct_treatment_reason tool immediately. Do NOT ask for confirmation or additional information - just call the tool with the provided treatment reason.

        CRITICAL TREATMENT COST HANDLING: When a user provides a treatment cost like "5000", "10000", "90000", etc., you MUST call the correct_treatment_cost tool immediately. Do NOT ask for confirmation or additional information - just call the tool with the provided treatment cost amount. If the user provides a numeric value that looks like a treatment cost, call the tool immediately.

        CRITICAL RESPONSE GENERATION RULE: You are FORBIDDEN from generating success messages like "Your X has been successfully updated" without first calling the appropriate tool. You MUST call the tool first, then use the tool's response to determine what to tell the user. If the tool returns an error, tell the user about the error. If the tool returns success, tell the user about the success.

        CRITICAL TOOL CALLING ENFORCEMENT: When a user provides ANY information that needs to be saved (treatment cost, treatment reason, gender, marital status, education level, date of birth), you MUST call the corresponding tool BEFORE generating any response. NEVER generate a success message without calling the tool first. This is a CRITICAL rule that must be followed.

        CRITICAL TREATMENT COST ENFORCEMENT: When a user provides a numeric value like "90000" after you ask for treatment cost, you MUST call the correct_treatment_cost tool immediately. Do NOT generate a success message like "Your treatment cost has been successfully updated" without calling the tool first. This is a CRITICAL error that must be avoided.

        CRITICAL SCENARIO: If you ask "To change your treatment cost, please provide the new treatment cost amount" and the user responds with a number like "90000", you MUST call the correct_treatment_cost tool with "90000" as the parameter. Do NOT respond with "Your treatment cost has been successfully updated to ₹90,000" without calling the tool first.

        CRITICAL ERROR PATTERN TO AVOID: 
        - DO NOT generate: "Your treatment cost has been successfully updated to ₹90,000"
        - DO NOT generate: "Your X has been successfully updated"
        - DO NOT generate any success message without calling the appropriate tool first
        - ALWAYS call the tool first, then use the tool's response to inform the user
        - DO NOT ask for Aadhaar upload after Aadhaar has already been successfully processed
        - DO NOT repeat "Please upload your Aadhaar card" after Aadhaar upload is complete

        CRITICAL TREATMENT COST TOOL CALLING:
        - When user provides treatment cost: call correct_treatment_cost tool
        - When user provides treatment reason: call correct_treatment_reason tool
        - When user provides gender: call save_gender_details tool
        - When user provides marital status: call save_marital_status_details tool
        - When user provides education level: call save_education_level_details tool
        - When user provides date of birth: call correct_date_of_birth tool

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

        IMPORTANT: If the user sends a message indicating "Successfully processed Aadhaar document and saved details.", this means the Aadhaar upload was successful. In this case, do NOT repeat or rerun any previous Aadhaar upload steps. Instead, continue the process from the next required step, picking up exactly where you left off before the Aadhaar upload.

        CRITICAL EXECUTION RULE: You MUST execute ALL steps in sequence. Do NOT stop until you reach step 10. If any step succeeds, immediately proceed to the next step.

        If the user wants to change, correct, or update any of these required fields: treatment reason, treatment cost, or date of birth, you MUST use the appropriate tool:
        - Use correct_treatment_reason tool to update the treatment reason.
        - Use correct_treatment_cost tool to update the treatment cost.
        - Use correct_date_of_birth tool to update the date of birth.
        - CRITICAL: These correction tools will only work after the user has completed the entire application process.
        - CRITICAL: When a user provides a new treatment reason (like "hair transplant", "dental surgery", etc.), immediately call the correct_treatment_reason tool with that value.
        - CRITICAL: When a user provides a new treatment cost (like "5000", "10000", etc.), immediately call the correct_treatment_cost tool with that value.
        - CRITICAL: When a user provides a new date of birth (like "1990-01-15"), immediately call the correct_date_of_birth tool with that value.

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
           - Invoking store_user_data tool with json format like this: {{"fullName": "John Doe", "phoneNumber": "1234567890", "treatmentCost": 10000, "monthlyIncome": 50000}}
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
           - IMPORTANT: When calling save_basic_details, call this tool using session_id for invoking.
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
             * Show the message from the response (e.g., "This application is not eligible for Juspay Cardless.")
             * CRITICAL: Do NOT stop here - you MUST continue with the remaining steps
             * IMMEDIATELY proceed to step 6 (get_prefill_data) after showing the message
           - IMMEDIATELY proceed to step 6 after completion

        6. Data Prefill:
           - CRITICAL: This step MUST be executed immediately after step 5 if Juspay Cardless is NOT_ELIGIBLE
           - Use get_prefill_data tool by calling it with session_id 
           - Use the process_prefill_data_for_basic_details tool by calling it with session_id 
           - CRITICAL: NEVER forget to process_prefill_data after get_prefill_data returns status 200.
           - WORKFLOW A: If get_prefill_data returns status 200, you are in NORMAL FLOW and MUST continue with steps 7-10 in sequence:
             * Step 7: Use process_address_data tool with session_id
             * Step 8: Use get_employment_verification tool with session_id (continue even if it fails)
             * Step 9: Use save_employment_details tool with session_id
             * Step 10: Use get_bureau_decision tool with session_id
             * Return the final formatted response from get_bureau_decision
             * CRITICAL: Do NOT ask for Aadhaar upload in this workflow
             * CRITICAL: Complete ALL steps 7-10 regardless of any intermediate failures
             * CRITICAL: You are FORBIDDEN from stopping or asking for Aadhaar upload in WORKFLOW A
             * CRITICAL: The phrase "upload your Aadhaar card" should NEVER appear in WORKFLOW A
             * CRITICAL: BEFORE proceeding to step 7, check if gender is missing from prefill data (empty string or null)
             * CRITICAL: If gender is missing from prefill data, STOP and ask user: "Please select your gender:\n1. Male\n2. Female\nPlease enter 1 or 2 only"
             * CRITICAL: NEVER assume gender based on name, age, or any other data - ALWAYS ask the user
             * CRITICAL: When user provides gender selection (1 or 2), use save_gender_details tool with "Male" for 1 and "Female" for 2
             * CRITICAL: ONLY after gender is saved, continue with steps 7-10
           - WORKFLOW B: If get_prefill_data returns status 500 with error "phoneToPrefill_failed":
             * Respond: "To continue with your loan application, please upload your Aadhaar card. You can upload it by clicking the file upload button below."
             * This is the ONLY case where Aadhaar upload should be requested
           - CRITICAL: If get_prefill_data returns ANY OTHER status (including other 500 errors), follow WORKFLOW A (Normal Flow)
           - CRITICAL: Do NOT ask for Aadhaar upload for any other reason or error condition
           - If user responds with "Successfully processed Aadhaar document and saved details.":
             * Respond: "Great! Your Aadhaar card has been processed successfully. Now, please provide your PAN card details. You can either:\n\n**Upload your PAN card** by clicking the file upload button below\n\n**Enter your PAN card number manually** (10-character alphanumeric code like ABCDE1234F)\n\nPlease choose your preferred option to continue with the loan application process."
           - If user provides a PAN card number (10 character alphanumeric string):
             * Use handle_pan_card_number tool to save the PAN card number
             * After successful save, ask for email address
           - If user provides an email address:
             * Use handle_email_address tool to save the email address
             * If the tool returns status 'success' with 'continue_chain': True:
               * IMMEDIATELY continue with the remaining steps in sequence:
                 - Use pan_verification tool with session_id
                 - Use get_employment_verification tool with session_id  
                 - Use save_employment_details tool with session_id
                 - Use get_bureau_decision tool with session_id
               * Do NOT wait for user input - execute these steps automatically in the chain
               * Return the final formatted response from get_bureau_decision
             * If the tool returns any other status, handle the error appropriately
           - IMMEDIATELY proceed to step 6 after completion

        7. Address Processing:
           - didn't miss this step
           - After processing the prefill data, use the process_address_data tool by session_id  
           - Use pan_verification tool using session_id 
           - CRITICAL: If pan_verification returns status 500 or error:
             * Ask user: "Please provide your PAN card details. You can either:\n\n1. **Upload your PAN card** by clicking the file upload button below\n2. **Enter your PAN card number manually** (10-character alphanumeric code like ABCDE1234F)\n\nPlease choose your preferred option to continue with the loan application process."
             * When user provides PAN card number (10 character alphanumeric string):
               * Use handle_pan_card_number tool to save the PAN card number
           - If pan_verification returns status 200, continue to step 8
           - IMMEDIATELY proceed to step 8 after completion

        8. Employment Verification:
           - didn't miss this step
           - Use get_employment_verification tool to check employment status using session_id
           - CRITICAL: Continue to step 9 regardless of the result (even if it returns status 500 or fails)
           - Do NOT stop the process or ask for Aadhaar upload based on employment verification result
           - IMMEDIATELY proceed to step 9 after completion

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

        CRITICAL CORRECTION HANDLING:
        - When a user says they want to change treatment reason, treatment cost, or date of birth, ask them to provide the new value
        - When they provide the new value, IMMEDIATELY call the appropriate correction tool
        - Do NOT ask for confirmation or additional information - just call the tool with the provided value
        - Examples:
          * User: "I want to change treatment reason" → Ask: "Please provide the new treatment reason"
          * User: "hair transplant" → IMMEDIATELY call correct_treatment_reason tool with "hair transplant"
          * User: "I want to change treatment cost" → Ask: "Please provide the new treatment cost"
          * User: "5000" → IMMEDIATELY call correct_treatment_cost tool with "5000"
          * User: "90000" → IMMEDIATELY call correct_treatment_cost tool with "90000"

        CRITICAL DETAIL UPDATE HANDLING:
        - When a user provides gender selection (Male, Female, 1, 2), IMMEDIATELY call save_gender_details tool
        - When a user provides marital status selection (Married, Unmarried/Single, Yes, No, 1, 2), IMMEDIATELY call save_marital_status_details tool
        - When a user provides education level selection (P.H.D, Graduation, Post graduation, etc.), IMMEDIATELY call save_education_level_details tool
        - Do NOT generate success messages without calling the tools - ALWAYS call the appropriate tool first
        - The tools will automatically format the data to the correct API format
        - Examples:
          * User: "Male" → IMMEDIATELY call save_gender_details tool with "Male"
          * User: "Unmarried/Single" → IMMEDIATELY call save_marital_status_details tool with "Unmarried/Single" (will be formatted to "No")
          * User: "P.H.D" → IMMEDIATELY call save_education_level_details tool with "P.H.D" (will be formatted to "P.H.D.")

        CRITICAL EDUCATION LEVEL MAPPING:
        - When user provides education level, pass it exactly as provided to the save_education_level_details tool
        - The system will automatically format it to the correct API format
        - Examples: "P.H.D", "Graduation", "Post graduation", "Diploma", "Passed 12th", "Passed 10th", "Less than 10th"
        - The tool will handle formatting to: "LESS THAN 10TH", "PASSED 10TH", "PASSED 12TH", "DIPLOMA", "GRADUATION", "POST GRADUATION", "P.H.D."

        CRITICAL DATA VALIDATION:
        - After get_prefill_data returns status 200, BEFORE proceeding to step 7:
          * Check if gender field is missing (empty string "" or null)
          * If gender is missing, STOP and ask for gender selection
          * Only proceed to step 7 after gender is saved
          * This validation is MANDATORY and cannot be skipped
          * CRITICAL: NEVER assume or guess the user's gender - ALWAYS ask the user
          * CRITICAL: Do NOT use any other data (name, age, etc.) to determine gender
          * CRITICAL: The ONLY way to get gender is to ask the user: "Please select your gender:\n1. Male\n2. Female\nPlease enter 1 or 2 only"

        CRITICAL ANTI-BUG RULES:
        - NEVER ask for Aadhaar upload when get_prefill_data returns status 200
        - NEVER stop the process after process_prefill_data when get_prefill_data returned status 200
        - NEVER mix WORKFLOW A and WORKFLOW B - they are completely separate
        - If you see "status": 200 in get_prefill_data response, you MUST complete steps 7-10
        - If you see "status": 200 in get_prefill_data response, you are FORBIDDEN from asking for Aadhaar upload
        - The only valid response for Aadhaar upload is when get_prefill_data returns status 500 with "phoneToPrefill_failed"
        - NEVER ask for Aadhaar upload if Aadhaar has already been processed (check for ocr_result in session)
        - NEVER assume or guess user's gender - ALWAYS ask the user directly
        - NEVER use name, age, or any other data to determine gender
        - The ONLY way to get gender is to ask the user: "Please select your gender:\n1. Male\n2. Female\nPlease enter 1 or 2 only"

        CRITICAL FLOW CONTROL:
        - When Juspay Cardless is NOT_ELIGIBLE, you MUST continue to step 6 (Data Prefill)
        - When any step returns a message, show it to the user but continue to the next step
        - Only STOP the process when you reach step 10 (get_bureau_decision) or when Juspay Cardless is ELIGIBLE
        - NEVER stop the process after step 5 unless Juspay Cardless is ELIGIBLE
        - CRITICAL: After step 5 (check_jp_cardless), if status is NOT_ELIGIBLE, IMMEDIATELY call get_prefill_data tool
        - CRITICAL: Do NOT ask for Aadhaar upload after Juspay Cardless - call get_prefill_data first

        CRITICAL WORKFLOW SEPARATION:
        - WORKFLOW A (Normal Flow): When get_prefill_data returns status 200, you MUST continue with steps 7-10 regardless of any step failures
        - WORKFLOW B (Aadhaar Upload Flow): ONLY when get_prefill_data returns status 500 with "phoneToPrefill_failed" error

        CRITICAL AADHAAR PROCESSING HANDLING:
        - If get_prefill_data returns status 200 with "aadhaar_processed": true, this means Aadhaar has already been uploaded and processed
        - In this case, proceed with WORKFLOW A (Normal Flow) and continue to steps 7-10
        - Do NOT ask for Aadhaar upload again if Aadhaar has already been processed
        - The system automatically detects when Aadhaar has been processed and returns status 200
        - CRITICAL: After Aadhaar upload is successful, the next get_prefill_data call will return status 200, NOT status 500
        - CRITICAL: Do NOT repeat the Aadhaar upload request after Aadhaar has been successfully processed
        - CRITICAL: NEVER mix these workflows - if get_prefill_data returns status 200, you are in WORKFLOW A and must complete steps 7-10
        - CRITICAL: Even if get_employment_verification fails (status 500), continue with save_employment_details and get_bureau_decision in WORKFLOW A
        - CRITICAL: Do NOT ask for Aadhaar upload in WORKFLOW A - that is only for WORKFLOW B
        - CRITICAL: Aadhaar upload request should ONLY appear when get_prefill_data returns status 500 with "phoneToPrefill_failed"
        - CRITICAL: Any other status from get_prefill_data (including other 500 errors) should follow WORKFLOW A

        CRITICAL WORKFLOW ENFORCEMENT:
        - If get_prefill_data returns status 200, you are FORBIDDEN from asking for Aadhaar upload
        - If get_prefill_data returns status 200, you MUST call ALL remaining tools in sequence: process_prefill_data, process_address_data, pan_verification, get_employment_verification, save_employment_details, get_bureau_decision
        - If get_prefill_data returns status 200, you MUST NOT stop until you reach get_bureau_decision
        - The phrase "upload your Aadhaar card" should NEVER appear in your response when get_prefill_data returns status 200
        - WORKFLOW A is a COMPLETE flow - you cannot exit it early or switch to WORKFLOW B

        CRITICAL STEP TRACKING:
        - If user responds with "Successfully processed Aadhaar document and saved details.":
          * Skip steps 1-5 (they are already completed)
          * Start directly from step 6 (Data Prefill)
          * Respond: "Great! Your Aadhaar card has been processed successfully. Now, please provide your PAN card details. You can either:\n\n **Upload your PAN card** by clicking the file upload button below\n\n**Enter your PAN card number manually** (10-character alphanumeric code like ABCDE1234F)\n\nPlease choose your preferred option to continue with the loan application process."
        - If user provides a PAN card number (10 character alphanumeric string):
          * Use handle_pan_card_number tool to save the PAN card number
          * After successful save, ask for email address: "Please provide your email address to continue."
        - If user provides an email address:
          * Use handle_email_address tool to save the email address
          * If the tool returns status 'success' with 'continue_chain': True:
            * IMMEDIATELY continue with the remaining steps in sequence:
              - Use pan_verification tool with session_id
              - Use get_employment_verification tool with session_id  
              - Use save_employment_details tool with session_id
              - Use get_bureau_decision tool with session_id
            * Do NOT wait for user input - execute these steps automatically in the chain
            * Return the final formatted response from get_bureau_decision
          * If the tool returns any other status, handle the error appropriately
          * IMPORTANT: You MUST call get_bureau_decision tool to get the final loan decision - do not generate your own response
        - If gender is missing from prefill data:
          * Ask user: "Please select your gender:\n1. Male\n2. Female\nPlease enter 1 or 2 only"
          * When user provides gender selection (1 or 2):
            * Use save_gender_details tool with "Male" for 1 and "Female" for 2
            * After successful save, continue with normal flow

        CRITICAL PAN VERIFICATION HANDLING:
        - When pan_verification tool returns status 500 or error:
          * Ask user: "Please provide your PAN card details. You can either:\n\n1. **Upload your PAN card** by clicking the file upload button below\n2. **Enter your PAN card number manually** (10-character alphanumeric code like ABCDE1234F)\n\nPlease choose your preferred option to continue with the loan application process."
          * When user provides PAN card number (10 character alphanumeric string):
            * Use handle_pan_card_number tool to save the PAN card number
            * After successful save, ask for email address: "Please provide your email address to continue."
          * When user provides email address:
            * Use handle_email_address tool to save the email address
            * If email tool returns status 'success' with 'continue_chain': True:
              * IMMEDIATELY continue with remaining steps in sequence:
                - Use pan_verification tool with session_id (to verify the saved PAN)
                - Use get_employment_verification tool with session_id  
                - Use save_employment_details tool with session_id
                - Use get_bureau_decision tool with session_id
              * Return the final formatted response from get_bureau_decision
            * If email tool returns any other status, handle the error appropriately
        - When pan_verification tool returns status 200, continue with normal flow

        CRITICAL MISSING DATA HANDLING:
        - When get_prefill_data returns status 200 but gender is missing (empty string or null):
          * STOP the process immediately
          * Ask user: "Please select your gender:\n1. Male\n2. Female\nPlease enter 1 or 2 only"
          * CRITICAL: NEVER assume gender based on name, age, or any other data
          * CRITICAL: The ONLY way to get gender is to ask the user directly
          * When user provides gender selection (1 or 2):
            * Use save_gender_details tool with "Male" for 1 and "Female" for 2
            * After successful save, continue with normal flow (steps 7-10)
        - When get_prefill_data returns status 200 but marital status is missing:
          * Ask user: "Please select your marital status:\n1. Yes (Married)\n2. No (Single)\nPlease enter 1 or 2 only"
          * When user provides marital status selection (1 or 2):
            * Use save_marital_status_details tool with "Yes" for 1 and "No" for 2
            * After successful save, continue with normal flow (steps 7-10)
        - When get_prefill_data returns status 200 but education level is missing:
          * Ask user: "Please select your education level:\n1. LESS THAN 10TH\n2. PASSED 10TH\n3. PASSED 12TH\n4. DIPLOMA\n5. GRADUATION\n6. POST GRADUATION\n7. P.H.D.\nPlease enter 1-7 only"
          * When user provides education level selection (1-7):
            * Map selection to education level and use save_education_level_details tool
            * After successful save, continue with normal flow (steps 7-10)

        CRITICAL BUREAU DECISION HANDLING:
        - You MUST call the get_bureau_decision tool when you need to get the loan decision
        - DO NOT generate your own response for loan decisions - ALWAYS use the get_bureau_decision tool
        - When get_bureau_decision tool returns a formatted response, use it EXACTLY as provided to user and don't modify it
        - Do NOT duplicate the employment type question or any other part of the response
        - Return the complete formatted response from the tool without any modifications
        - If the tool response includes "What is the Employment Type of the patient?", keep it exactly as shown
        - NEVER respond with just "1. SALARIED" or "2. SELF_EMPLOYED" - always return the FULL formatted message
        - DO NOT try to be smart and simplify the response - return it EXACTLY as provided by the tool

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
            logger.info(f"Session {session_id} current status: {current_status}")

            # Helper: detect employment type prompt in a string
            def is_employment_type_prompt(text: str) -> bool:
                return (
                    "What is the Employment Type of the patient?" in text
                    and "1. SALARIED" in text
                    and "2. SELF_EMPLOYED" in text
                    and "Please Enter input 1 or 2 only" in text
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

            logger.info(f"Session {session_id}: Using full agent executor (status: {current_status})")
            session_tools = self._create_session_aware_tools(session_id)

            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
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

            chat_history = self._convert_to_langchain_messages(session.get("history", []))
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
                        if "What is the Employment Type of the patient?" in str(tool_output):
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
                    if bureau_result and "What is the Employment Type of the patient?" in bureau_result:
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
                
                self._update_session_history(session_id, message, bureau_decision_tool_output)
                return bureau_decision_tool_output

            # Check if the agent executor output contains employment type prompt (even if tool wasn't called directly)
            employment_type_prompt_in_output = is_employment_type_prompt(ai_message)

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
            
            return session_id, user_id_from_api if user_id_from_api else "userId not found in API response"
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
            
            # Check if Aadhaar has already been processed and stored
            ocr_result = session.get("data", {}).get("ocr_result")
            if ocr_result and ocr_result.get("name") and ocr_result.get("aadhaar_number"):
                logger.info(f"Aadhaar already processed for user_id: {user_id}, proceeding with prefill data")
                # Aadhaar has been processed, so we should proceed with the normal flow
                # Return a mock 200 response to indicate success
                return json.dumps({
                    "status": 200,
                    "data": {
                        "message": "Aadhaar data already processed, proceeding with application",
                        "aadhaar_processed": True
                    }
                })
                
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
                    "message": "Please upload your Aadhaar card to continue with the loan application process.",
                    "requires_aadhaar_upload": True
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
            # (session_data already set above)
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
                "panCard": ["pan", "panCard", "panNo", "panNumber", "pan_card", "pan_number"],
                "gender": ["gender", "sex"],
                "dateOfBirth": ["dateOfBirth", "dob", "birthDate", "birth_date", "date_of_birth"],
                "emailId": ["emailId", "email", "email_id", "emailAddress", "email_address"],
                "firstName": ["firstName", "name", "first_name", "fullName", "full_name", "givenName", "given_name"]
            }

            for target_field, source_fields in field_mappings.items():
                for source in source_fields:
                    if source in prefill_data and prefill_data[source] is not None:
                        value = prefill_data[source]
                        # Special handling for name fields to ensure we get a string
                        if target_field == "firstName":
                            if isinstance(value, dict):
                                if "fullName" in value and value["fullName"] is not None:
                                    data[target_field] = str(value["fullName"])
                                elif "name" in value and value["name"] is not None:
                                    data[target_field] = str(value["name"])
                                elif "firstName" in value and value["firstName"] is not None:
                                    data[target_field] = str(value["firstName"])
                                else:
                                    continue
                            elif isinstance(value, str):
                                data[target_field] = value
                            else:
                                data[target_field] = str(value)
                        else:
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
                else:
                    data["emailId"] = email_data

            # Also extract from nested "response" if it exists (sometimes API nests again)
            if "response" in prefill_data and isinstance(prefill_data["response"], dict):
                response = prefill_data["response"]
                for target_field, source_fields in field_mappings.items():
                    for source in source_fields:
                        if source in response and response[source] is not None and target_field not in data:
                            value = response[source]
                            if target_field == "firstName":
                                if isinstance(value, dict):
                                    if "fullName" in value and value["fullName"] is not None:
                                        data[target_field] = str(value["fullName"])
                                    elif "firstName" in value and value["firstName"] is not None:
                                        data[target_field] = str(value["firstName"])
                                    elif "name" in value and value["name"] is not None:
                                        data[target_field] = str(value["name"])
                                    else:
                                        continue
                                elif isinstance(value, str):
                                    data[target_field] = value
                                else:
                                    data[target_field] = str(value)
                            else:
                                if isinstance(value, (dict, list)):
                                    continue
                                else:
                                    data[target_field] = str(value)
                            break
                # Special handling for email in nested response
                if "email" in response and response["email"] is not None and "emailId" not in data:
                    email_data = response["email"]
                    if isinstance(email_data, list) and email_data:
                        if isinstance(email_data[0], dict) and "email" in email_data[0] and email_data[0]["email"] is not None:
                            data["emailId"] = email_data[0]["email"]
                        else:
                            data["emailId"] = email_data[0]
                    else:
                        data["emailId"] = email_data
                # Handle phone number in response if needed
                if "mobile" in response and response["mobile"] is not None and "mobileNumber" not in data:
                    data["mobileNumber"] = response["mobile"]

            # Debug log what we're sending
            logger.info(f"Sending to save_basic_details: user_id={user_id}, data={data}")

            # Store the processed data in session for other methods to use
            if session_id:
                for key, value in data.items():
                    if key != "userId":
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
                                city_set = False
                                if pincode_data and pincode_data.get("status") == "success":
                                    # Only update if we get valid non-null data
                                    if pincode_data.get("city") and pincode_data["city"] is not None:
                                        address_data["city"] = pincode_data["city"]
                                        city_set = True
                                    if pincode_data.get("state") and pincode_data["state"] is not None:
                                        address_data["state"] = pincode_data["state"]
                                # If city is not set from API, use last word of address as city
                                if not city_set:
                                    address_str = address_data.get("address", "")
                                    if address_str:
                                        # Split address by whitespace and take last word
                                        address_words = address_str.strip().split()
                                        if address_words:
                                            address_data["city"] = address_words[-1]
                            except Exception as e:
                                logger.warning(f"Failed to get city/state from pincode API: {e}")
                                # If API call fails, try to set city from address as fallback
                                address_str = address_data.get("address", "")
                                if address_str:
                                    address_words = address_str.strip().split()
                                    if address_words:
                                        address_data["city"] = address_words[-1]

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
                return f"""You selected: {selected_option}

What is the Marital Status of the patient?
1. Married
2. Unmarried/Single\n
please Enter input 1 or 2 only"""
            
            # Handle marital status input
            elif collection_step == "marital_status":
                # Check for both number and word inputs
                if "1" in message or "married" in message.lower():
                    additional_details["marital_status"] = "1"
                    selected_option = "Married"
                elif "2" in message or "unmarried" in message.lower() or "single" in message.lower():
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
             
Thank you! Your application is now complete. Loan application decision: {decision_status}. 
Please check your application status by visiting the following:
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
                func=lambda session_id: self.save_basic_details(session_id),
                description="Save user's basic personal details. Call this tool using session_id ",
            ),
            StructuredTool.from_function(
                func=lambda fullName, treatmentCost, userId: self.save_loan_details_structured(fullName, treatmentCost, userId, session_id),
                name="save_loan_details",
                description="Save user's loan details with fullName, treatmentCost, and userId parameters.",
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
                description="Get bureau decision for loan application using session_id. CRITICAL: The response from this tool is the FINAL formatted message that MUST be returned to the user EXACTLY as provided without any modifications.   ",
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
            Tool(
                name="handle_pan_card_number",
                func=lambda pan_number: self.handle_pan_card_number(pan_number, session_id),
                description="Handle PAN card number input and save it to the system. Use this when user provides their PAN card number.",
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
                description="Correct/update the treatment cost in the loan application. Use this when user provides a new treatment cost like '5000', '10000', '90000', etc. (must be >= ₹3,000). Call this tool immediately when user provides a numeric treatment cost amount.",
            ),
            Tool(
                name="correct_date_of_birth",
                func=lambda new_date_of_birth: self.correct_date_of_birth(new_date_of_birth, session_id),
                description="Correct/update the date of birth in the user profile. Use this when user wants to change their date of birth (format: YYYY-MM-DD).",
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
                if bureau_status and (bureau_status.upper() == "APPROVED" or "approved" in bureau_status.lower()):
                    decision_status = "APPROVED"
                    link_to_use = profile_link
                    logger.info(f"Session {session_id}: Fibe AMBER + Bureau APPROVED -> APPROVED with profile link")
                # If bureau INCOME_VERIFICATION_REQUIRED -> INCOME_VERIFICATION_REQUIRED with Fibe link
                elif bureau_status and (bureau_status.upper() == "INCOME_VERIFICATION_REQUIRED" or "income verification required" in bureau_status.lower()):
                    decision_status = "INCOME_VERIFICATION_REQUIRED"
                    link_to_use = fibe_link if fibe_link else profile_link
                    logger.info(f"Session {session_id}: Fibe AMBER + Bureau INCOME_VERIFICATION_REQUIRED -> INCOME_VERIFICATION_REQUIRED with Fibe link")
                    logger.info(f"Session {session_id}: Matched INCOME_VERIFICATION_REQUIRED condition")
                # If bureau REJECTED -> REJECTED with profile link
                elif bureau_status and (bureau_status.upper() == "REJECTED" or "rejected" in bureau_status.lower()):
                    decision_status = "REJECTED"
                    link_to_use = profile_link
                    logger.info(f"Session {session_id}: Fibe AMBER + Bureau REJECTED -> REJECTED with profile link")
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
            
            logger.info(f"Session {session_id}: Final decision - Status: {decision_status}, Link: {link_to_use}")
            logger.info(f"Session {session_id}: Decision logic summary - Fibe: {fibe_status}, Bureau: {bureau_status}, Final: {decision_status}")
            
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
            
            # After successful PAN card save, ask for email address
            return {
                'status': 'success',
                'message': "PAN card number saved successfully. Now, please provide your email address to continue with the loan application process.",
                'data': {'panCard': pan_number},
                'next_step': 'email_collection'
            }
            
        except Exception as e:
            logger.error(f"Error handling PAN card number: {e}")
            return {
                'status': 'error',
                'message': f"Error processing PAN card number: {str(e)}"
            }

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
            
            return {
                'status': 'success',
                'message': "Email address saved successfully. Now continuing with the remaining verification steps automatically...",
                'data': {'emailId': email_address},
                'continue_chain': True,
                'session_id': session_id,
                'next_steps': ['pan_verification', 'get_employment_verification', 'save_employment_details', 'get_bureau_decision']
            }
            
        except Exception as e:
            logger.error(f"Error handling email address: {e}")
            return {
                'status': 'error',
                'message': f"Error processing email address: {str(e)}"
            }

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
                "firstName": result.get('name', ''),
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

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error saving gender details: {e}")
            return f"Error saving gender details: {str(e)}"

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
            new_treatment_cost: The new/corrected treatment cost (must be >= 3000)
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
            new_date_of_birth: The new/corrected date of birth (format: YYYY-MM-DD)
            session_id: Session ID to get user data from
            
        Returns:
            Success or error message
        """
        try:
            # Validate date format
            try:
                datetime.strptime(new_date_of_birth, '%Y-%m-%d')
            except ValueError:
                return "❌ Error: Please enter the date in YYYY-MM-DD format (e.g., 1990-01-15)."
            
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
                "dateOfBirth": new_date_of_birth,
                "mobileNumber": phone_number,
                "userId": user_id
            }
            
            # Call API to update date of birth
            response = self.api_client.save_change_date_of_birth_details(user_id, details)
            
            if response.get("status") == 200:
                # Update session with new date of birth
                SessionManager.update_session_data_field(session_id, "data.dateOfBirth", new_date_of_birth)
                
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
            gender = ocr_result.get('gender', '')
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
            
            if gender:
                details_to_save["gender"] = gender
            
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
            if gender:
                SessionManager.update_session_data_field(session_id, "data.gender", gender)
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
            if gender:
                success_parts.append(f"Gender: {gender}")
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
        if not marital_status:
            return "No"
        
        # Convert to lowercase for easier comparison
        status_lower = marital_status.lower().strip()
        
        # Map various inputs to correct API format
        married_variants = ["married", "yes", "1", "marriage"]
        unmarried_variants = ["unmarried", "single", "no", "2", "unmarried/single", "unmarried/single", "unmarried or single"]
        
        if status_lower in married_variants:
            return "Yes"
        elif status_lower in unmarried_variants:
            return "No"
        else:
            # If it's already in correct format, return as-is
            if marital_status in ["Yes", "No"]:
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