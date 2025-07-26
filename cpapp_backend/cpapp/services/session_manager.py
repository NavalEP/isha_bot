import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from cpapp.models.session_data import SessionData

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class SessionManager:
    """
    Session management utilities for CarePay Agent
    """
    
    @staticmethod
    def get_session_from_db(session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data from the database
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data dictionary or None if not found
        """
        try:
            session_uuid = uuid.UUID(session_id)
            session_data = SessionData.objects.get(session_id=session_uuid)
            logger.debug(f"Session {session_id} retrieved from database.")
            if not session_data:
                logger.warning(f"Session {session_id} not found in database")
                return None
            
            # History is already in serializable format, no conversion needed
            session = {
                "id": str(session_data.session_id),
                "data": session_data.data or {},
                "history": session_data.history or [],
                "status": session_data.status or "active",
                "created_at": session_data.created_at.isoformat() if session_data.created_at else datetime.now().isoformat(),
                "phone_number": session_data.phone_number
            }
            
            return session
        except SessionData.DoesNotExist:
            logger.warning(f"Session {session_id} not found in database.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving session from database: {e}")
            return None
    
    @staticmethod
    def update_session_in_db(session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Update session data in the database
        
        Args:
            session_id: Session ID
            session_data: Session data dictionary
        """
        try:
            # Convert session_id to UUID if it's a string
            if isinstance(session_id, str):
                session_uuid = uuid.UUID(session_id)
            else:
                session_uuid = session_id
            
            # History is already in serializable format
            history = session_data.get('history', [])
            
            # Update or create session in database
            SessionData.objects.update_or_create(
                session_id=session_uuid,
                defaults={
                    'data': session_data.get('data', {}),
                    'history': history,
                    'status': session_data.get('status', 'active'),
                    'phone_number': session_data.get('phone_number'),
                }
            )
            
            logger.info(f"Session {session_id} updated in database")
        except Exception as e:
            logger.error(f"Error updating session in database: {e}")
    
    @staticmethod
    def update_session_data_field(session_id: str, field_path: str, value: Any) -> None:
        """
        Update a specific field in session data
        
        Args:
            session_id: Session ID
            field_path: Dot-separated path to the field (e.g., "data.userId")
            value: Value to set
        """
        try:
            session = SessionManager.get_session_from_db(session_id)
            if not session:
                logger.error(f"Session {session_id} not found for field update")
                return
            
            # Navigate to the field using the path
            path_parts = field_path.split('.')
            current = session
            
            # Navigate to the parent of the target field
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[path_parts[-1]] = value
            
            # Save back to database
            SessionManager.update_session_in_db(session_id, session)
            
            logger.info(f"Updated field {field_path} in session {session_id}")
        except Exception as e:
            logger.error(f"Error updating session field {field_path}: {e}") 