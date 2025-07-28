from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from cpapp.services.document_service import DocumentService
from cpapp.services.agent import CarepayAgent
import logging
import tempfile
import os
from pdf2image import convert_from_path
import mimetypes

logger = logging.getLogger(__name__)

class AadhaarUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.document_service = DocumentService()
        self.agent = CarepayAgent()
    
    def post(self, request):
        """
        Handle Aadhaar document upload and processing
        """
        try:
            # Check if document file is uploaded
            if 'document' not in request.FILES:
                return Response(
                    {'error': 'No file uploaded. Please select a document to upload.'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Check if session_id is provided
            session_id = request.data.get('session_id')
            if not session_id:
                return Response(
                    {'error': 'Session ID is required. Please provide a valid session ID.'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate session_id format (should be a UUID)
            try:
                import uuid
                uuid.UUID(session_id)
            except ValueError:
                return Response(
                    {'error': 'Invalid session ID format. Please provide a valid session ID.'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            file_obj = request.FILES['document']
            logger.info(f"Received file: {file_obj.name}, size: {file_obj.size} bytes")
            tmp_path = None
            
            try:
                # Validate file
                is_valid, message = self.document_service.validate_file(file_obj)
                if not is_valid:
                    return Response(
                        {'error': message}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Get file type
                file_type = self.document_service.get_file_type(file_obj)
                logger.info(f"Processing file of type: {file_type}")
                
                # Save to temporary file for OCR
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as tmp:
                    for chunk in file_obj.chunks():
                        tmp.write(chunk)
                    tmp_path = tmp.name
                    logger.info(f"Saved temporary file: {tmp_path}")

                # Convert PDF to image if needed
                if file_type == 'application/pdf':
                    try:
                        logger.info("Converting PDF to image")
                        images = convert_from_path(tmp_path)
                        if images:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as img_tmp:
                                images[0].save(img_tmp.name, 'JPEG')
                                os.unlink(tmp_path)
                                tmp_path = img_tmp.name
                                logger.info(f"PDF converted to image: {tmp_path}")
                    except Exception as e:
                        logger.error(f"PDF conversion error: {e}")
                        return Response(
                            {'error': 'Failed to process PDF file'}, 
                            status=status.HTTP_400_BAD_REQUEST
                        )

                # Process the document using the agent with OCR
                logger.info(f"Processing document with agent for session: {session_id}")
                agent_result = self.agent.handle_aadhaar_upload(tmp_path, session_id)
                logger.info(f"Agent result: {agent_result}")
                
                # Check if agent processing was successful
                if agent_result.get('status') == 'success':
                    return Response({
                        'status': 'success',
                        'message': agent_result.get('message', 'Document processed successfully'),
                        'data': {
                            'ocr_result': agent_result.get('data', {}),
                            'basic_details_saved': True,
                            'address_saved': bool(agent_result.get('data', {}).get('address'))
                        }
                    })
                elif agent_result.get('status') == 'warning':
                    return Response({
                        'status': 'warning',
                        'message': agent_result.get('message', 'Document processed with warnings'),
                        'data': {
                            'ocr_result': agent_result.get('data', {}),
                            'basic_details_saved': True,
                            'address_saved': False
                        }
                    })
                else:
                    return Response({
                        'status': 'error',
                        'message': agent_result.get('message', 'Failed to process document'),
                        'data': {
                            'ocr_result': agent_result.get('data', {})
                        }
                    }, status=status.HTTP_400_BAD_REQUEST)
                
            except Exception as e:
                logger.error(f"Document processing error: {e}")
                return Response(
                    {'error': f'Error processing document: {str(e)}'}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
            finally:
                # Clean up temporary file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                        logger.info(f"Cleaned up temporary file: {tmp_path}")
                    except Exception as e:
                        logger.error(f"Error deleting temp file: {e}")
                        
        except Exception as e:
            logger.error(f"Document upload error: {e}")
            return Response(
                {'error': f'Upload error: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )