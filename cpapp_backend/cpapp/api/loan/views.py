import logging
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from cpapp.services.loan_api_client import LoanAPIClient

logger = logging.getLogger(__name__)

class BaseLoanAPIView(APIView):
    """Base class for loan API views with common functionality"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_client = LoanAPIClient()

class GetQrCodeView(BaseLoanAPIView):
    """Get QR code for doctor"""
    
    def get(self, request):
        doctor_id = request.GET.get('doctorId')
        if not doctor_id:
            return Response({
                'status': 400,
                'message': 'doctorId parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        result = self.api_client.get_qr_code(doctor_id)
        
        if result and result.get('status') == 200:
            return Response({
                'status': 200,
                'data': result.get('data'),
                'attachment': None,
                'message': 'QR code retrieved successfully'
            })
        
        return Response({
            'status': 500,
            'data': None,
            'attachment': None,
            'message': 'Failed to get QR code'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ActivitiesLogView(BaseLoanAPIView):
    """Get activities log for user"""
    
    def get(self, request):
        user_id = request.GET.get('userId')
        if not user_id:
            return Response({
                'status': 400,
                'message': 'userId parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        result = self.api_client.get_activities_log(user_id)
        
        if result and result.get('status') == 200:
            return Response({
                'status': 200,
                'data': result.get('data', []),
                'attachment': None,
                'message': 'Activities log retrieved successfully'
            })
        
        return Response({
            'status': 500,
            'data': [],
            'attachment': None,
            'message': 'Failed to get activities log'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AssignedProductView(BaseLoanAPIView):
    """Get assigned product for user"""
    
    def get(self, request):
        user_id = request.GET.get('userId')
        if not user_id:
            return Response({
                'status': 400,
                'message': 'userId parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        result = self.api_client.get_assigned_product(user_id)
        
        if result and result.get('status') == 200:
            return Response({
                'status': 200,
                'data': result.get('data'),
                'attachment': None,
                'message': 'Assigned product retrieved successfully'
            })
        
        return Response({
            'status': 404,
            'data': None,
            'attachment': None,
            'message': 'No assigned product found'
        }, status=status.HTTP_404_NOT_FOUND)

class BureauDecisionView(BaseLoanAPIView):
    """Get bureau decision for loan"""
    
    def get(self, request):
        loan_id = request.GET.get('loanId')
        if not loan_id:
            return Response({
                'status': 400,
                'message': 'loanId parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        result = self.api_client.get_bureau_decision(loan_id)
        
        if result and result.get('status') == 200:
            return Response({
                'status': 200,
                'data': result.get('data'),
                'attachment': None,
                'message': 'Bureau decision retrieved successfully'
            })
        
        return Response({
            'status': 500,
            'data': None,
            'attachment': None,
            'message': 'Failed to get bureau decision'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DisburseDetailReportView(BaseLoanAPIView):
    """Get disbursal report for user"""
    
    def get(self, request):
        user_id = request.GET.get('userId')
        if not user_id:
            return Response({
                'status': 400,
                'message': 'userId parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        response = self.api_client.get_disburse_detail_report(user_id)
        
        if response and response.status_code == 200:
            # Create response with PDF content
            http_response = HttpResponse(
                response.content,
                content_type='application/pdf'
            )
            http_response['Content-Disposition'] = f'attachment; filename="disbursal-report-{user_id}.pdf"'
            return http_response
        
        return Response({
            'status': 500,
            'data': None,
            'attachment': None,
            'message': 'Failed to download disbursal report'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class UploadDocumentsView(APIView):
    """Upload documents (prescriptions, etc.)"""
    
    parser_classes = (MultiPartParser, FormParser)
    authentication_classes = []  # Disable authentication for file uploads
    permission_classes = []  # Disable permissions
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_client = LoanAPIClient()
    
    def post(self, request):
        try:
            logger.info("UploadDocumentsView: Starting file upload process")
            
            file = request.FILES.get('file')
            user_id = request.POST.get('userId')
            file_type = request.POST.get('type', 'img')
            file_name = request.POST.get('fileName', 'treatmentProof')
            
            logger.info(f"UploadDocumentsView: Received data - user_id: {user_id}, file_type: {file_type}, file_name: {file_name}")
            logger.info(f"UploadDocumentsView: File received: {file.name if file else 'None'}")
            
            if not file or not user_id:
                logger.error("UploadDocumentsView: Missing file or userId")
                return Response({
                    'status': 400,
                    'message': 'file and userId are required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Reset file pointer to beginning
            file.seek(0)
            
            logger.info("UploadDocumentsView: Calling API client upload_documents")
            result = self.api_client.upload_documents(
                file=file,
                user_id=user_id,
                file_type=file_type,
                file_name_param=file_name
            )
            
            logger.info(f"UploadDocumentsView: API client result: {result}")
            
            if result and (result.get('status') == 200 or result.get('status') == '200' or result.get('message') == 'success'):
                logger.info("UploadDocumentsView: Upload successful")
                return Response({
                    'status': 200,
                    'data': result.get('data', ''),
                    'attachment': None,
                    'message': 'Document uploaded successfully'
                })
            
            logger.error(f"UploadDocumentsView: Upload failed - result: {result}")
            return Response({
                'status': 500,
                'data': None,
                'attachment': None,
                'message': result.get('message', 'Failed to upload document')
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        except Exception as e:
            logger.error(f"UploadDocumentsView: Exception occurred: {str(e)}")
            logger.error(f"UploadDocumentsView: Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"UploadDocumentsView: Traceback: {traceback.format_exc()}")
            return Response({
                'status': 500,
                'data': None,
                'attachment': None,
                'message': f'Failed to upload document: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LoanTransactionsView(BaseLoanAPIView):
    """Get loan transactions for doctor"""
    
    def get(self, request):
        doctor_id = request.GET.get('doctorId', '')
        parent_doctor_id = request.GET.get('parentDoctorId', '')
        clinic_name = request.GET.get('clinicName', '')
        start_date = request.GET.get('startDate', '')
        end_date = request.GET.get('endDate', '')
        loan_status = request.GET.get('loanStatus', '')
        
        result = self.api_client.get_loan_transactions(
            doctor_id=doctor_id,
            parent_doctor_id=parent_doctor_id,
            clinic_name=clinic_name,
            start_date=start_date,
            end_date=end_date,
            loan_status=loan_status
        )
        
        if result and result.get('status') == 200:
            return Response({
                'status': 200,
                'data': result.get('data', []),
                'attachment': None,
                'message': 'Loan transactions retrieved successfully'
            })
        
        return Response({
            'status': 404,
            'data': [],
            'attachment': None,
            'message': result.get('message', 'No loan transactions found')
        }, status=status.HTTP_404_NOT_FOUND)

class MatchingEmiPlansView(BaseLoanAPIView):
    """Get matching EMI plans for user and loan"""
    
    def get(self, request):
        user_id = request.GET.get('userId')
        loan_id = request.GET.get('loanId')
        
        if not user_id or not loan_id:
            return Response({
                'status': 400,
                'message': 'userId and loanId parameters are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            result = self.api_client.get_matching_emi_plans(user_id, loan_id)
            
            return Response({
                'status': 200,
                'data': result,
                'attachment': None,
                'message': 'EMI plans retrieved successfully'
            })
            
        except Exception as e:
            logger.error(f"Error getting matching EMI plans: {str(e)}")
            return Response({
                'status': 500,
                'data': {
                    'plans': [],
                    'hasMatchingProduct': False,
                    'isApproved': False,
                    'assignedProductFailed': True
                },
                'attachment': None,
                'message': 'Failed to get matching EMI plans'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LoanCountAndAmountView(BaseLoanAPIView):
    """Get loan count and amount statistics for doctor"""
    
    def get(self, request):
        doctor_id = request.GET.get('doctorId')
        if not doctor_id:
            return Response({
                'status': 400,
                'message': 'doctorId parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get optional clinic name parameter
        clinic_name = request.GET.get('clinicName', '')
        
        result = self.api_client.get_loan_count_and_amount_for_doctor(
            doctor_id=doctor_id,
            clinic_name=clinic_name
        )
        
        if result and result.get('status') == 200:
            return Response({
                'status': 200,
                'data': result.get('data', {}),
                'attachment': None,
                'message': 'Loan count and amount statistics retrieved successfully'
            })
        
        return Response({
            'status': 500,
            'data': {
                'total_loan_amount': 0,
                'pending_count': 0,
                'expired_count': 0,
                'total_applied': 0,
                'expired_amount': 0,
                'disbursed_count': 0,
                'approved_amount': 0,
                'rejected_amount': 0,
                'disbursed_amount': 0,
                'rejected_count': 0,
                'approved_count': 0,
                'pending_amount': 0
            },
            'attachment': None,
            'message': result.get('message', 'Failed to fetch loan count and amount statistics')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UserLoanStatusView(BaseLoanAPIView):
    """Get user loan status"""
    
    def get(self, request):
        loan_id = request.GET.get('loanId')
        if not loan_id:
            return Response({
                'status': 400,
                'message': 'loanId parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        result = self.api_client.get_user_loan_status(loan_id)
        
        if result and result.get('status') == 200:
            return Response({
                'status': 200,
                'data': result.get('data', []),
                'attachment': None,
                'message': 'User loan status retrieved successfully'
            })
        
        return Response({
            'status': 500,
            'data': [],
            'attachment': None,
            'message': result.get('message', 'Failed to fetch user loan status')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GetAllChildClinicsView(BaseLoanAPIView):
    """Get all child clinics for a doctor"""
    
    def get(self, request):
        doctor_id = request.GET.get('doctorId')
        if not doctor_id:
            return Response({
                'status': 400,
                'message': 'doctorId parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        result = self.api_client.get_all_child_clinics(doctor_id)
        
        if result and result.get('status') == 200:
            return Response({
                'status': 200,
                'data': result.get('data', []),
                'attachment': None,
                'message': 'Child clinics retrieved successfully'
            })
        
        return Response({
            'status': 500,
            'data': [],
            'attachment': None,
            'message': result.get('message', 'Failed to fetch child clinics')
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
