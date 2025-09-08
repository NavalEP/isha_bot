from django.urls import path
from django.http import JsonResponse
from cpapp.api.chat.views import ChatSessionView, ChatMessageView, SessionDetailsView, ShortlinkRedirectView, UserDetailsView, SaveUserBasicDetailsView, SaveUserAddressDetailsView, SaveUserEmploymentDetailsView, DoctorSessionsView, PatientSessionView
from cpapp.api.login.views import SendOtpView, VerifyOtpView, DoctorStaffView
from cpapp.api.document.views import AadhaarUploadView, PanCardUploadView
from cpapp.api.treatment.views import TreatmentSearchView, TreatmentCategoriesView
from cpapp.api.loan.views import (
    GetQrCodeView, ActivitiesLogView, AssignedProductView, BureauDecisionView,
    DisburseDetailReportView, UploadDocumentsView, LoanTransactionsView, MatchingEmiPlansView,
    LoanCountAndAmountView, UserLoanStatusView, GetAllChildClinicsView, GetLoanDetailsByUserIdView,
    GetDoctorDashboardDataView, GetDoctorProfileDetailsView, SaveLoanDetailsView, GetUserAddressView,
    GetAllFinDocDistrictsView, SaveAddressDetailsView
)


def api_root_view(request):
    return JsonResponse({"message": "Agent API root is working"})

urlpatterns = [
    path('', api_root_view, name='api_root'),
    path('session/', ChatSessionView.as_view(), name='create_session'),
    path('message/', ChatMessageView.as_view(), name='send_message'),
    path('doctor-sessions/', DoctorSessionsView.as_view(), name='doctor_sessions'),
    path('patient-sessions/', PatientSessionView.as_view(), name='patient_sessions'),
    path('login/send-otp/', SendOtpView.as_view(), name='send_otp'),
    path('login/verify-otp/', VerifyOtpView.as_view(), name='verify_otp'),
    path('login/doctor-staff/', DoctorStaffView.as_view(), name='doctor_staff'),
    path('session-details/<uuid:session_uuid>/', SessionDetailsView.as_view(), name='session_details'),
    path('user-details/<uuid:session_uuid>/', UserDetailsView.as_view(), name='user_details'),
    path('save-user-basic-details/<uuid:session_uuid>/', SaveUserBasicDetailsView.as_view(), name='save_user_basic_details'),
    path('save-user-address-details/<uuid:session_uuid>/', SaveUserAddressDetailsView.as_view(), name='save_user_address_details'),
    path('save-user-employment-details/<uuid:session_uuid>/', SaveUserEmploymentDetailsView.as_view(), name='save_user_employment_details'),
    path('s/<str:short_code>/', ShortlinkRedirectView.as_view(), name='shortlink_redirect'),
    path('documents/upload/', AadhaarUploadView.as_view(), name='aadhaar_upload'),
    path('documents/upload-pan/', PanCardUploadView.as_view(), name='pan_card_upload'),
    path('treatments/search/', TreatmentSearchView.as_view(), name='treatment_search'),
    path('treatments/categories/', TreatmentCategoriesView.as_view(), name='treatment_categories'),
    
    # Loan API endpoints
    path('getQrCode/', GetQrCodeView.as_view(), name='get_qr_code'),
    path('activitiesLog/', ActivitiesLogView.as_view(), name='activities_log'),
    path('userDetails/getAssignedProductByUserId/', AssignedProductView.as_view(), name='assigned_product'),
    path('bureauDecisionNew/', BureauDecisionView.as_view(), name='bureau_decision'),
    path('getDisburseDetailForReport/', DisburseDetailReportView.as_view(), name='disburse_detail_report'),
    path('uploadDocuments/', UploadDocumentsView.as_view(), name='upload_documents'),
    path('getAllLoanDetailForDoctorNew/', LoanTransactionsView.as_view(), name='loan_transactions'),
    path('matchingEmiPlans/', MatchingEmiPlansView.as_view(), name='matching_emi_plans'),
    path('getLoanCountAndAmountForDoctor/', LoanCountAndAmountView.as_view(), name='loan_count_and_amount'),
    path('status/getUserLoanStatus/', UserLoanStatusView.as_view(), name='user_loan_status'),
    path('getAllChildClinic/', GetAllChildClinicsView.as_view(), name='get_all_child_clinics'),
    path('userDetails/getLoanDetailsByUserId/', GetLoanDetailsByUserIdView.as_view(), name='get_loan_details_by_user_id'),
    path('getDoctorDashboardData/', GetDoctorDashboardDataView.as_view(), name='get_doctor_dashboard_data'),
    path('getDoctorProfDetailsByDoctorId/', GetDoctorProfileDetailsView.as_view(), name='get_doctor_profile_details'),
    path('userDetails/saveLoanDetails/', SaveLoanDetailsView.as_view(), name='save_loan_details'),
    path('userDetails/getUserAddress/', GetUserAddressView.as_view(), name='get_user_address'),
    path('userDetails/addressDetail/', SaveAddressDetailsView.as_view(), name='save_address_details'),
    path('finDoc/allFindocDistricts/', GetAllFinDocDistrictsView.as_view(), name='get_all_findoc_districts'),
]