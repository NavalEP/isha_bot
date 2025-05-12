import uuid
from django.db import models


class SessionData(models.Model):
    """
    Model to store session data with loan inquiry information.
    """
    application_id = models.UUIDField(primary_key=False, default=uuid.uuid4, editable=False)
    session_id = models.UUIDField(default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(null=True, blank=True)
    data = models.JSONField(null=True, blank=True)
    loan_inquiry = models.JSONField(null=True, blank=True)
    status = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'session_data'
        indexes = [
            models.Index(fields=['application_id']),
            models.Index(fields=['session_id']),
            models.Index(fields=['user_id']),
        ]

    def __str__(self):
        return f"Session {self.session_id} - Application {self.application_id}"
