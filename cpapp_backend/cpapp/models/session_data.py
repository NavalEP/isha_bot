import uuid
from django.db import models


class SessionData(models.Model):
    """
    Model to store session data with loan inquiry information.
    """
    id = models.AutoField(primary_key=True)
    phone_number = models.CharField(max_length=100, null=True, blank=True)
    application_id = models.UUIDField(default=uuid.uuid4, editable=False)
    session_id = models.UUIDField(default=uuid.uuid4, editable=False)
    data = models.JSONField(null=True, blank=True)
    history = models.JSONField(null=True, blank=True)
    status = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'session_data'
        indexes = [
            models.Index(fields=['application_id']),
            models.Index(fields=['session_id']),
        ]

    def __str__(self):
        return f"Session {self.session_id} - Application {self.application_id}"
