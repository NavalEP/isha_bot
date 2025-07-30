from django.db import models
from django.utils import timezone


class Treatment(models.Model):
    """
    Model to store treatment information including name, category, and additional name field
    """
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255, help_text="Treatment name")
    category = models.CharField(max_length=100, help_text="Treatment category")
    created_at = models.DateTimeField(default=timezone.now, null=True, blank=True, help_text="Creation timestamp")
    updated_at = models.DateTimeField(default=timezone.now, null=True, blank=True, help_text="Last update timestamp")

    class Meta:
        db_table = 'treatments'
        verbose_name = 'Treatment'
        verbose_name_plural = 'Treatments'
        ordering = ['category', 'name']

    def __str__(self):
        return f"{self.category} - {self.name}"

    def __repr__(self):
        return f"<Treatment: {self.name} ({self.category})>"
    
    def save(self, *args, **kwargs):
        # Update the updated_at field on every save
        self.updated_at = timezone.now()
        super().save(*args, **kwargs)