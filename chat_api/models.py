from django.db import models

class ChatLog(models.Model):
    question = models.TextField()
    answer = models.JSONField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=[('pending_review', 'Pending Review'), ('approved', 'Approved')],
        default='pending_review'
    )
    reviewed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ChatLog #{self.id} - {self.status}"

