from django.db import models
from django.contrib.auth.models import User


# Bảng mở rộng thông tin User (Phân quyền)
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    ROLE_CHOICES = (
        ('student', 'Sinh viên'),
        ('teacher', 'Giảng viên'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='student')

    def __str__(self):
        return f"{self.user.username} - {self.role}"


# Bảng lưu lịch sử dự đoán của Sinh viên
class PredictionHistory(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE)

    # Các feature đầu vào
    study_hours = models.FloatField()
    class_attendance = models.FloatField()
    sleep_hours = models.FloatField()
    sleep_quality = models.IntegerField()
    facility_rating = models.IntegerField()
    study_method = models.CharField(max_length=50)

    # Kết quả AI trả về
    predicted_score = models.FloatField()

    # Thời gian dự đoán
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.username} - {self.predicted_score} điểm ({self.created_at.strftime('%d/%m/%Y')})"