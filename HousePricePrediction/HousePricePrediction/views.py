import os
import pandas as pd
import joblib
import random
import sqlite3
import io  # Cần thêm cái này để đọc dữ liệu nhị phân

from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from ews.models import PredictionHistory, UserProfile

# ================= 1. LOAD MÔ HÌNH AI TỪ SQLITE =================
def load_model_from_sqlite():
    """Hàm lấy model mới nhất từ bảng ai_model_storage trong SQLite"""
    try:
        # Đường dẫn tới file db.sqlite3 của Django
        db_path = os.path.join(settings.BASE_DIR, 'db.sqlite3')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Truy vấn lấy model mới nhất (ID cao nhất)
        cursor.execute("SELECT model_binary FROM ai_model_storage ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if row:
            model_binary = row[0]
            # Giải nén dữ liệu nhị phân thành object Python
            ai_data = joblib.load(io.BytesIO(model_binary))
            print("SUCCESS: Da tai AI Model thanh cong tu SQLite!")
            return ai_data
    except Exception as e:
        print(f"WARNING: Khong the load model tu SQLite...: {e}")
    return None

# Thực hiện load ngay khi khởi động server
ai_data = load_model_from_sqlite()
print(f"Dữ liệu ai_data lấy từ DB: {type(ai_data)}")
if ai_data:
    lr_model = ai_data.get('lr_model')
    rf_model = ai_data.get('rf_model')
    ai_metrics = ai_data.get('metrics', {})
else:
    # Nếu DB chưa có model, hệ thống vẫn không bị crash
    lr_model, rf_model, ai_metrics = None, None, {}

def get_ai_metrics():
    return ai_metrics


# ================= 2. QUẢN LÝ TÀI KHOẢN (AUTH) =================
def user_register(request):
    if request.method == "POST":
        u_name = request.POST.get("username")
        p_word = request.POST.get("password")
        if User.objects.filter(username=u_name).exists():
            return render(request, "register.html", {"error": "Tên đăng nhập đã tồn tại!"})
        user = User.objects.create_user(username=u_name, password=p_word)
        UserProfile.objects.get_or_create(user=user, defaults={'role': 'student'})
        return redirect('login')
    return render(request, "register.html")


def user_login(request):
    if request.method == "POST":
        u_name = request.POST.get("username")
        p_word = request.POST.get("password")
        user = authenticate(request, username=u_name, password=p_word)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            return render(request, "login.html", {"error": "Sai tài khoản hoặc mật khẩu!"})
    return render(request, "login.html")


def user_logout(request):
    logout(request)
    return redirect('home')


# ================= 3. TRANG CHỦ & DỰ ĐOÁN =================
def home(request):
    return render(request, "home.html")


@login_required(login_url='/login/')
def predict(request):
    return render(request, "predict.html", {"metrics": get_ai_metrics()})


@login_required(login_url='/login/')
def result(request):
    if lr_model is None or rf_model is None:
        return render(request, "predict.html",
                      {"error": "Lỗi: Không tìm thấy file ai_models.joblib.", "metrics": get_ai_metrics()})

    try:
        # 3.1 LẤY 9 FEATURES CHUẨN XÁC
        study_hours = float(request.GET.get("study_hours", 0))
        class_attendance = float(request.GET.get("class_attendance", 0))
        sleep_hours = float(request.GET.get("sleep_hours", 0))
        sleep_quality = int(request.GET.get("sleep_quality", 0))
        facility_rating = int(request.GET.get("facility_rating", 0))

        study_method = request.GET.get("study_method", "self_study")
        selected_model = request.GET.get("ai_model", "rf")

        # One-hot encoding
        sm_group_study = True if study_method == "group_study" else False
        sm_mixed = True if study_method == "mixed" else False
        sm_online_videos = True if study_method == "online_videos" else False
        sm_self_study = True if study_method == "self_study" else False

        input_data = pd.DataFrame([[
            study_hours, class_attendance, sleep_hours, sleep_quality, facility_rating,
            sm_group_study, sm_mixed, sm_online_videos, sm_self_study
        ]], columns=[
            'study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality',
            'facility_rating', 'study_method_group study', 'study_method_mixed',
            'study_method_online videos', 'study_method_self-study'
        ])

        # 3.2 DỰ ĐOÁN
        if selected_model == "rf":
            prediction = rf_model.predict(input_data)[0]
            model_name = "Random Forest"
        else:
            prediction = lr_model.predict(input_data)[0]
            model_name = "Linear Regression"

        final_score = round(max(0, min(100, prediction)), 2)

        # 3.3 LƯU DATABASE
        PredictionHistory.objects.create(
            student=request.user, study_hours=study_hours, class_attendance=class_attendance,
            sleep_hours=sleep_hours, sleep_quality=sleep_quality, facility_rating=facility_rating,
            study_method=study_method, predicted_score=final_score
        )

        # 3.4 GAMIFICATION: HUY HIỆU & EMOJI & ÂM THANH
        badges = []
        if study_hours >= 0.8: badges.append({"icon": "🔥", "name": "Thánh Cày Cuốc", "color": "danger"})
        if sleep_hours <= 0.3: badges.append({"icon": "🦉", "name": "Cú Đêm", "color": "dark"})
        if class_attendance <= 0.4: badges.append({"icon": "🥷", "name": "Tàng Hình", "color": "secondary"})
        if sleep_quality == 2 and facility_rating == 2: badges.append(
            {"icon": "👑", "name": "Rich Kid", "color": "warning"})
        if len(badges) == 0: badges.append({"icon": "🌱", "name": "Chăm Ngoan", "color": "success"})

        if final_score >= 80:
            warning_level, msg, emoji, sound_effect = "success", "Phong độ xuất sắc!", "😎", "win"
        elif final_score >= 50:
            warning_level, msg, emoji, sound_effect = "warning", "Mức an toàn.", "🤔", "neutral"
        else:
            warning_level, msg, emoji, sound_effect = "danger", "CẢNH BÁO NGUY CƠ RỚT!", "🥶", "lose"

        # 3.5 GAMIFICATION: GACHA HỘP QUÀ
        show_gacha = True if final_score >= 70 else False
        gacha_quotes = [
            "Buff: +100% tự tin khi đi thi! 🚀",
            "Vật phẩm: Thẻ bài Miễn nhiễm Deadline 🃏",
            "Bạn nhận được: Nụ cười của Giảng viên! 🌟",
            "Mở khóa: Bí kíp lụi trắc nghiệm thần chưởng! 🔮"
        ]
        gacha_reward = random.choice(gacha_quotes) if show_gacha else ""

        # 3.6 LEADERBOARD (Top 5 Cao thủ của cả lớp)
        leaderboard = PredictionHistory.objects.order_by('-predicted_score')[:5]

        return render(request, "predict.html", {
            "score": final_score, "warning_level": warning_level, "message": msg,
            "metrics": get_ai_metrics(), "model_used": model_name,
            "badges": badges, "emoji": emoji, "sound_effect": sound_effect,
            "show_gacha": show_gacha, "gacha_reward": gacha_reward,
            "leaderboard": leaderboard
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, "predict.html", {"error": f"Lỗi: {str(e)}", "metrics": get_ai_metrics()})


# ================= 4. DASHBOARD BIỂU ĐỒ =================
@login_required(login_url='/login/')
def dashboard(request):
    history = PredictionHistory.objects.filter(student=request.user).order_by('created_at')
    dates = [r.created_at.strftime("%d/%m %H:%M") for r in history]
    scores = [r.predicted_score for r in history]
    return render(request, "dashboard.html", {
        'dates': dates,
        'scores': scores,
        'history_records': history.order_by('-created_at')
    })