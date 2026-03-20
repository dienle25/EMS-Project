import random


def get_student_status(score, study_hours, sleep_hours):
    """Xác định trạng thái và lời khuyên dựa trên AI Score và thói quen thực tế"""

    # 1. Tính toán danh hiệu (Badges)
    badges = []
    if study_hours >= 0.8: badges.append({"icon": "🔥", "name": "Thánh Cày", "color": "danger"})
    if sleep_hours <= 0.3: badges.append({"icon": "🦉", "name": "Cú Đêm", "color": "dark"})
    if score >= 90: badges.append({"icon": "💎", "name": "Huyền Thoại", "color": "info"})
    if len(badges) == 0: badges.append({"icon": "🌱", "name": "Tân Thủ", "color": "success"})

    # 2. Logic Gacha Quà tặng
    gacha_pool = [
        "Thẻ bài: Miễn nhiễm Deadline (Dùng 1 lần) 🃏",
        "Buff: +50% Độ tập trung khi nghe giảng 🎧",
        "Vật phẩm: Ly cà phê bất tử ☕",
        "Skill: Nhìn thấu đề thi (Đang tải...) 🔮"
    ]

    status_data = {
        'badges': badges,
        'gacha_reward': random.choice(gacha_pool) if score >= 70 else None,
        'mascot_mood': 'happy' if score >= 50 else 'crying',
        'is_pro': True if score >= 85 else False
    }
    return status_data