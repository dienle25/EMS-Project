import pandas as pd
import joblib
import sqlite3
import io
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 1. ĐƯỜNG DẪN FILE TỰ ĐỘNG (Đảm bảo trỏ đúng vào thư mục Web)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'notebooks', 'dataADY201m_cleaned_normalized1.csv')
DB_PATH = os.path.join(BASE_DIR, 'HousePricePrediction', 'db.sqlite3')

# 2. ĐỌC VÀ CHỌN ĐÚNG 9 FEATURES
df = pd.read_csv(DATA_PATH)
selected_features = [
    'study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality', 'facility_rating',
    'study_method_group study', 'study_method_mixed', 'study_method_online videos', 'study_method_self-study'
]
target = 'exam_score'

X = df[selected_features]
y = df[target]

print("--- Đang huấn luyện lại mô hình AI (9 features)... ---")
lr_model = LinearRegression()
lr_model.fit(X, y)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

ai_data = {
    'lr_model': lr_model,
    'rf_model': rf_model,
    'metrics': {
        'lr_r2': round(lr_model.score(X, y) * 100, 2),
        'rf_r2': round(rf_model.score(X, y) * 100, 2)
    }
}

# 3. LƯU VÀO DATABASE
try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_model_storage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_binary BLOB
        )
    ''')

    # XÓA SẠCH MODEL CŨ TRONG DB ĐỂ KHÔNG BỊ NHẦM LẪN
    cursor.execute("DELETE FROM ai_model_storage")

    # Lưu model mới
    buffer = io.BytesIO()
    joblib.dump(ai_data, buffer)
    binary_data = buffer.getvalue()

    cursor.execute("INSERT INTO ai_model_storage (model_binary) VALUES (?)", (binary_data,))
    conn.commit()
    conn.close()

    print(f"SUCCESS: Đã xóa model cũ và lưu model mới (9 features) vào: {DB_PATH}")

except Exception as e:
    print(f"ERROR: Lỗi khi lưu vào SQLite: {e}")