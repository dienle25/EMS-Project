import pandas as pd
import joblib
import sqlite3
import io
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

print("⏳ Đang đọc dữ liệu từ file CSV...")
df = pd.read_csv('dataADY201m_reduced.csv')

# Lấy đủ 9 features như ban đầu
X = df[['study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality',
        'facility_rating', 'study_method_group study', 'study_method_mixed',
        'study_method_online videos', 'study_method_self-study']]
y = df['exam_score']

print("🧠 Đang huấn luyện các mô hình Machine Learning...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

ai_data = {
    'lr_model': lr_model,
    'rf_model': rf_model,
    'metrics': {
        'lr_r2': round(r2_score(y_test, lr_preds) * 100, 2),
        'lr_mse': round(mean_squared_error(y_test, lr_preds), 2),
        'rf_r2': round(r2_score(y_test, rf_preds) * 100, 2),
        'rf_mse': round(mean_squared_error(y_test, rf_preds), 2)
    }
}

print("📦 Đang đóng gói và lưu mô hình vào Database...")
try:
    buffer = io.BytesIO()
    joblib.dump(ai_data, buffer)
    model_binary = buffer.getvalue()

    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO ai_model_storage (model_binary) VALUES (?)", (model_binary,))
    
    conn.commit()
    conn.close()
    print("✅ THÀNH CÔNG! Đã lưu mô hình AI vào Database (bảng ai_model_storage)!")
except Exception as e:
    print(f"❌ Lỗi khi lưu vào DB: {e}")