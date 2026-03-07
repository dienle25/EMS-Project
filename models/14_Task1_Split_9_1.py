import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 1. ĐỌC DỮ LIỆU
df=pd.read_csv('D:\ADY201m\dataADY201m_cleaned_normalized.csv')

# 2. TÁCH BIẾN X VÀ y
X = df.drop(columns=['exam_score'])
y = df['exam_score']

# 3. CHIA TẬP DATA THEO TỶ LỆ 9/1 (Test = 10% = 0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ==========================================
# BƯỚC 1: MODELING VỚI DỮ LIỆU THÔ
# ==========================================
model_raw = LinearRegression()
model_raw.fit(X_train, y_train)
y_pred_raw = model_raw.predict(X_test)
r2_raw = r2_score(y_test, y_pred_raw)

# ==========================================
# BƯỚC 2: DATA NORMALIZATION & MODELING LẠI
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Chỉ fit trên tập Train
X_test_scaled = scaler.transform(X_test)       # Áp dụng scale cho tập Test

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

# ==========================================
# BƯỚC 3: IN BÁO CÁO RA MÀN HÌNH (DẠNG BẢNG)
# ==========================================
print("\n" + "="*60)
print(" BÁO CÁO MODELING - TỶ LỆ CHIA 9/1 (TRAIN 90% - TEST 10%) ".center(60))
print("="*60)
print(f"- Tổng số data ban đầu : {len(df)} dòng")
print(f"- Số data dùng để Train: {len(X_train)} dòng (90%)")
print(f"- Số data dùng để Test : {len(X_test)} dòng (10%)")
print(f"- Thuật toán sử dụng   : Linear Regression")
print("-" * 60)
print(f"👉 Độ chính xác (Dữ liệu THÔ)       : {r2_raw:.4f}")
print(f"👉 Độ chính xác (Đã CHUẨN HÓA)      : {r2_scaled:.4f}")
print("="*60 + "\n")

# ==========================================
# BƯỚC 4: VẼ BIỂU ĐỒ TRỰC QUAN
# ==========================================
labels = ['Dữ liệu Thô', 'Đã Chuẩn Hóa (StandardScaler)']
scores = [r2_raw, r2_scaled]
colors = ['#3498db', '#e74c3c']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, scores, color=colors, width=0.5)

# Trang trí biểu đồ
ax.set_ylabel('Độ chính xác (R² Score)', fontsize=12)
ax.set_title('So sánh R² Score - Tỷ lệ chia 9/1', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, max(scores) * 1.2) # Tạo khoảng trống trên đỉnh cột
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Ghi chú con số cụ thể lên đỉnh cột
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # Đẩy text lên trên 5 points
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()