import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# =====================================================================
# PHẦN 1: TẢI DỮ LIỆU ĐÃ ĐƯỢC LEADER CHUẨN BỊ (Tỷ lệ 9/1)
# =====================================================================
TRAIN_FILE = 'train_91_norm.csv'
TEST_FILE = 'test_91_norm.csv'

print("Đang tải dữ liệu...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Tách Feature (X) và Target (y)
X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']

X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# =====================================================================
# PHẦN 2: HUẤN LUYỆN MÔ HÌNH (THÀNH VIÊN VIẾT CODE Ở KHU VỰC NÀY)
# =====================================================================
# Hướng dẫn: Import mô hình được giao và gọi lệnh fit(X_train, y_train)
# Ví dụ mẫu (xóa đi và thay bằng mô hình của bạn):

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# =====================================================================
# PHẦN 3: DỰ ĐOÁN & ĐÁNH GIÁ (GIỮ NGUYÊN CODE NÀY)
# =====================================================================
# 1. Dự đoán trên tập Test
y_pred = model.predict(X_test)

# 2. Tính toán độ lỗi RMSE theo chuẩn yêu cầu dự án
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n>>> Kết quả RMSE của mô hình: {rmse:.4f}")
if rmse < 8:
    print("=> CHÚC MỪNG! Mô hình đã đạt chỉ tiêu của dự án (RMSE < 8).")
else:
    print("=> Mô hình chưa đạt chỉ tiêu (RMSE >= 8). Cần tinh chỉnh thêm!")
