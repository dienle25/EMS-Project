import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# PHẦN 1: TẢI DỮ LIỆU
# =====================================================================

TRAIN_FILE = 'data/train_91_norm.csv'
TEST_FILE = 'data/test_91_norm.csv'

print("Đang tải dữ liệu...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Tách feature (X) và nhãn (y)
X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']

X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# =====================================================================
# PHẦN 2: XÂY DỰNG MÔ HÌNH RANDOM FOREST
# =====================================================================

model = RandomForestRegressor(
    n_estimators=2000,      # Số lượng cây lớn để tăng độ ổn định
    max_depth=None,        # Không giới hạn độ sâu
    min_samples_split=2,   # Tách node tối thiểu 2 mẫu
    min_samples_leaf=1,    # Lá tối thiểu 1 mẫu
    max_features='sqrt',   # Lấy căn bậc 2 số feature tại mỗi split
    random_state=42,
    n_jobs=-1              # Dùng tất cả CPU để train nhanh hơn
)

print("Đang huấn luyện mô hình Random Forest...")
model.fit(X_train, y_train)

# =====================================================================
# PHẦN 3: DỰ ĐOÁN & ĐÁNH GIÁ
# =====================================================================

y_pred = model.predict(X_test)

# Tính RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n>>> Kết quả RMSE của mô hình: {rmse:.4f}")

if rmse < 8:
    print("=> CHÚC MỪNG! Mô hình đã đạt chỉ tiêu (RMSE < 8).")
else:
    print("=> Mô hình chưa đạt chỉ tiêu (RMSE >= 8).")
