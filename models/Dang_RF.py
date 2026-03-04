import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# =====================================================================
# PHẦN 1: TẢI DỮ LIỆU ĐÃ ĐƯỢC LEADER CHUẨN BỊ (Tỷ lệ 9/1)
# =====================================================================
TRAIN_FILE = 'data/train_91_norm.csv'
TEST_FILE = 'data/test_91_norm.csv'

print("Đang tải dữ liệu...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# =====================================================================
# PHẦN 1.1: XỬ LÝ CỘT DẠNG CHỮ (ONE-HOT ENCODING)
# RandomForest chỉ nhận dữ liệu số -> cần chuyển cột text thành số
# =====================================================================
full_df = pd.concat([train_df, test_df], axis=0)
full_df = pd.get_dummies(full_df)

train_df = full_df.iloc[:len(train_df), :]
test_df = full_df.iloc[len(train_df):, :]

# Tách Feature (X) và Target (y)
X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']

X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# =====================================================================
# PHẦN 2: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST REGRESSOR
# =====================================================================
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=1500,      # Số lượng cây
    max_depth=None,         # Không giới hạn độ sâu
    min_samples_split=2,    # Điều kiện tách node
    min_samples_leaf=1,     # Số mẫu tối thiểu tại lá
    max_features=None,      # Sử dụng toàn bộ feature
    bootstrap=True,
    random_state=42,
    n_jobs=-1               # Dùng toàn bộ CPU để train nhanh hơn
)

print("Đang huấn luyện mô hình Random Forest...")
model.fit(X_train, y_train)

# =====================================================================
# PHẦN 3: DỰ ĐOÁN & ĐÁNH GIÁ (GIỮ NGUYÊN THEO YÊU CẦU DỰ ÁN)
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
