import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# =====================================================================
# PHẦN 1: TẢI DỮ LIỆU
# =====================================================================

TRAIN_FILE = r'/data/train_91_norm.csv'
TEST_FILE = r'/data/test_91_norm.csv'

print("Đang tải dữ liệu...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Tách X và y
X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']

X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# =====================================================================
# XỬ LÝ TRƯỜNG HỢP CÒN STRING (AN TOÀN TUYỆT ĐỐI)
# =====================================================================

# Encode nếu còn biến dạng object (ví dụ: female/male)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Đồng bộ cột giữa train và test
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# =====================================================================
# PHẦN 2: HUẤN LUYỆN MÔ HÌNH (Decision Tree - Khôi)
# =====================================================================

model = DecisionTreeRegressor(
    random_state=42,
    max_depth=None
)

model.fit(X_train, y_train)

# =====================================================================
# PHẦN 3: DỰ ĐOÁN & ĐÁNH GIÁ
# =====================================================================

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n>>> Kết quả RMSE của mô hình Decision Tree: {rmse:.4f}")

if rmse < 8:
    print("=> CHÚC MỪNG! Mô hình đã đạt chỉ tiêu của dự án (RMSE < 8).")
else:
    print("=> Mô hình chưa đạt chỉ tiêu (RMSE >= 8). Cần tinh chỉnh thêm!")