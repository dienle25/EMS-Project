import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# =====================================================================
# PHẦN 1: TẢI DỮ LIỆU ĐÃ ĐƯỢC LEADER CHUẨN BỊ (Tỷ lệ 9/1)
# =====================================================================
TRAIN_FILE = r'/data/train_91_norm.csv'
TEST_FILE = r'/data/test_91_norm.csv'

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
# 1. Xác định các cột phân loại (categorical columns)
categorical_cols = X_train.select_dtypes(include=['object']).columns

# 2. Tạo preprocessor để mã hoá OneHot các cột phân loại và drop cột student_id
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('drop_id', 'drop', ['student_id'])
    ],
    remainder='passthrough'
)

# 3. Kết hợp tiền xử lý và mô hình vào trong một Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# 4. Huấn luyện mô hình
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