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

# =====================================================================
# PHẦN 2: ENCODE CATEGORICAL (QUAN TRỌNG)
# =====================================================================
# Gộp train + test để đảm bảo encode giống nhau

full_df = pd.concat([train_df, test_df], axis=0)

# One-hot encoding cho các cột dạng string
full_df = pd.get_dummies(full_df)

# Tách lại train và test
train_df = full_df.iloc[:len(train_df), :]
test_df = full_df.iloc[len(train_df):, :]

# =====================================================================
# PHẦN 3: TÁCH X VÀ y
# =====================================================================

X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']

X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# =====================================================================
# PHẦN 4: RANDOM FOREST
# =====================================================================

model = RandomForestRegressor(
    n_estimators=1500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("Đang huấn luyện mô hình Random Forest...")
model.fit(X_train, y_train)

# =====================================================================
# PHẦN 5: ĐÁNH GIÁ
# =====================================================================

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n>>> Kết quả RMSE của mô hình: {rmse:.4f}")

if rmse < 8:
    print("=> CHÚC MỪNG! Mô hình đạt chỉ tiêu (RMSE < 8).")
else:
    print("=> Mô hình chưa đạt chỉ tiêu (RMSE >= 8).")
