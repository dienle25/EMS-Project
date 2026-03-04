import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# =====================================================================
# PHẦN 1: TẢI DỮ LIỆU
# =====================================================================
TRAIN_FILE = 'data/train_91_norm.csv'
TEST_FILE = 'data/test_91_norm.csv'

print("Đang tải dữ liệu...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# =====================================================================
# ENCODE CÁC CỘT DẠNG CHỮ (ONE-HOT ENCODING)
# =====================================================================
full_df = pd.concat([train_df, test_df], axis=0)
full_df = pd.get_dummies(full_df)

train_df = full_df.iloc[:len(train_df), :]
test_df = full_df.iloc[len(train_df):, :]

# =====================================================================
# TÁCH X và y
# =====================================================================
X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']

X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# =====================================================================
# RANDOM FOREST (TUNED VERSION)
# =====================================================================
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=800,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

print("Đang huấn luyện mô hình Random Forest...")
model.fit(X_train, y_train)

# =====================================================================
# ĐÁNH GIÁ
# =====================================================================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n>>> Kết quả RMSE của mô hình: {rmse:.4f}")

if rmse < 8:
    print("=> CHÚC MỪNG! Mô hình đã đạt chỉ tiêu của dự án (RMSE < 8).")
else:
    print("=> Mô hình chưa đạt chỉ tiêu (RMSE >= 8). Cần tinh chỉnh thêm!")
