import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ==========================================================
# LOAD DATA
# ==========================================================

TRAIN_FILE = r'/data/train_91_norm.csv'
TEST_FILE = r'/data/test_91_norm.csv'

print("Đang tải dữ liệu...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# One-hot encode
full_df = pd.concat([train_df, test_df], axis=0)
full_df = pd.get_dummies(full_df)

train_df = full_df.iloc[:len(train_df)]
test_df = full_df.iloc[len(train_df):]

X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']

X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# ==========================================================
# DEFINE BASE MODELS
# ==========================================================

rf = RandomForestRegressor(
    n_estimators=800,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42
)

gb = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

et = ExtraTreesRegressor(
    n_estimators=800,
    max_depth=30,
    random_state=42
)

# ==========================================================
# STACKING
# ==========================================================

stack_model = StackingRegressor(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('et', et)
    ],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=1
)

print("Đang train Stacking Model...")
stack_model.fit(X_train, y_train)

# Predict
y_pred = stack_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n🔥 RMSE Stacking: {rmse:.4f}")

if rmse < 8:
    print("🚀 ĐÃ XUỐNG DƯỚI 8 !!!")
else:
    print("⚠ Vẫn chưa xuống 8.")
