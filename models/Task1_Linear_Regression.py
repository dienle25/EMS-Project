import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# LOAD DATA (đổi sang file đúng)
# ==============================

data_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "dataADY201m_cleaned_normalized.csv"
)

print("Loading file:", data_path)

data = pd.read_csv(data_path)

print("Dataset shape:", data.shape)
print(data.head())

# ==============================
# DATA PREPROCESS
# ==============================

# bỏ student_id nếu có
if "student_id" in data.columns:
    data = data.drop("student_id", axis=1)

# chuyển dữ liệu text -> number
data = pd.get_dummies(data)

# ==============================
# SPLIT DATA
# ==============================

X = data.drop("exam_score", axis=1)
y = data["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==============================
# TRAIN MODEL
# ==============================

model = LinearRegression()

model.fit(X_train, y_train)

# ==============================
# PREDICT
# ==============================

y_pred = model.predict(X_test)

# ==============================
# EVALUATE MODEL
# ==============================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL RESULT =====")
print("MSE:", mse)
print("R2 Score:", r2)
