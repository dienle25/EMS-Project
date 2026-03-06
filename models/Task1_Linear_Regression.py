import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# đường dẫn file
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', '04_Normalized_Data.csv')

print("Loading file:", data_path)

# đọc data
data = pd.read_csv(data_path)

print("Dataset shape:", data.shape)
print(data.head())

# bỏ student_id vì không cần
if "student_id" in data.columns:
    data = data.drop("student_id", axis=1)

# convert text -> number
data = pd.get_dummies(data)

print("\nAfter Encoding:")
print(data.head())

# tách X và y
X = data.drop("exam_score", axis=1)
y = data["exam_score"]

# chia train test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# tạo model
model = LinearRegression()

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# đánh giá
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL RESULT =====")
print("MSE:", mse)
print("R2 Score:", r2)
