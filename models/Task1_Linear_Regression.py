import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle

# 1. Đọc dữ liệu
df = pd.read_csv('04_Normalized_Data.csv')

# Xác định features (X) và target (y)
X = df.drop(columns=['exam_score'])
y = df['exam_score']

# 2. Phân loại các cột để cấu hình Preprocessor
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 3. Tạo Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        # Loại bỏ cột student_id
        ('drop_id', 'drop', ['student_id']),
        # Mã hoá One-Hot các cột phân loại
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    # Giữ nguyên các cột số còn lại
    remainder='passthrough'
)

# 4. Tạo Pipeline kết hợp preprocessor và Linear Regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Lưu trữ kết quả của các lần chạy
results = []

# 5. Vòng lặp 3 lần đánh giá
for i in range(3):
    # Bước 5.1: Shuffle dataset (BỎ random_state để trộn ngẫu nhiên 100% qua mỗi vòng lặp)
    X_shuffled, y_shuffled = shuffle(X, y)
    
    # Bước 5.2: Split dataset 90% train - 10% test (BỎ random_state để chia ngẫu nhiên)
    X_train, X_test, y_train, y_test = train_test_split(
        X_shuffled, y_shuffled, 
        test_size=0.1
    )
    
    # Bước 5.3: Huấn luyện mô hình (Tìm hàm số f(x))
    pipeline.fit(X_train, y_train)
    
    # Bước 5.4: Dự đoán trên tập Test
    y_pred = pipeline.predict(X_test)
    
    # Bước 5.5: Đánh giá độ chính xác
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Lưu kết quả
    results.append({
        'Lần lặp': f'Lần {i + 1}',
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    })

# 6. Tạo bảng và tính trung bình
results_df = pd.DataFrame(results)

# Tính trung bình các cột số
mean_metrics = {
    'Lần lặp': 'Trung bình',
    'R2': results_df['R2'].mean(),
    'RMSE': results_df['RMSE'].mean(),
    'MAE': results_df['MAE'].mean()
}

# Thêm hàng trung bình vào cuối bảng
final_df = pd.concat([results_df, pd.DataFrame([mean_metrics])], ignore_index=True)

# In kết quả
print("Đánh giá độ chính xác (shuffle ngẫu nhiên hoàn toàn):\n")
print(final_df.to_string(index=False))
