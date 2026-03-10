import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

# 1. Tải tập dữ liệu đã làm sạch
df = pd.read_csv('dataADY201m_cleaned_normalized1.csv')

model = LinearRegression()
results = []
all_intercepts = []
all_coefs = []

# Lặp lại quá trình 3 lần
for i in range(1, 4):
    print(f"\n{'='*70}")
    print(f"ITERATION {i} (RUN {i})")
    print(f"{'='*70}")
    
    # Bước 1: Trộn tập dữ liệu (Ngẫu nhiên 100%)
    df_shuffled = shuffle(df)
    
    # Bước 2: Xuất tập dữ liệu đã trộn ra file CSV
    file_name = f'shuffled_dataset_run_{i}.csv'
    df_shuffled.to_csv(file_name, index=False)
    print(f"Exported file: {file_name}\n")
    
    # Bước 3: Chia tập Train/Test (Tỷ lệ 90/10)
    X = df_shuffled.drop(columns=['exam_score'])
    y = df_shuffled['exam_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    # Bước 4: Xây dựng mô hình (Tìm hàm f(x))
    model.fit(X_train, y_train)
    
    # Trích xuất điểm gốc (intercept) và trọng số (coefficients)
    intercept = model.intercept_
    coefs = model.coef_
    features = X_train.columns
    
    # Lưu lại trọng số để tính trung bình sau này
    all_intercepts.append(intercept)
    all_coefs.append(coefs)
    
    # In công thức dự đoán f(x) trên MỘT DÒNG
    print("PREDICTION FUNCTION f(x):")
    formula_parts = [f"{intercept:.4f}"]
    for coef, feat in zip(coefs, features):
        sign = "+" if coef >= 0 else "-"
        # Xóa ký tự \n ở đây để công thức nằm gọn trên 1 hàng
        formula_parts.append(f" {sign} ({abs(coef):.4f} * {feat})")
    
    full_formula = "f(x) = " + "".join(formula_parts)
    print(full_formula)
    
    # Bước 5: Đánh giá độ chính xác (Chỉ dùng R2 Score)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nACCURACY EVALUATION:")
    print(f"  - Accuracy (R² Score): {r2:.4f}")
    
    # Lưu độ chính xác vào danh sách kết quả
    results.append({
        'Iteration': f'Run {i}',
        'Accuracy (R² Score)': r2
    })

# ======================================================================
# BÁO CÁO CUỐI CÙNG: TRUNG BÌNH f(x) VÀ TRUNG BÌNH ĐỘ CHÍNH XÁC
# ======================================================================

print(f"\n{'='*70}")
print("AVERAGE PREDICTION FUNCTION f(x) ACROSS 3 RUNS")
print(f"{'='*70}")

# Tính trung bình điểm gốc và trọng số
avg_intercept = np.mean(all_intercepts)
avg_coefs = np.mean(all_coefs, axis=0)

avg_formula_parts = [f"{avg_intercept:.4f}"]
for coef, feat in zip(avg_coefs, features):
    sign = "+" if coef >= 0 else "-"
    # Xóa ký tự \n ở đây để công thức trung bình nằm gọn trên 1 hàng
    avg_formula_parts.append(f" {sign} ({abs(coef):.4f} * {feat})")

avg_full_formula = "Average f(x) = " + "".join(avg_formula_parts)
print(avg_full_formula)

print(f"\n{'='*70}")
print("FINAL RESULTS & AVERAGE ACCURACY")
print(f"{'='*70}")

# Tạo DataFrame cho bảng kết quả cuối
results_df = pd.DataFrame(results)
mean_metrics = {
    'Iteration': 'AVERAGE',
    'Accuracy (R² Score)': results_df['Accuracy (R² Score)'].mean()
}

final_df = pd.concat([results_df, pd.DataFrame([mean_metrics])], ignore_index=True)
print(final_df.to_string(index=False))