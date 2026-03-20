import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 1. Tải dữ liệu
df = pd.read_csv(r'D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv')

# 2. Loại bỏ biến phụ thuộc (Ví dụ: 'exam_score') để xét các biến độc lập
X = df.drop(columns=['exam_score'])

# 3. Chuyển đổi các cột kiểu boolean (True/False) sang số thực (float) để tính toán
X = X.astype(float)

# 4. Thêm hằng số (constant/intercept) vào dataframe (Khuyến nghị để tính VIF chuẩn xác)
X_with_const = add_constant(X)

# 5. Khởi tạo DataFrame để lưu kết quả và tính VIF cho từng biến
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

# 6. Loại bỏ dòng 'const' ra khỏi kết quả hiển thị cuối cùng
vif_data = vif_data[vif_data["Feature"] != "const"].reset_index(drop=True)

# Hiển thị kết quả
print(vif_data)