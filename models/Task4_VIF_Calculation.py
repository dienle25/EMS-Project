import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os

# 1. Khai báo đường dẫn
# Dùng os.path để đảm bảo chạy được trên mọi máy tính
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, 'notebooks', 'dataADY201m_reduced.csv')

# 2. Đọc dữ liệu
df = pd.read_csv(file_path)

# 3. Chuẩn bị dữ liệu (Loại bỏ biến mục tiêu exam_score)
X = df.drop(columns=['exam_score'])

# --- BƯỚC QUAN TRỌNG: Sửa lỗi TypeError ---
# Chuyển đổi tất cả các cột (bao gồm Boolean) sang kiểu float
X = X.astype(float)

# 4. Thêm hằng số (intercept) - Đây là bước bắt buộc để tính VIF chính xác
X_vif = add_constant(X)

# 5. Tính toán VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

# 6. Hiển thị kết quả
print("\n--- KẾT QUẢ KIỂM TRA ĐA CỘNG TUYẾN (VIF) ---")
# Loại bỏ dòng 'const' khi in ra nếu bạn chỉ muốn xem các features
print(vif_data[vif_data['feature'] != 'const'].sort_values(by="VIF", ascending=False))

# Gợi ý đọc kết quả cho báo cáo:
# VIF < 5: Tốt, không có đa cộng tuyến.
# VIF > 10: Có đa cộng tuyến mạnh, cần xem xét loại bỏ biến.