import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# =====================================================================
# PHẦN 1: TẢI DỮ LIỆU (Đường dẫn tự động an toàn)
# =====================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

TRAIN_FILE = os.path.join(PROJECT_ROOT, 'data', 'train_91_norm.csv')
TEST_FILE = os.path.join(PROJECT_ROOT, 'data', 'test_91_norm.csv')

print("Đang tải dữ liệu...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']
X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# =====================================================================
# PHẦN 2: THIẾT LẬP LƯỚI THAM SỐ (THÀNH VIÊN TỰ SỬA PHẦN NÀY)
# =====================================================================
# HƯỚNG DẪN:
# 1. Import model của bạn (Random Forest hoặc Gradient Boosting)
# 2. Khai báo param_grid với các con số muốn thử nghiệm

from sklearn.ensemble import GradientBoostingRegressor # VD: Thay bằng thuật toán của bạn
model = GradientBoostingRegressor(random_state=42)

# Khai báo các tổ hợp tham số muốn dò nghiệm (Cẩn thận: Càng nhiều càng chạy lâu!)
param_grid = {
    'n_estimators': [100, 200, 300],       # Số lượng cây
    'max_depth': [3, 5, 7],                # Độ sâu của cây
    'learning_rate': [0.01, 0.05, 0.1]     # Tốc độ học (Chỉ dùng cho Gradient Boosting)
}

# =====================================================================
# PHẦN 3: HUẤN LUYỆN VÀ TÌM KIẾM TỰ ĐỘNG
# =====================================================================
print(f"Bắt đầu dò tìm tham số tối ưu. Quá trình này có thể mất 2-5 phút...")
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,                      # Chia tập train thành 3 phần để test chéo
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,                 # Huy động tối đa sức mạnh CPU của máy tính
    verbose=2
)

grid_search.fit(X_train, y_train)

# =====================================================================
# PHẦN 4: KẾT QUẢ CHUNG CUỘC
# =====================================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*50)
print("🏆 BỘ THAM SỐ TỐT NHẤT TÌM ĐƯỢC:")
print(grid_search.best_params_)
print("="*50)
print(f">>> RMSE TỐI ƯU CỦA MÔ HÌNH NÀY LÀ: {final_rmse:.4f}")

if final_rmse < 8:
    print("=> CHÚC MỪNG! ĐÃ PHÁ ĐẢO MỤC TIÊU DỰ ÁN (RMSE < 8) 🎉")
else:
    print("=> Vẫn chưa đạt < 8. Hãy thử thay đổi các con số trong param_grid!")