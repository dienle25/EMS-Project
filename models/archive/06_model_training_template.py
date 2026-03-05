import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os  # Thêm thư viện này

# =====================================================================
# PHẦN 1: TẢI DỮ LIỆU (ĐƯỜNG DẪN TỰ ĐỘNG THÍCH ỨNG MỌI MÁY)
# =====================================================================

# 1. Lấy đường dẫn tuyệt đối của thư mục đang chứa file code này (thư mục 'models')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Lùi ra một cấp để về thư mục gốc của dự án
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 3. Tự động nối đường dẫn từ thư mục gốc vào thư mục 'data'
TRAIN_FILE = os.path.join(PROJECT_ROOT, 'data', 'train_91_norm.csv')
TEST_FILE = os.path.join(PROJECT_ROOT, 'data', 'test_91_norm.csv')

print(f"Đang tải dữ liệu từ: {TRAIN_FILE}")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)