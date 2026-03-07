import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. ĐỌC DỮ LIỆU
df=pd.read_csv('D:\ADY201m\dataADY201m_cleaned_normalized.csv')


X = df.drop(columns=['exam_score'])

# Lấy thử 5 cột dữ liệu đầu tiên (hoặc các cột có giá trị lớn nhỏ khác nhau) để dễ nhìn
X_sample = X.iloc[:, :5]

# 2. CHUẨN HÓA DỮ LIỆU
scaler = StandardScaler()
# Tạo DataFrame mới từ dữ liệu đã chuẩn hóa để giữ nguyên tên cột
X_scaled = pd.DataFrame(scaler.fit_transform(X_sample), columns=X_sample.columns)

# 3. VẼ BIỂU ĐỒ TRỰC QUAN SÂU (BOXPLOT SO SÁNH)
# Cài đặt phong cách vẽ đẹp hơn của Seaborn
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- BÊN TRÁI: TRƯỚC KHI CHUẨN HÓA ---
sns.boxplot(data=X_sample, ax=axes[0], palette="Blues")
axes[0].set_title('TRƯỚC KHI CHUẨN HÓA (Dữ liệu Thô)', fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('Giá trị thực tế của dữ liệu', fontsize=12)
axes[0].tick_params(axis='x', rotation=20) # Xoay chữ ở trục X cho đỡ bị đè lên nhau

# --- BÊN PHẢI: SAU KHI CHUẨN HÓA ---
sns.boxplot(data=X_scaled, ax=axes[1], palette="Reds")
axes[1].set_title('SAU KHI CHUẨN HÓA (StandardScaler)', fontsize=14, fontweight='bold', pad=15)
axes[1].set_ylabel('Giá trị Z-score (Mean = 0, Std = 1)', fontsize=12)
axes[1].tick_params(axis='x', rotation=20)

# Thêm tiêu đề tổng
fig.suptitle('SỰ BIẾN ĐỔI CỦA PHÂN PHỐI DỮ LIỆU TRƯỚC VÀ SAU NORMALIZATION', fontsize=16, fontweight='heavy', color='#2c3e50', y=1.05)

plt.tight_layout()
plt.show()