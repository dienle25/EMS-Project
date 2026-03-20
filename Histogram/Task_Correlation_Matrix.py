import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. NẠP DỮ LIỆU ĐÃ CHUẨN HÓA
# ==========================================
# Mình để đường dẫn chung để khớp với file bạn vừa upload
# Bạn có thể đổi lại thành r'D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv' trên máy bạn nhé
file_path = r'D:\ADY201m\notebooks\dataADY201m_cleaned_normalized1.csv'
df = pd.read_csv(file_path)

print("="*70)
print("ĐANG TÍNH TOÁN MA TRẬN TƯƠNG QUAN (CORRELATION MATRIX)")
print("="*70)

# ==========================================
# 2. TÍNH TOÁN VÀ SẮP XẾP ĐỘ TƯƠNG QUAN VỚI EXAM_SCORE
# ==========================================
# Tính ma trận Pearson Correlation cho toàn bộ các cột
corr_matrix = df.corr()

# Rút trích riêng cột 'exam_score' để xem biến nào ảnh hưởng điểm thi nhất
target_corr = corr_matrix['exam_score'].sort_values(ascending=False)
target_corr_df = pd.DataFrame(target_corr).reset_index()
target_corr_df.columns = ['Feature', 'Correlation with Exam Score']

print("\n🚀 TOP CÁC YẾU TỐ KÉO ĐIỂM SỐ LÊN (POSITIVE CORRELATION):")
print(target_corr_df.head(6).to_string(index=False))

print("\n⚠️ TOP CÁC YẾU TỐ KÉO ĐIỂM SỐ XUỐNG (NEGATIVE CORRELATION):")
print(target_corr_df.tail(5).to_string(index=False))

# ==========================================
# 3. VẼ BIỂU ĐỒ HEATMAP TRỰC QUAN (GIỐNG HÌNH MẪU)
# ==========================================
print("\nĐang xuất biểu đồ Heatmap...")

# Tăng kích thước canvas lên "siêu to khổng lồ" (20x16)
plt.figure(figsize=(20, 16))

# Đã XÓA dòng tạo mask vì hình mẫu hiện Full ma trận
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# CẬP NHẬT:
# - Bỏ mask
# - Đổi annot=True (để hiện số)
# - Thêm fmt=".2f" (để làm tròn 2 chữ số thập phân giống hình)
# - Thêm annot_kws={"size": 10} để chữ số bên trong không bị quá to và đè lên nhau
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 fmt=".2f",
                 cmap='coolwarm',
                 vmax=1, vmin=-1, center=0,
                 linewidths=.5,
                 cbar_kws={"shrink": .8},
                 annot_kws={"size": 10})

# Xoay nhãn trục X nghiêng 45 độ, căn lề phải để chữ không chạm nhau
plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='500')
plt.yticks(fontsize=11, fontweight='500')

plt.title('Correlation Matrix of Student Performance',
          fontsize=22, fontweight='bold', pad=30)

# Dùng tính năng tự động canh lề chuẩn của Matplotlib
plt.tight_layout()
image_name = 'Correlation_Matrix_Heatmap.png'
plt.savefig(image_name, dpi=300, bbox_inches='tight')

# Hiển thị luôn hình ảnh lên màn hình (nếu bạn dùng Jupyter Notebook hoặc IDE hỗ trợ)
plt.show()

print(f"[THÀNH CÔNG] Đã lưu biểu đồ sắc nét thành file '{image_name}'.")