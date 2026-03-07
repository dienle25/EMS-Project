import pandas as pd
from ydata_profiling import ProfileReport

# Đọc file dữ liệu của dự án ADY201m
df=pd.read_csv('D:\ADY201m\dataADY201m_cleaned_normalized.csv')

# Khởi tạo báo cáo (Explorative=True để có phân tích sâu nhất)
profile = ProfileReport(df, title="Báo cáo Phân tích Dữ liệu EWS", explorative=True)

# Xuất ra file HTML
profile.to_file("ews_analysis_report.html")

print("Đã xong! Hãy mở file 'ews_analysis_report.html' để hưởng thụ thành quả.")