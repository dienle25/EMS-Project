import sqlite3
import os

db_path = 'db.sqlite3'

if not os.path.exists(db_path):
    print("❌ Không tìm thấy file db.sqlite3")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Kiểm tra bảng ai_model_storage
        cursor.execute("SELECT id, length(model_binary) FROM ai_model_storage")
        rows = cursor.fetchall()

        print("--- THÔNG TIN BẢNG ai_model_storage ---")
        if not rows:
            print("Bảng trống rỗng! Chưa có model nào được lưu.")
        else:
            for row in rows:
                print(f"✅ Đã lưu Model ID: {row[0]} | Kích thước: {row[1]} bytes")

        conn.close()
    except Exception as e:
        print(f"Lỗi: {e} (Có thể bảng chưa được tạo)")