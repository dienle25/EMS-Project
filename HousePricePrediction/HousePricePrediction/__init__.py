import sqlite3
import joblib
import io
import os


def init_db_and_save_model():
    db_path = 'db.sqlite3'
    # Đường dẫn tới file model bạn đã train (sửa lại cho đúng tên file bạn đang có)
    model_src = os.path.join('..', 'models', 'linear_model.pkl')

    if not os.path.exists(model_src):
        print(f"❌ Không tìm thấy file model tại: {model_src}")
        return

    try:
        # 1. Kết nối SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 2. Tạo bảng ai_model_storage
        print("1. Đang tạo bảng ai_model_storage...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_model_storage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_binary BLOB
            )
        ''')

        # 3. Đóng gói model (Giả lập cấu trúc dict mà views.py đang yêu cầu)
        print("2. Đang đóng gói model vào Dictionary...")
        loaded_model = joblib.load(model_src)

        # Đóng gói đúng các key mà views.py của bạn đang gọi: lr_model, rf_model, metrics
        ai_data = {
            'lr_model': loaded_model,  # Tạm thời gán cả 2 là 1 model để test
            'rf_model': loaded_model,
            'metrics': {'r2': 0.85}
        }

        # Chuyển sang binary
        buffer = io.BytesIO()
        joblib.dump(ai_data, buffer)
        binary_data = buffer.getvalue()

        # 4. Chèn vào Database
        cursor.execute("INSERT INTO ai_model_storage (model_binary) VALUES (?)", (binary_data,))

        conn.commit()
        conn.close()
        print("✅ ĐÃ KHỞI TẠO BẢNG VÀ LƯU MODEL THÀNH CÔNG!")
        print("Giờ bạn hãy chạy lại file test_ai_terminal.py hoặc chạy Server Django.")

    except Exception as e:
        print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    init_db_and_save_model()