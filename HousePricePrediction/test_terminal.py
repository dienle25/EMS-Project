import sqlite3
import joblib
import io
import pandas as pd
import os


def test_load_and_predict():
    print("--- BẮT ĐẦU KIỂM TRA HỆ THỐNG AI ---")

    # 1. Kết nối thử tới Database
    db_path = 'db.sqlite3'
    if not os.path.exists(db_path):
        print(f"❌ Lỗi: Không tìm thấy file {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 2. Lấy dữ liệu model mới nhất
        print("1. Đang truy vấn Database...")
        cursor.execute("SELECT id, model_binary FROM ai_model_storage ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if not row:
            print("❌ Lỗi: Bảng 'ai_model_storage' đang trống. Bạn chưa chạy Task lưu model vào DB.")
            return

        model_id, model_binary = row
        print(f"✅ Tìm thấy Model ID: {model_id}")

        # 3. Load model bằng Joblib
        ai_data = joblib.load(io.BytesIO(model_binary))

        # Kiểm tra xem ai_data có phải là dict không
        if not isinstance(ai_data, dict):
            print("❌ Lỗi: Dữ liệu trong DB không phải là Dictionary (lr_model, rf_model, metrics).")
            return

        rf_model = ai_data.get('rf_model')
        print("✅ Load Random Forest Model thành công!")

        # 4. Giả lập dữ liệu đầu vào (9 features chuẩn)
        # Lưu ý: Các giá trị này phải khớp với thang đo lúc bạn Train
        test_input = pd.DataFrame([[
            0.5, 0.8, 0.6, 1, 1,  # study_hours, attendance, sleep_hours, sleep_quality, facility
            False, False, True, False  # 4 cột One-hot encoding của study_method
        ]], columns=[
            'study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality',
            'facility_rating', 'study_method_group study', 'study_method_mixed',
            'study_method_online videos', 'study_method_self-study'
        ])

        # 5. Chạy dự đoán thử
        print("2. Đang chạy dự đoán thử nghiệm...")
        prediction = rf_model.predict(test_input)[0]
        print(f"🚀 KẾT QUẢ DỰ ĐOÁN: {round(prediction, 2)} điểm")
        print("--- KIỂM TRA HOÀN TẤT: HỆ THỐNG AI HOẠT ĐỘNG TỐT ---")

    except Exception as e:
        print(f"❌ Lỗi phát sinh: {e}")


if __name__ == "__main__":
    test_load_and_predict()