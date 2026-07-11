import os
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMAGE_DATASET_DIR = r"D:\Phan_tich_va_nhan_dang_mau\Phase1_OfflineTraining\image_dataset\asl_alphabet_train"
OUTPUT_RAW_CSV = "data/dataset_raw.csv"
MODEL_TASK_PATH = "hand_landmarker.task"


def extract_features_per_folder():
    print("[*] Khởi tạo MediaPipe Tasks API...")
    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    raw_data_list = []

    # Danh sách các thư mục chữ cái (Class)
    classes = [d for d in os.listdir(IMAGE_DATASET_DIR) if os.path.isdir(os.path.join(IMAGE_DATASET_DIR, d))]
    total_classes = len(classes)

    global_success = 0
    global_total = 0

    print(f"[*] Tìm thấy {total_classes} thư mục (lớp) dữ liệu. Bắt đầu xử lý...\n")

    # VÒNG LẶP CÁC FOLDER (A, B, C...)
    for class_name in classes:
        class_dir = os.path.join(IMAGE_DATASET_DIR, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        folder_total = len(image_files)
        folder_success = 0

        print(f"--- Đang xử lý FOLDER: {class_name} ({folder_total} ảnh) ---")

        # VÒNG LẶP CÁC ẢNH TRONG FOLDER
        for idx, img_name in enumerate(image_files, 1):
            img_path = os.path.join(class_dir, img_name)

            try:
                mp_image = mp.Image.create_from_file(img_path)
                result = detector.detect(mp_image)

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]

                    # --- ĐÃ SỬA: Thu thập 63 tọa độ theo đúng cụm (X, Y, Z) của từng điểm ---
                    row = []
                    for lm in hand_landmarks:
                        row.extend([lm.x, lm.y, lm.z])

                    row.append(class_name)  # Gắn nhãn
                    raw_data_list.append(row)

                    folder_success += 1
                    global_success += 1

                # Báo cáo tiến độ trong folder (cứ mỗi 500 ảnh báo 1 lần)
                if idx % 500 == 0:
                    print(f"    [Tiến độ {class_name}]: Đã quét {idx}/{folder_total} ảnh...")

            except Exception as e:
                pass

            global_total += 1

        # KẾT LUẬN SAU MỖI FOLDER
        percentage = (folder_success / folder_total) * 100 if folder_total > 0 else 0
        print(f"[Xong] Folder {class_name}: Tìm thấy {folder_success}/{folder_total} mẫu ({percentage:.2f}%)")
        print("-" * 40)

    # LƯU CSV
    print(f"\n[*] Đang lưu dữ liệu ra file {OUTPUT_RAW_CSV}...")
    columns = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
    df = pd.DataFrame(raw_data_list, columns=columns)

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_RAW_CSV, index=False)

    print(f"\n[+] HOÀN TẤT TOÀN BỘ")
    print(
        f"    Tổng mẫu thực tế: {global_success}/{global_total} (Tỷ lệ: {(global_success / global_total) * 100:.2f}%)")


if __name__ == "__main__":
    extract_features_per_folder()