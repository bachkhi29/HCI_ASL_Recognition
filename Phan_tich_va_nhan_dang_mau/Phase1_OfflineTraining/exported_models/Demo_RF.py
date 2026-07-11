# ==============================================================================
# PHASE 3: HỆ ĐIỀU HÀNH THỦ NGỮ (PHIÊN BẢN HIERARCHICAL RANDOM FOREST)
# Mô tả: Tay TRÁI vật lý gõ chữ, Tay PHẢI vật lý bấm Enter. Tích hợp AI Phân cấp.
# ==============================================================================

import cv2
import time
import joblib
import warnings
import math
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# --- CÁCH IMPORT CHUẨN MỰC ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS
drawing_styles = mp.solutions.drawing_styles

# ==============================================================================
# HÀM CHUẨN HÓA TOÁN HỌC (BẮT BUỘC TRƯỚC KHI ĐƯA CHO AI)
# ==============================================================================
def normalize_exact_math(raw_coords):
    points = np.array(raw_coords).reshape(21, 3)
    wrist = points[0]
    translated = points - wrist  # Dời tâm về cổ tay
    distances = np.linalg.norm(translated, axis=1)
    max_dist = np.max(distances)
    if max_dist > 0:
        scaled = translated / max_dist  # Thu phóng
    else:
        scaled = translated
    return scaled.flatten().tolist()

# ==============================================================================
# HÀM KIỂM TRA "NÚT BẤM" (TAY PHẢI VẬT LÝ)
# ==============================================================================
def is_open_palm(landmarks):
    def get_dist(p1, p2):
        return math.hypot(landmarks[p1].x - landmarks[p2].x, landmarks[p1].y - landmarks[p2].y)

    index_open = get_dist(0, 8) > get_dist(0, 5) * 1.5
    middle_open = get_dist(0, 12) > get_dist(0, 9) * 1.5
    ring_open = get_dist(0, 16) > get_dist(0, 13) * 1.5
    pinky_open = get_dist(0, 20) > get_dist(0, 17) * 1.5
    fingers_straight = index_open and middle_open and ring_open and pinky_open

    thumb_spread = get_dist(4, 17) > get_dist(0, 9)
    is_pointing_up = landmarks[9].y < landmarks[0].y
    delta_y = abs(landmarks[0].y - landmarks[9].y)
    delta_x = abs(landmarks[0].x - landmarks[9].x)
    is_vertical = delta_y > delta_x

    return fingers_straight and thumb_spread and is_pointing_up and is_vertical

# ==========================================
# 1. NẠP HỆ THỐNG MÔ HÌNH PHÂN CẤP (V2)
# ==========================================
print("[*] Đang khởi động OS Controller (Kiến trúc Phân cấp)...")
MODEL_DIR = "exported_models/rf_hierarchical"

try:
    print(" -> Nạp Mô hình Cha (Super-Model)...")
    super_rf = joblib.load(f"{MODEL_DIR}/super_model_rf.pkl")

    print(" -> Nạp 7 Mô hình Con (Sub-Models)...")
    sub_models = {}
    for i in range(1, 8):
        group_id = f"G{i}"
        sub_models[group_id] = joblib.load(f"{MODEL_DIR}/sub_model_rf_{group_id}.pkl")
except Exception as e:
    print(f"[!] LỖI TẢI MÔ HÌNH: {e}")
    print("[!] Hãy đảm bảo bạn đã chạy xong file 3a_train_hierarchical_rf.py")
    exit()

# Cấu hình MediaPipe
MODEL_PATH = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

final_text = ""
current_letter = ""
can_confirm_next = True
prev_frame_time = 0

print("\n[*] HỆ THỐNG ĐÃ SẴN SÀNG!")
print(" ---> Tay TRÁI của bạn: Ra ký hiệu AI.")
print(" ---> Tay PHẢI của bạn: Xòe thẳng lên để Enter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: continue

    frame = cv2.flip(frame, 1)  # Lật gương

    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 30
    prev_frame_time = current_time
    timestamp_ms = int(current_time * 1000)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    try:
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        signing_hand_landmarks = None  # Tay TRÁI vật lý (dùng để ra ký hiệu)
        trigger_hand_landmarks = None  # Tay PHẢI vật lý (dùng làm nút Enter)

        if result.hand_landmarks:
            for idx, handedness in enumerate(result.handedness):
                hand_label = handedness[0].category_name
                landmarks = result.hand_landmarks[idx]

                if hand_label == 'Right':
                    signing_hand_landmarks = landmarks
                elif hand_label == 'Left':
                    trigger_hand_landmarks = landmarks

                from mediapipe.framework.formats import landmark_pb2

                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
                ])
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks_proto, mp_hands_connections,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style()
                )

        # -------------------------------------------------------------
        # LUỒNG 1: TAY TRÁI VẬT LÝ (AI DỰ ĐOÁN CHỮ THEO CƠ CHẾ PHÂN CẤP)
        # -------------------------------------------------------------
        if signing_hand_landmarks:
            row = []
            for lm in signing_hand_landmarks:
                row.extend([lm.x, lm.y, lm.z])

            # 1. BẮT BUỘC CHUẨN HÓA
            normalized_row = normalize_exact_math(row)

            # --- ĐÃ SỬA: Ép kiểu sang DataFrame để mô hình không bỡ ngỡ ---
            feature_cols = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
            df_input = pd.DataFrame([normalized_row], columns=feature_cols)

            # 2. AI CHA DỰ ĐOÁN CỤM
            predicted_group = super_rf.predict(df_input)[0]

            # 3. AI CON DỰ ĐOÁN CHỮ CÁI
            current_letter = sub_models[predicted_group].predict(df_input)[0]

            cv2.putText(frame, f"Tay trai (Go): {current_letter}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        else:
            current_letter = ""

        # -------------------------------------------------------------
        # LUỒNG 2: TAY PHẢI VẬT LÝ (NÚT XÁC NHẬN BẰNG TOÁN HỌC)
        # -------------------------------------------------------------
        if trigger_hand_landmarks:
            if is_open_palm(trigger_hand_landmarks):
                cv2.putText(frame, "[ ENTER ]", (450, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                if can_confirm_next and current_letter != "":
                    if current_letter == "Del":
                        final_text = final_text[:-1]
                    elif current_letter == "Space":
                        final_text += " "
                    else:
                        final_text += current_letter

                    can_confirm_next = False
            else:
                cv2.putText(frame, "[ CHO... ]", (450, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
                can_confirm_next = True
        else:
            can_confirm_next = True

        # Vẽ UI hiển thị FPS và Text
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.rectangle(frame, (0, 400), (720, 480), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"Van ban: {final_text}", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    except Exception as e:
        print(f"\r[!] Báo lỗi ngầm: {e}", end="")

    cv2.imshow("Sign Language OS Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n[*] Đã đóng Camera. Chương trình kết thúc.")