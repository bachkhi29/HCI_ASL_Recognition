import numpy as np


def normalize_landmarks(landmarks):
    # Bước 1: Chuyển đổi list dict từ Frontend sang Numpy Array dạng (21, 3)
    points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])

    # Bước 2: Dời tâm về gốc tọa độ cổ tay (điểm số 0)
    wrist = points[0]
    translated = points - wrist

    # Bước 3: Tính khoảng cách Euclidean từ tất cả các khớp đến cổ tay
    distances = np.linalg.norm(translated, axis=1)
    max_dist = np.max(distances)

    # Bước 4: Thu phóng (Scaling) theo khoảng cách lớn nhất chuẩn bài của ông
    if max_dist > 0:
        scaled = translated / max_dist
    else:
        scaled = translated

    # Duỗi phẳng ra thành list 63 phần tử để nạp vào SVM
    return scaled.flatten().tolist()