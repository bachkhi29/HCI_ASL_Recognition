import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
NORMALIZED_CSV = "data/dataset_normalized.csv"
MODEL_DIR = "exported_models/knn_hierarchical"
PLOT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

GROUP_MAPPING = {
    'Y': 'G1', 'J': 'G1',
    'C': 'G2', 'O': 'G2',
    'G': 'G3', 'H': 'G3',
    'B': 'G4', 'D': 'G4', 'F': 'G4', 'I': 'G4', 'U': 'G4', 'V': 'G4', 'K': 'G4', 'R': 'G4', 'W': 'G4',
    'P': 'G5', 'Q': 'G5', 'Z': 'G5',
    'A': 'G6', 'E': 'G6', 'M': 'G6', 'N': 'G6', 'S': 'G6', 'T': 'G6',
    'L': 'G7', 'X': 'G7', 'Space': 'G7', 'Del': 'G7'
}


def plot_confusion_matrix(y_true, y_pred, labels):
    print("[*] Đang vẽ Ma trận Nhầm lẫn (Confusion Matrix) cho KNN...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(16, 12))
    # Dùng màu Xanh Lá (Greens) để phân biệt với Xanh Dương (RF) và Cam (SVM)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title('BÁO CÁO: Ma Trận Nhầm Lẫn - Hệ thống KNN Phân Cấp', fontsize=16, fontweight='bold')
    plt.xlabel('Nhãn Dự Đoán (Predicted)', fontsize=12)
    plt.ylabel('Nhãn Thực Tế (True)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/6_KNN_Global_Confusion_Matrix.png", dpi=300)


def train_knn_hierarchical():
    print(f"[*] Đọc dữ liệu sạch từ: {NORMALIZED_CSV}...")
    df = pd.read_csv(NORMALIZED_CSV)

    X = df.drop('label', axis=1)
    y_true_label = df['label']
    y_super_label = y_true_label.map(GROUP_MAPPING)

    print("[*] Đang chia tập dữ liệu (80% Train, 20% Test)...")
    X_train, X_test, y_train_super, y_test_super, y_train_true, y_test_true = train_test_split(
        X, y_super_label, y_true_label, test_size=0.2, random_state=42, stratify=y_true_label
    )

    # =========================================================
    # BƯỚC 0: CHUẨN HÓA PHÂN BỐ (STANDARD SCALER) - SỐNG CÒN CHO KNN
    # =========================================================
    print("[*] Đang tinh chỉnh phân bố tỷ lệ (Standard Scaler)...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, f"{MODEL_DIR}/knn_scaler.pkl")

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    # =========================================================
    # BƯỚC 1: HUẤN LUYỆN MÔ HÌNH CHA (SUPER-KNN)
    # =========================================================
    print("\n" + "=" * 50)
    print("1. HUẤN LUYỆN MÔ HÌNH CHA (SUPER-KNN)")
    print("=" * 50)

    # Dùng n_neighbors=5 (mặc định tối ưu) và weights='distance'
    super_knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    super_knn.fit(X_train_scaled, y_train_super)

    y_pred_super = super_knn.predict(X_test_scaled)
    super_acc = accuracy_score(y_test_super, y_pred_super) * 100

    print(f"[+] Độ chính xác Mô hình Cha (Super-KNN): {super_acc:.2f}%\n")
    joblib.dump(super_knn, f"{MODEL_DIR}/super_model_knn.pkl")

    # =========================================================
    # BƯỚC 2: HUẤN LUYỆN CÁC MÔ HÌNH CON (SUB-KNN)
    # =========================================================
    print("\n" + "=" * 50)
    print("2. HUẤN LUYỆN CÁC MÔ HÌNH CON (SUB-KNN)")
    print("=" * 50)

    sub_models_dict = {}
    unique_groups = sorted(list(set(GROUP_MAPPING.values())))

    for group_name in unique_groups:
        mask_train = (y_train_super == group_name)
        X_train_sub = X_train_scaled[mask_train]
        y_train_sub = y_train_true[mask_train]

        sub_knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        sub_knn.fit(X_train_sub, y_train_sub)
        sub_models_dict[group_name] = sub_knn

        joblib.dump(sub_knn, f"{MODEL_DIR}/sub_model_knn_{group_name}.pkl")
        print(f" -> Đã huấn luyện xong Cụm: {group_name}")

    # =========================================================
    # BƯỚC 3: ĐÁNH GIÁ TỔNG THỂ CẢ HỆ THỐNG
    # =========================================================
    print("\n" + "=" * 50)
    print("3. BÀI THI TỔNG THỂ KẾT HỢP (SUPER -> SUB)")
    print("=" * 50)

    pred_groups_all = super_knn.predict(X_test_scaled)
    final_predictions = np.empty_like(y_test_true.values)

    for group_name in unique_groups:
        mask = (pred_groups_all == group_name)
        if np.any(mask):
            X_test_masked = X_test_scaled[mask]
            final_predictions[mask] = sub_models_dict[group_name].predict(X_test_masked)

    final_acc = accuracy_score(y_test_true, final_predictions) * 100

    print(f"\n[+] ĐỘ CHÍNH XÁC CUỐI CÙNG HỆ THỐNG KNN: {final_acc:.2f}%\n")
    print("Báo cáo chi tiết Tổng thể 28 Class:")
    print(classification_report(y_test_true, final_predictions))

    unique_labels = sorted(list(y_true_label.unique()))
    plot_confusion_matrix(y_test_true, final_predictions, unique_labels)

    print(f"\n[+] HOÀN TẤT! Đã xuất xưởng toàn bộ gia tài KNN vào thư mục: {MODEL_DIR}/")


if __name__ == "__main__":
    train_knn_hierarchical()