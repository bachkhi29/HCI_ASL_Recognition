import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
NORMALIZED_CSV = "data/dataset_normalized.csv"
MODEL_DIR = "exported_models/rf_hierarchical"
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
    print("[*] Đang vẽ Ma trận Nhầm lẫn (Confusion Matrix)...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('BÁO CÁO: Ma Trận Nhầm Lẫn - Hệ thống Random Forest Phân Cấp', fontsize=16, fontweight='bold')
    plt.xlabel('Nhãn Dự Đoán (Predicted Label)', fontsize=12)
    plt.ylabel('Nhãn Thực Tế (True Label)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/3_RF_Global_Confusion_Matrix.png", dpi=300)

def plot_feature_importance(rf_model, feature_names):
    print("[*] Đang vẽ Biểu đồ Tầm quan trọng Đặc trưng (Feature Importance)...")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-20:]

    plt.figure(figsize=(12, 8))
    plt.title('BÁO CÁO: Top 20 Tọa độ Không gian Quyết định (Super-Model)', fontsize=14, fontweight='bold')
    plt.barh(range(len(indices)), importances[indices], color='mediumseagreen', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Mức độ Quan trọng (Relative Importance)')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/4_RF_Feature_Importance.png", dpi=300)

def train_rf_hierarchical():
    print(f"[*] Đọc dữ liệu đã chuẩn hóa từ: {NORMALIZED_CSV}...")
    df = pd.read_csv(NORMALIZED_CSV)

    X = df.drop('label', axis=1)
    y_true_label = df['label']
    y_super_label = y_true_label.map(GROUP_MAPPING)

    print("[*] Đang chia tập dữ liệu (80% Train, 20% Test)...")
    X_train, X_test, y_train_super, y_test_super, y_train_true, y_test_true = train_test_split(
        X, y_super_label, y_true_label, test_size=0.2, random_state=42, stratify=y_true_label
    )

    # =========================================================
    # BƯỚC 1: HUẤN LUYỆN MÔ HÌNH CHA (SUPER-MODEL RF)
    # =========================================================
    print("\n" + "=" * 50)
    print("1. HUẤN LUYỆN MÔ HÌNH CHA (SUPER-MODEL RF)")
    print("=" * 50)

    # ĐÃ THÊM: class_weight='balanced'
    super_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    super_rf.fit(X_train, y_train_super)
    joblib.dump(super_rf, f"{MODEL_DIR}/super_model_rf.pkl")

    # In báo cáo và vẽ Feature Importance cho Super-Model
    y_pred_super = super_rf.predict(X_test)
    print(f"[+] Độ chính xác Mô hình Cha (Phân Cụm): {accuracy_score(y_test_super, y_pred_super) * 100:.2f}%\n")
    plot_feature_importance(super_rf, X.columns)

    # =========================================================
    # BƯỚC 2: HUẤN LUYỆN CÁC MÔ HÌNH CON (SUB-MODELS RF)
    # =========================================================
    print("\n" + "=" * 50)
    print("2. HUẤN LUYỆN CÁC MÔ HÌNH CON (SUB-MODELS RF)")
    print("=" * 50)

    sub_models_dict = {}
    unique_groups = sorted(list(set(GROUP_MAPPING.values())))

    for group_name in unique_groups:
        # Lọc dữ liệu riêng cho từng cụm
        mask_train = (y_train_super == group_name)
        X_train_sub, y_train_sub = X_train[mask_train], y_train_true[mask_train]

        # ĐÃ THÊM: class_weight='balanced'
        sub_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        sub_rf.fit(X_train_sub, y_train_sub)
        sub_models_dict[group_name] = sub_rf

        joblib.dump(sub_rf, f"{MODEL_DIR}/sub_model_rf_{group_name}.pkl")
        print(f" -> Đã huấn luyện xong nhóm: {group_name}")

    # =========================================================
    # BƯỚC 3: ĐÁNH GIÁ TỔNG THỂ
    # =========================================================
    print("\n" + "=" * 50)
    print("3. BÀI THI TỔNG THỂ KẾT HỢP (SUPER -> SUB)")
    print("=" * 50)

    # Máy Cha đoán Cụm cho toàn bộ tập Test
    pred_groups_all = super_rf.predict(X_test)

    # Mảng rỗng chứa kết quả chữ cái cuối cùng
    final_predictions = np.empty_like(y_test_true.values)

    # Duyệt qua từng Cụm để gọi Máy Con tương ứng giải quyết
    for group_name in unique_groups:
        mask = (pred_groups_all == group_name)
        if np.any(mask):
            X_test_masked = X_test[mask]
            final_predictions[mask] = sub_models_dict[group_name].predict(X_test_masked)

    final_acc = accuracy_score(y_test_true, final_predictions) * 100
    print(f"\n[+] ĐỘ CHÍNH XÁC CUỐI CÙNG HỆ THỐNG RF: {final_acc:.2f}%\n")

    print("Báo cáo chi tiết Tổng thể 28 Class:")
    print(classification_report(y_test_true, final_predictions))

    # Vẽ Confusion Matrix Tổng
    unique_labels = sorted(list(y_true_label.unique()))
    plot_confusion_matrix(y_test_true, final_predictions, unique_labels)

    print(f"\n[+] HOÀN TẤT! Đã đóng gói toàn bộ mô hình Random Forest vào thư mục: {MODEL_DIR}/")

if __name__ == "__main__":
    train_rf_hierarchical()