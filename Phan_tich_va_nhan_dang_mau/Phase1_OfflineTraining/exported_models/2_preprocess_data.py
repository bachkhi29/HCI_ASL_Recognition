import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.pipeline import Pipeline
from collections import Counter

RAW_CSV = "data/dataset_raw.csv"
NORMALIZED_CSV = "data/dataset_normalized.csv"
PLOT_DIR = "reports"

os.makedirs(PLOT_DIR, exist_ok=True)


def normalize_exact_math(raw_coords):
    points = np.array(raw_coords).reshape(21, 3)
    wrist = points[0]
    translated = points - wrist
    distances = np.linalg.norm(translated, axis=1)
    max_dist = np.max(distances)
    if max_dist > 0:
        scaled = translated / max_dist
    else:
        scaled = translated
    return scaled.flatten().tolist()


def plot_class_distribution(y_original, y_resampled):
    print("[*] Đang vẽ Biểu đồ Phân bố Dữ liệu...")
    count_orig = Counter(y_original)
    count_res = Counter(y_resampled)
    labels = sorted(list(count_orig.keys()))
    val_orig = [count_orig[l] for l in labels]
    val_res = [count_res[l] for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, val_orig, width, label='Trước cân bằng (Raw)', color='#FF9999')
    ax.bar(x + width / 2, val_res, width, label='Sau cân bằng & Khử nhiễu ENN', color='#66B2FF')

    ax.set_ylabel('Số lượng mẫu (Images)')
    ax.set_title('BÁO CÁO: Phân bố dữ liệu trước và sau khi khử nhiễu ENN', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/1_Class_Distribution.png", dpi=300)


def plot_tsne_visualization(X_features, y_labels):
    print("[*] Đang tính toán không gian t-SNE 2D...")
    # Đã sửa lại tham số max_iter để không bị lỗi trên các thư viện đời mới
    tsne = TSNE(n_components=2, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_features)

    tsne_df = pd.DataFrame({'TSNE_X': X_tsne[:, 0], 'TSNE_Y': X_tsne[:, 1], 'Label': y_labels})
    tsne_df = tsne_df.sort_values(by='Label')

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='TSNE_X', y='TSNE_Y', hue='Label', palette=sns.color_palette("husl", len(np.unique(y_labels))),
                    data=tsne_df, legend="full", alpha=0.7, s=15)
    plt.title('BÁO CÁO: Không gian phân tách của 63 tọa độ sau chuẩn hóa Toán học', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/2_TSNE_Visualization.png", dpi=300)


def process_and_balance_data_enn():
    print(f"[*] Đọc dữ liệu thô từ {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)

    if 'Nothing' in df['label'].values:
        df = df[df['label'] != 'Nothing']

    X_raw = df.drop('label', axis=1).values
    y_raw = df['label'].values

    # BƯỚC 1: BẮT BUỘC CHUẨN HÓA TRƯỚC KHI ĐO KHOẢNG CÁCH
    print("[*] Đang áp dụng Dời tâm và Thu phóng (Exact Math)...")
    X_normalized = np.array([normalize_exact_math(row) for row in X_raw])

    # BƯỚC 2: CÂN BẰNG & KHỬ NHIỄU (Pipeline)
    # Lọc thô ép về 1200 mẫu bằng RUS
    dict_rus = {label: min(count, 1200) for label, count in Counter(y_raw).items()}
    rus = RandomUnderSampler(sampling_strategy=dict_rus, random_state=42)

    # Gọt tinh (Chỉ xóa các mẫu nằm sai chỗ, giữ lại mẫu dễ) bằng ENN
    enn = EditedNearestNeighbours(n_neighbors=3, kind_sel='all', n_jobs=-1)
    pipeline = Pipeline(steps=[('rus', rus), ('enn', enn)])

    print("[*] Đang tiến hành gọt giũa và khử nhiễu biên giới bằng ENN...")
    X_resampled, y_resampled = pipeline.fit_resample(X_normalized, y_raw)

    # BƯỚC 3: XUẤT BÁO CÁO & LƯU FILE
    plot_class_distribution(y_raw, y_resampled)

    columns = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
    df_normalized = pd.DataFrame(X_resampled, columns=columns)
    df_normalized['label'] = y_resampled
    df_normalized.to_csv(NORMALIZED_CSV, index=False)
    print(f"\n[+] Đã lưu file dữ liệu siêu sạch tại: {NORMALIZED_CSV}")

    # Vẽ t-SNE trên tập dữ liệu đã làm sạch
    sample_size = min(len(X_resampled), 5000)  # Chỉ lấy 5000 mẫu để vẽ cho nhẹ máy
    idx = np.random.choice(len(X_resampled), sample_size, replace=False)
    plot_tsne_visualization(X_resampled[idx], y_resampled[idx])

    print("\n[+] HOÀN TẤT TOÀN BỘ QUÁ TRÌNH TIỀN XỬ LÝ!")


if __name__ == "__main__":
    process_and_balance_data_enn()