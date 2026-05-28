"""Siêu tham số và đường dẫn dùng chung cho toàn bộ pipeline.

Đặt mọi hằng số ở một nơi để báo cáo/slide/notebook và run_all.py luôn
dùng cùng một cấu hình, tránh lệch nhau. Chế độ "smoke" (chạy nhanh kiểm
tra logic) được khai báo riêng để không lẫn với cấu hình thật.
"""
from pathlib import Path

# --- Thư mục & tệp ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CLICKS_PATH = DATA_DIR / "yoochoose-clicks.dat"
PROCESSED_PATH = DATA_DIR / "processed_data.pkl"
RESULTS_PATH = OUTPUT_DIR / "all_results.pkl"
GRU_MODEL_PATH = OUTPUT_DIR / "gru4rec_model.pt"

# --- Tiền xử lý ---
SEED = 42
SAMPLE_FRACTION = 64          # lấy 1/64 dữ liệu theo phiên
MIN_ITEM_FREQ = 5             # bỏ item xuất hiện ít hơn ngưỡng (chỉ tính trên train)
MIN_LEN, MAX_LEN = 2, 20      # giữ phiên có độ dài trong khoảng này
TRAIN_RATIO = 0.9            # 90% phiên sớm nhất làm train, 10% muộn nhất làm test

# --- Đánh giá ---
TOP_N = 20                    # Recall@20 / MRR@20
MAX_EVAL = None               # None = đánh giá toàn bộ tập test

# --- SKNN ---
SKNN_K = 500                  # số phiên hàng xóm gần nhất
K_VALUES = [50, 100, 200, 500]  # dải K cho thí nghiệm ảnh hưởng của K

# --- GRU4Rec ---
N_EPOCHS = 10
EMB_SIZE = 64
HIDDEN_SIZE = 128
N_LAYERS = 1
DROPOUT = 0.25
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0


# Cấu hình chế độ smoke: ghi đè vài tham số để chạy trong vài giây.
SMOKE_OVERRIDES = {
    "SAMPLE_FRACTION": 512,
    "N_EPOCHS": 1,
    "MAX_EVAL": 300,
    "K_VALUES": [50, 500],
}


def resolve(smoke: bool = False) -> dict:
    """Trả về dict cấu hình hiệu lực.

    Lấy mọi hằng số viết HOA trong module này; nếu ``smoke=True`` thì áp
    các giá trị trong ``SMOKE_OVERRIDES`` đè lên. Orchestrator dùng dict
    này rồi truyền tường minh xuống các hàm, nên hàm không phụ thuộc biến
    toàn cục (dễ kiểm thử, dễ tái lập).
    """
    g = globals()
    cfg = {k: v for k, v in g.items() if k.isupper()}
    if smoke:
        cfg.update(SMOKE_OVERRIDES)
    return cfg
