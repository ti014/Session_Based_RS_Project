"""Đọc và tiền xử lý dữ liệu Yoochoose.

Quy trình (giữ đúng thứ tự để chống rò rỉ thông tin):
  1. Lấy mẫu 1/N theo phiên (cố định seed)
  2. Sắp xếp theo (SessionID, Timestamp) để chuỗi item đúng thứ tự thời gian
  3. Chia train/test THEO THỜI GIAN: 90% phiên sớm nhất làm train,
     10% muộn nhất làm test (mô phỏng "dùng quá khứ dự đoán tương lai")
  4. Lọc item hiếm CHỈ dựa trên tập train (tránh rò rỉ từ test)
  5. Lọc phiên theo độ dài [MIN_LEN, MAX_LEN]
"""
import time
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from . import config as cfg


def load_clicks(path=cfg.CLICKS_PATH, verbose: bool = True) -> pd.DataFrame:
    """Đọc file click thô của Yoochoose (không có header)."""
    if verbose:
        print(f"\nĐang đọc {path}...")
    start = time.time()
    df = pd.read_csv(
        path,
        header=None,
        names=["SessionID", "Timestamp", "ItemID", "Category"],
        dtype={"SessionID": int, "ItemID": int, "Category": str},
        parse_dates=["Timestamp"],
    )
    if verbose:
        print(f"  OK đọc xong trong {time.time()-start:.1f}s")
        print(f"  Tổng click: {len(df):,} | Phiên: {df['SessionID'].nunique():,} "
              f"| Item: {df['ItemID'].nunique():,}")
    return df


def preprocess(df: pd.DataFrame,
               sample_fraction: int = cfg.SAMPLE_FRACTION,
               min_item_freq: int = cfg.MIN_ITEM_FREQ,
               min_len: int = cfg.MIN_LEN,
               max_len: int = cfg.MAX_LEN,
               train_ratio: float = cfg.TRAIN_RATIO,
               seed: int = cfg.SEED,
               verbose: bool = True):
    """Tiền xử lý → trả về ``(train_sessions, test_sessions, info)``.

    ``train_sessions`` / ``test_sessions``: dict ``sid -> [item, ...]``.
    ``info``: dict thống kê (mốc thời gian chia, số item giữ lại, ...).
    """
    # --- Lấy mẫu 1/N theo phiên (giữ đúng idiom seed gốc để tái lập) ---
    if verbose:
        print(f"\nLấy mẫu 1/{sample_fraction} dữ liệu...")
    np.random.seed(seed)
    all_sids = df["SessionID"].unique()
    sample_sids = np.random.choice(all_sids, size=len(all_sids) // sample_fraction, replace=False)
    df = df[df["SessionID"].isin(sample_sids)].copy()
    if verbose:
        print(f"  Sau lấy mẫu: {df['SessionID'].nunique():,} phiên, {len(df):,} click")

    # --- Sắp xếp để chuỗi item đúng thứ tự thời gian ---
    df = df.sort_values(["SessionID", "Timestamp"])

    # --- Nhóm theo phiên + thời điểm bắt đầu mỗi phiên ---
    session_sequences = df.groupby("SessionID")["ItemID"].apply(list).to_dict()
    session_start = df.groupby("SessionID")["Timestamp"].min()
    if verbose:
        print(f"\n  Bước 0: {len(session_sequences):,} phiên")

    # --- Chia train/test THEO THỜI GIAN ---
    ordered_sids = session_start.sort_values().index.tolist()
    split = int(len(ordered_sids) * train_ratio)
    train_ids_all = set(ordered_sids[:split])
    test_ids_all = set(ordered_sids[split:])
    split_time = session_start.loc[ordered_sids[split]]
    if verbose:
        print(f"  Bước 1: Chia theo thời gian tại mốc {split_time}")
        print(f"          Train (sớm): {len(train_ids_all):,} | Test (muộn): {len(test_ids_all):,}")

    # --- Lọc item hiếm CHỈ trên train (tránh rò rỉ thông tin từ test) ---
    train_item_counts = Counter(
        i for sid in train_ids_all for i in session_sequences[sid]
    )
    popular_items = {i for i, c in train_item_counts.items() if c >= min_item_freq}
    if verbose:
        print(f"  Bước 2: Giữ {len(popular_items):,} item (xuất hiện >= {min_item_freq} lần TRÊN TRAIN)")

    def _clean(seq):
        return [i for i in seq if i in popular_items]

    train_sessions = {}
    for sid in train_ids_all:
        s = _clean(session_sequences[sid])
        if min_len <= len(s) <= max_len:
            train_sessions[sid] = s

    test_sessions = {}
    for sid in test_ids_all:
        s = _clean(session_sequences[sid])
        if min_len <= len(s) <= max_len:
            test_sessions[sid] = s

    if verbose:
        print(f"  Bước 3: Lọc độ dài [{min_len},{max_len}] + bỏ item hiếm")
        print(f"          Train: {len(train_sessions):,} phiên | Test: {len(test_sessions):,} phiên")
        lengths = [len(v) for v in train_sessions.values()]
        print(f"\n  Độ dài phiên train: trung bình={np.mean(lengths):.1f}, "
              f"trung vị={np.median(lengths):.0f}")

    info = {
        "n_train": len(train_sessions),
        "n_test": len(test_sessions),
        "split_time": str(split_time),
        "n_popular_items": len(popular_items),
        "split": "time-based (90% sớm / 10% muộn theo thời điểm bắt đầu phiên)",
    }
    return train_sessions, test_sessions, info


def save_processed(train_sessions, test_sessions, path=cfg.PROCESSED_PATH, verbose: bool = True):
    """Lưu train/test đã xử lý ra pickle để tái dùng."""
    with open(path, "wb") as f:
        pickle.dump({"train_sessions": train_sessions, "test_sessions": test_sessions}, f)
    if verbose:
        print(f"  OK đã lưu {path}")


def load_processed(path=cfg.PROCESSED_PATH):
    """Đọc lại train/test đã xử lý."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["train_sessions"], d["test_sessions"]
