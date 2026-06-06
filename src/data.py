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
    # Giữ thứ tự thời gian dưới dạng LIST (không dùng set) để sau này có thể
    # tách validation từ cuối train đúng theo trục thời gian.
    ordered_sids = session_start.sort_values().index.tolist()
    split = int(len(ordered_sids) * train_ratio)
    train_ordered_all = ordered_sids[:split]   # sớm nhất -> muộn dần
    test_ordered_all = ordered_sids[split:]
    split_time = session_start.loc[ordered_sids[split]]
    if verbose:
        print(f"  Bước 1: Chia theo thời gian tại mốc {split_time}")
        print(f"          Train (sớm): {len(train_ordered_all):,} | Test (muộn): {len(test_ordered_all):,}")

    # --- Lọc item hiếm CHỈ trên train (tránh rò rỉ thông tin từ test) ---
    train_item_counts = Counter(
        i for sid in train_ordered_all for i in session_sequences[sid]
    )
    popular_items = {i for i, c in train_item_counts.items() if c >= min_item_freq}
    if verbose:
        print(f"  Bước 2: Giữ {len(popular_items):,} item (xuất hiện >= {min_item_freq} lần TRÊN TRAIN)")

    def _clean(seq):
        return [i for i in seq if i in popular_items]

    # Loop theo list đã sắp thời gian: dict giữ đúng thứ tự, và ta lưu thêm
    # danh sách sid (sau lọc) để tách val về sau không phụ thuộc thứ tự dict.
    train_sessions = {}
    train_ordered_sids = []
    for sid in train_ordered_all:
        s = _clean(session_sequences[sid])
        if min_len <= len(s) <= max_len:
            train_sessions[sid] = s
            train_ordered_sids.append(sid)

    test_sessions = {}
    test_ordered_sids = []
    for sid in test_ordered_all:
        s = _clean(session_sequences[sid])
        if min_len <= len(s) <= max_len:
            test_sessions[sid] = s
            test_ordered_sids.append(sid)

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
        # Thứ tự thời gian (sau lọc) để tách validation từ cuối train.
        "train_ordered_sids": train_ordered_sids,
        "test_ordered_sids": test_ordered_sids,
    }
    return train_sessions, test_sessions, info


def split_train_val_by_time(train_sessions: dict,
                            train_ordered_sids: list,
                            val_ratio: float = cfg.VAL_RATIO,
                            min_val_sessions: int = 1,
                            verbose: bool = True):
    """Tách validation từ CUỐI tập train (theo thời gian) → ``(train_inner, val)``.

    Lấy ``val_ratio`` phần phiên muộn nhất trong train làm validation, phần còn
    lại (sớm hơn) làm train_inner. KHÔNG lọc lại item hiếm để không ảnh hưởng
    cách chia/lọc đã cố định ở ``preprocess`` (test giữ nguyên).

    ``train_ordered_sids``: danh sách sid của train đã sắp theo thời gian
    (lấy từ ``info['train_ordered_sids']``).
    """
    # Phòng trường hợp truyền vào sid không còn trong train_sessions.
    ordered = [sid for sid in train_ordered_sids if sid in train_sessions]
    n_total = len(ordered)
    n_val = max(min_val_sessions, int(round(n_total * val_ratio)))
    if n_total <= n_val:
        raise ValueError(
            f"Không đủ phiên train để tách val: n_total={n_total}, n_val={n_val}"
        )

    train_inner_ids = ordered[:-n_val]   # sớm hơn
    val_ids = ordered[-n_val:]           # muộn nhất

    train_inner = {sid: train_sessions[sid] for sid in train_inner_ids}
    val_sessions = {sid: train_sessions[sid] for sid in val_ids}
    if verbose:
        print(f"  Tách val theo thời gian: train_inner={len(train_inner):,} | "
              f"val={len(val_sessions):,} (val_ratio={val_ratio:.4f})")
    return train_inner, val_sessions


def save_processed(train_sessions, test_sessions, path=cfg.PROCESSED_PATH, verbose: bool = True,
                   train_inner_sessions=None, val_sessions=None, info=None):
    """Lưu các tập đã xử lý ra pickle để tái dùng.

    ``train_sessions`` là train-full (cho Popularity/SKNN). Nếu có tách val cho
    GRU thì truyền thêm ``train_inner_sessions`` và ``val_sessions``.
    """
    payload = {
        "train_sessions": train_sessions,            # train-full (Pop/SKNN)
        "test_sessions": test_sessions,              # bất biến
        "train_inner_sessions": train_inner_sessions,  # GRU train (có thể None)
        "val_sessions": val_sessions,                # GRU validation (có thể None)
        "info": info or {},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    if verbose:
        print(f"  OK đã lưu {path}")


def load_processed(path=cfg.PROCESSED_PATH, include_val: bool = False, include_info: bool = False):
    """Đọc lại các tập đã xử lý.

    Mặc định trả ``(train_full, test)`` (tương thích ngược với caller cũ).
    - ``include_val=True``: trả thêm train_inner, val →
      ``(train_inner, val, train_full, test)``.
    - ``include_info=True``: trả kèm dict ``info`` ở cuối tuple.
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    train_full = d["train_sessions"]
    test = d["test_sessions"]
    info = d.get("info", {})

    if include_val:
        train_inner = d.get("train_inner_sessions")
        val = d.get("val_sessions")
        if train_inner is None or val is None:
            raise ValueError(
                "Pickle không có train_inner/val. Chạy lại tiền xử lý (run_all.py) "
                "để sinh validation split."
            )
        result = (train_inner, val, train_full, test)
    else:
        result = (train_full, test)

    if include_info:
        return (*result, info)
    return result
