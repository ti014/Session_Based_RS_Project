"""SKNN — Session K-Nearest Neighbors với inverted index.

Thay vì duyệt TẤT CẢ phiên lịch sử, ta chỉ tính độ tương tự với những
phiên chia sẻ ít nhất một item với phiên truy vấn (nhờ inverted index
item -> các phiên chứa item). Độ phức tạp giảm từ O(N*|Sq|) xuống
O(|Sq|*F_tb + C*logC), với C là số phiên ứng viên.
"""
import math
from collections import defaultdict


class SKNN:
    def __init__(self, k: int = 500):
        # k = số phiên hàng xóm gần nhất cần giữ.
        self.k = k
        self.train_sets = {}                 # sid -> set(items)
        self.train_items = {}                # sid -> list(items)
        self.item2sids = defaultdict(list)   # item -> [sid, ...]

    def fit(self, sessions: dict):
        # Lưu phiên lịch sử + dựng inverted index (không cần huấn luyện).
        self.train_items = sessions
        self.train_sets = {sid: set(items) for sid, items in sessions.items()}
        self.item2sids = defaultdict(list)
        for sid, items in self.train_sets.items():
            for i in items:
                self.item2sids[i].append(sid)

    def predict(self, query_session, top_n: int = 20):
        query_set = set(query_session)

        # Bước 1: tập ứng viên (chỉ phiên chia sẻ >= 1 item với truy vấn).
        cand = set()
        for i in query_set:
            cand.update(self.item2sids.get(i, ()))

        # Bước 2: tính similarity = |giao| / sqrt(|Sq|*|Sn|), lấy K hàng xóm.
        q_len = len(query_set)
        sims = []
        for sid in cand:
            sn_set = self.train_sets[sid]
            inter = len(query_set & sn_set)
            if inter:
                sims.append((sid, inter / math.sqrt(q_len * len(sn_set))))

        sims.sort(key=lambda x: x[1], reverse=True)
        knn = sims[: self.k]

        # Bước 3: cộng dồn điểm cho từng item ứng viên (chưa nằm trong truy vấn).
        scores = defaultdict(float)
        for sid, sim in knn:
            for item in self.train_sets[sid]:
                if item not in query_set:
                    scores[item] += sim

        # Bước 4: sắp xếp giảm dần theo điểm, trả về Top-N.
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
