"""Popularity Baseline: luôn gợi ý các item phổ biến nhất.

Mô hình tham chiếu đơn giản nhất — không dùng thông tin chuỗi. Mọi mô
hình có ý nghĩa đều phải vượt qua baseline này.
"""
from collections import Counter


class PopularityBaseline:
    def __init__(self):
        self.popular_items = []

    def fit(self, sessions: dict):
        # Đếm tần suất xuất hiện của item trên toàn tập train.
        counter = Counter(item for items in sessions.values() for item in items)
        self.popular_items = [item for item, _ in counter.most_common()]

    def predict(self, query_session, top_n: int = 20):
        # Trả về Top-N item phổ biến nhất chưa xuất hiện trong phiên hiện tại.
        query_set = set(query_session)
        return [(item, 0.0) for item in self.popular_items if item not in query_set][:top_n]
