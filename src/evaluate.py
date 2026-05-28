"""Đánh giá theo giao thức next-item (leave-one-out).

Với mỗi phiên test ``[i_1, ..., i_t]``: lấy ``[i_1, ..., i_{t-1}]`` làm
đầu vào, ``i_t`` làm đáp án, rồi kiểm tra đáp án có nằm trong Top-N gợi ý
hay không.
  - Recall@N: tỉ lệ phiên có đáp án nằm trong Top-N.
  - MRR@N  : trung bình nghịch đảo thứ hạng của đáp án.

Hàm này dùng cho các mô hình có giao diện ``predict(seq, top_n)`` trả về
danh sách ``(item, score)`` (Popularity, SKNN). GRU4Rec có hàm đánh giá
riêng trong ``src.models.gru4rec`` vì bước suy luận khác.
"""
import time


def evaluate(model, test_sessions, top_n: int = 20, max_eval=None, tag: str = "", verbose: bool = True):
    """Trả về ``(recall, mrr, n_eval)``."""
    hits, mrr_sum, n_eval = 0, 0.0, 0
    items_list = list(test_sessions.items())
    if max_eval:
        items_list = items_list[:max_eval]
    total = len(items_list)
    start = time.time()
    for idx, (sid, items) in enumerate(items_list):
        if len(items) < 2:
            continue
        input_seq, label = items[:-1], items[-1]
        recs = model.predict(input_seq, top_n=top_n)
        rec_items = [item for item, _ in recs]
        if label in rec_items:
            hits += 1
            mrr_sum += 1.0 / (rec_items.index(label) + 1)
        n_eval += 1
        if verbose and (idx + 1) % 1000 == 0:
            print(f"    {tag}[{idx+1}/{total}] {time.time()-start:.0f}s...", end="\r")
    recall = hits / n_eval if n_eval > 0 else 0
    mrr = mrr_sum / n_eval if n_eval > 0 else 0
    return recall, mrr, n_eval
