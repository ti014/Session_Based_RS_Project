"""Gói lõi cho hệ khuyến nghị dựa trên phiên (Session-based RS).

Các module:
  - config    : siêu tham số + đường dẫn dùng chung
  - data      : đọc + tiền xử lý + chia train/test theo thời gian
  - models    : Popularity, SKNN, GRU4Rec
  - evaluate  : Recall@N / MRR@N (giao thức loại item đã xem)
  - plots     : vẽ biểu đồ từ kết quả

Import gói này cũng tự cấu hình stdout sang UTF-8 để in tiếng Việt có dấu
trên console Windows (mặc định cp1252 sẽ lỗi với ký tự đa byte).
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

__all__ = ["config", "data", "models", "evaluate", "plots"]
