"""Vẽ lại biểu đồ từ số THẬT trong output/all_results.pkl.

File này KHÔNG hardcode số và KHÔNG huấn luyện lại. Chạy `python run_all.py`
trước để sinh output/all_results.pkl, rồi chạy `python plot_final.py` nếu
muốn vẽ lại (run_all.py cũng đã tự vẽ sẵn các biểu đồ này).
"""
import os
import pickle

from src import config
from src.plots import plot_all

PKL = config.RESULTS_PATH
assert os.path.exists(PKL), (
    "Chưa có output/all_results.pkl. Hãy chạy `python run_all.py` trước."
)

with open(PKL, "rb") as f:
    results = pickle.load(f)

plot_all(results, config.OUTPUT_DIR)
print("\nTất cả biểu đồ đã lưu trong output/ (từ số thật trong all_results.pkl)")
