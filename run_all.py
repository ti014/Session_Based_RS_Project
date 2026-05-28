"""
============================================================
CHỦ ĐỀ 9: SESSION-BASED RECOMMENDATION SYSTEM
============================================================
Orchestrator (điều phối) — nguồn chân lý cho báo cáo/slide/notebook.
Toàn bộ logic nằm trong gói ``src/``; file này chỉ ghép các bước:

  1. Đọc dữ liệu Yoochoose                         (src.data)
  2. Tiền xử lý + chia train/test theo thời gian    (src.data)
  3. Popularity Baseline                            (src.models)
  4. SKNN (inverted index)                          (src.models)
  5. GRU4Rec                                        (src.models)
  6. Đánh giá Recall@20 / MRR@20 trên toàn bộ test  (src.evaluate)
  7. Lưu kết quả + lịch sử loss vào output/all_results.pkl
  8. Vẽ biểu đồ từ số THẬT                          (src.plots)

Chạy nhanh để kiểm tra logic:  python run_all.py --smoke
Chạy đầy đủ (lấy số thật):     python run_all.py
============================================================
"""
import pickle
import argparse

import torch

from src import config
from src.data import load_clicks, preprocess, save_processed
from src.models import PopularityBaseline, SKNN, build_item_index, train_gru, evaluate_gru
from src.models.gru4rec import get_device
from src.evaluate import evaluate
from src.plots import plot_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Chạy nhanh trên mẫu nhỏ để kiểm tra logic")
    args = parser.parse_args()
    smoke = args.smoke

    cfg = config.resolve(smoke=smoke)
    TOP_N = cfg["TOP_N"]
    MAX_EVAL = cfg["MAX_EVAL"]
    SKNN_K = cfg["SKNN_K"]
    N_EPOCHS = cfg["N_EPOCHS"]
    K_VALUES = cfg["K_VALUES"]

    print("=" * 60)
    print("  CHỦ ĐỀ 9: SESSION-BASED RECOMMENDATION SYSTEM")
    print("  Dataset: Yoochoose (RecSys Challenge 2015)")
    print(f"  Chế độ: {'SMOKE (kiểm tra nhanh)' if smoke else 'FULL (số thật)'}")
    print("=" * 60)

    # --- Phần 1: Đọc + tiền xử lý ---
    print("\n" + "-" * 60)
    print("  PHẦN 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU")
    print("-" * 60)
    df = load_clicks(cfg["CLICKS_PATH"])
    train_sessions, test_sessions, info = preprocess(
        df,
        sample_fraction=cfg["SAMPLE_FRACTION"],
        min_item_freq=cfg["MIN_ITEM_FREQ"],
        min_len=cfg["MIN_LEN"],
        max_len=cfg["MAX_LEN"],
        train_ratio=cfg["TRAIN_RATIO"],
        seed=cfg["SEED"],
    )
    save_processed(train_sessions, test_sessions, cfg["PROCESSED_PATH"])

    # --- Phần 2: Popularity Baseline ---
    print("\n" + "-" * 60)
    print("  PHẦN 2: POPULARITY BASELINE")
    print("-" * 60)
    model_pop = PopularityBaseline()
    model_pop.fit(train_sessions)
    print(f"  Top-5 item phổ biến: {model_pop.popular_items[:5]}")
    r_pop, mrr_pop, n_pop = evaluate(model_pop, test_sessions, top_n=TOP_N, max_eval=MAX_EVAL, tag="POP ")
    print(f"  OK xong ({n_pop} phiên)")
    print(f"  Recall@20 = {r_pop:.4f} ({r_pop*100:.2f}%) | MRR@20 = {mrr_pop:.4f}")

    # --- Phần 3: SKNN ---
    print("\n" + "-" * 60)
    print(f"  PHẦN 3: SKNN (k={SKNN_K}, inverted index)")
    print("-" * 60)
    model_sknn = SKNN(k=SKNN_K)
    model_sknn.fit(train_sessions)
    print(f"  Đã lưu {len(train_sessions):,} phiên lịch sử")
    r_sknn, mrr_sknn, n_sknn = evaluate(model_sknn, test_sessions, top_n=TOP_N, max_eval=MAX_EVAL, tag="SKNN ")
    print(f"  OK xong ({n_sknn} phiên)")
    print(f"  Recall@20 = {r_sknn:.4f} ({r_sknn*100:.2f}%) | MRR@20 = {mrr_sknn:.4f}")
    print(f"  Cải thiện so với Popularity: +{(r_sknn-r_pop)*100:.2f}%")

    # --- Phần 4: GRU4Rec ---
    print("\n" + "-" * 60)
    print(f"  PHẦN 4: GRU4Rec ({N_EPOCHS} epoch)")
    print("-" * 60)
    device = get_device()
    item2idx, n_items = build_item_index(train_sessions)
    model_gru, loss_history = train_gru(
        train_sessions, item2idx, n_items,
        n_epochs=N_EPOCHS, device=device, seed=cfg["SEED"],
    )
    r_gru, mrr_gru, n_gru = evaluate_gru(
        model_gru, test_sessions, item2idx, top_n=TOP_N, max_eval=MAX_EVAL, device=device,
    )
    print(f"  Recall@20 = {r_gru:.4f} ({r_gru*100:.2f}%) | MRR@20 = {mrr_gru:.4f}")
    torch.save(model_gru.state_dict(), cfg["GRU_MODEL_PATH"])

    # --- Phần 5: Thí nghiệm K (SKNN) ---
    print("\n" + "-" * 60)
    print("  PHẦN 5: THÍ NGHIỆM K (SKNN)")
    print("-" * 60)
    recall_by_k = []
    for k in K_VALUES:
        m = SKNN(k=k)
        m.fit(train_sessions)
        r_k, _, _ = evaluate(m, test_sessions, top_n=TOP_N, max_eval=MAX_EVAL, tag=f"K={k} ")
        recall_by_k.append(r_k)
        print(f"    K={k:4d} -> Recall@20 = {r_k:.4f}")

    # --- Phần 6: Tổng hợp + vẽ biểu đồ ---
    print("\n" + "-" * 60)
    print("  PHẦN 6: TỔNG HỢP + VẼ BIỂU ĐỒ")
    print("-" * 60)
    all_results = {
        "popularity": {"recall": r_pop, "mrr": mrr_pop},
        "sknn": {"recall": r_sknn, "mrr": mrr_sknn},
        "gru4rec": {"recall": r_gru, "mrr": mrr_gru},
        "k_experiment": {"k_values": K_VALUES, "recalls": recall_by_k},
        "gru_loss_history": loss_history,
        "meta": {
            "n_train": len(train_sessions),
            "n_test": len(test_sessions),
            "n_eval": n_sknn,
            "n_items": n_items,
            "n_epochs": N_EPOCHS,
            "sknn_k": SKNN_K,
            "split": info["split"],
            "smoke": smoke,
        },
    }
    with open(cfg["RESULTS_PATH"], "wb") as f:
        pickle.dump(all_results, f)
    print(f"  OK đã lưu {cfg['RESULTS_PATH']}")

    plot_all(all_results, cfg["OUTPUT_DIR"])

    # --- Bảng tổng hợp ---
    print("\n  " + "=" * 52)
    print(f"  {'Mô hình':<22}{'Recall@20':>14}{'MRR@20':>14}")
    print("  " + "-" * 52)
    print(f"  {'Popularity Baseline':<22}{r_pop:>14.4f}{mrr_pop:>14.4f}")
    print(f"  {'SKNN (k='+str(SKNN_K)+')':<22}{r_sknn:>14.4f}{mrr_sknn:>14.4f}")
    print(f"  {'GRU4Rec ('+str(N_EPOCHS)+' epoch)':<22}{r_gru:>14.4f}{mrr_gru:>14.4f}")
    print("  " + "=" * 52)
    print(f"  Đánh giá trên {n_sknn:,} phiên test (toàn bộ)")
    print("\n" + "=" * 60)
    print("  OK HOÀN THÀNH! Kết quả trong output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
