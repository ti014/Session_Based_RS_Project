"""Vẽ biểu đồ từ dict kết quả (nguồn: output/all_results.pkl).

Mọi con số đều lấy từ dict ``results`` nên không hardcode. Cả
``run_all.py`` (sau khi chạy) lẫn ``plot_final.py`` (vẽ lại) đều gọi
chung ``plot_all`` để tránh trùng lặp logic vẽ.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as cfg

COLORS = ["#e74c3c", "#2ecc71", "#3498db"]


def _model_labels(results: dict):
    meta = results.get("meta", {})
    sknn_k = meta.get("sknn_k", cfg.SKNN_K)
    n_epochs = meta.get("n_epochs", cfg.N_EPOCHS)
    return {
        "Popularity\nBaseline": results["popularity"],
        f"SKNN\n(k={sknn_k})": results["sknn"],
        f"GRU4Rec\n({n_epochs} epoch)": results["gru4rec"],
    }


def plot_comparison(results: dict, outdir=cfg.OUTPUT_DIR):
    """Biểu đồ cột so sánh Recall@20 và MRR@20 của 3 mô hình."""
    models = _model_labels(results)
    names = list(models.keys())
    recalls = [models[m]["recall"] for m in names]
    mrrs = [models[m]["mrr"] for m in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(names, recalls, color=COLORS, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Recall@20", fontsize=13)
    ax1.set_title("So sánh Recall@20 — Yoochoose", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, max(recalls) * 1.35 if max(recalls) > 0 else 1)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, recalls):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(recalls) * 0.02,
                 f"{val:.4f}\n({val*100:.1f}%)", ha="center", fontsize=11, fontweight="bold")

    bars2 = ax2.bar(names, mrrs, color=COLORS, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("MRR@20", fontsize=13)
    ax2.set_title("So sánh MRR@20 — Yoochoose", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, max(mrrs) * 1.35 if max(mrrs) > 0 else 1)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, mrrs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mrrs) * 0.02,
                 f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{outdir}/so_sanh_3_mo_hinh.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{outdir}/so_sanh_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  OK đã lưu: so_sanh_3_mo_hinh.png + so_sanh_final.png")


def plot_recall_by_k(results: dict, outdir=cfg.OUTPUT_DIR):
    """Biểu đồ ảnh hưởng của K (SKNN) đến Recall@20."""
    ke = results.get("k_experiment")
    if not ke:
        return
    pop = results["popularity"]["recall"]
    gru = results["gru4rec"]["recall"]
    plt.figure(figsize=(10, 6))
    plt.plot(ke["k_values"], ke["recalls"], "g-o", linewidth=2.5, markersize=10, label="SKNN")
    plt.axhline(y=pop, color="r", linestyle="--", linewidth=1.5, label=f"Popularity ({pop*100:.1f}%)")
    plt.axhline(y=gru, color="b", linestyle="-.", linewidth=1.5, label=f"GRU4Rec ({gru*100:.1f}%)")
    plt.xlabel("K (số hàng xóm)", fontsize=13)
    plt.ylabel("Recall@20", fontsize=13)
    plt.title("SKNN: Ảnh hưởng của K đến Recall@20", fontsize=14, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/recall_theo_k.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{outdir}/recall_theo_k_final.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK đã lưu: recall_theo_k.png + recall_theo_k_final.png")


def plot_loss_curve(results: dict, outdir=cfg.OUTPUT_DIR):
    """Đường cong loss huấn luyện GRU4Rec theo epoch."""
    loss_history = results.get("gru_loss_history")
    if not loss_history:
        return
    epochs = list(range(1, len(loss_history) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, "b-o", linewidth=2.5, markersize=8)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Loss huấn luyện", fontsize=13)
    plt.title("GRU4Rec: Loss huấn luyện theo Epoch", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3)
    if len(epochs) > 1:
        plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(f"{outdir}/gru4rec_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK đã lưu: gru4rec_loss_curve.png")


def plot_all(results: dict, outdir=cfg.OUTPUT_DIR):
    """Vẽ cả 3 biểu đồ từ dict kết quả."""
    plot_comparison(results, outdir)
    plot_recall_by_k(results, outdir)
    plot_loss_curve(results, outdir)
