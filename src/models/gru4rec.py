"""GRU4Rec — mạng hồi quy GRU cho khuyến nghị dựa trên phiên.

Luồng xử lý:
  1. Item -> Embedding (vector EMB_SIZE chiều)
  2. Chuỗi embedding -> GRU (học pattern theo thứ tự)
  3. Hidden state cuối -> Linear -> điểm số cho mọi item

Bao gồm cả dataset, hàm huấn luyện và hàm đánh giá riêng cho GRU (vì bước
suy luận của GRU khác với các mô hình dạng predict()).
"""
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .. import config as cfg


def get_device() -> str:
    """Trả về 'cuda' nếu có GPU, ngược lại 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_item_index(train_sessions: dict):
    """Đánh số item bắt đầu từ 1 (0 dành cho padding).

    Trả về ``(item2idx, n_items)`` với ``n_items`` đã cộng 1 cho padding.
    """
    all_items = sorted(set(i for items in train_sessions.values() for i in items))
    item2idx = {item: idx + 1 for idx, item in enumerate(all_items)}
    n_items = len(all_items) + 1
    return item2idx, n_items


class SessionDataset(Dataset):
    """Sinh mẫu (chuỗi tiền tố -> item kế tiếp) từ các phiên."""

    def __init__(self, sessions: dict, item2idx: dict, max_len: int = cfg.MAX_LEN):
        self.samples = []
        for items in sessions.values():
            items = items[-max_len:]
            for t in range(1, len(items)):
                input_seq = [item2idx.get(i, 0) for i in items[:t]]
                label = item2idx.get(items[t], 0)
                if label > 0:
                    self.samples.append((input_seq, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.LongTensor(seq), torch.LongTensor([label])


def collate_fn(batch):
    """Đệm (pad) các chuỗi trong batch về cùng độ dài."""
    seqs, labels = zip(*batch)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, torch.cat(labels)


class GRU4Rec(nn.Module):
    def __init__(self, n_items, emb_size=cfg.EMB_SIZE, hidden_size=cfg.HIDDEN_SIZE,
                 n_layers=cfg.N_LAYERS, dropout=cfg.DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(n_items, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, n_items)

    def forward(self, seq):
        emb = self.dropout(self.embedding(seq))
        lengths = (seq != 0).sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # Lấy hidden state tại bước cuối (không phải padding).
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_size).to(gru_out.device)
        h_last = gru_out.gather(1, idx).squeeze(1)
        return self.fc_out(self.dropout(h_last))


def train_gru(train_sessions, item2idx, n_items,
              val_sessions=None,
              n_epochs=cfg.N_EPOCHS, device=None, seed=cfg.SEED,
              emb_size=cfg.EMB_SIZE, hidden_size=cfg.HIDDEN_SIZE,
              batch_size=cfg.BATCH_SIZE, lr=cfg.LEARNING_RATE,
              weight_decay=cfg.WEIGHT_DECAY, grad_clip=cfg.GRAD_CLIP,
              top_n=cfg.TOP_N, patience=cfg.PATIENCE, min_delta=cfg.MIN_DELTA,
              val_max_eval=cfg.VAL_MAX_EVAL, early_stop_metric=cfg.EARLY_STOP_METRIC,
              verbose: bool = True):
    """Huấn luyện GRU4Rec. Trả về ``(model, history)``.

    Nếu truyền ``val_sessions``: mỗi epoch đánh giá Recall@20 trên val, lưu lại
    trọng số tốt nhất (best) và dừng sớm nếu không cải thiện sau ``patience``
    epoch (early stopping). Cuối cùng khôi phục trọng số best.

    Nếu ``val_sessions=None``: huấn luyện đủ ``n_epochs`` (hành vi cũ).

    ``history`` là dict gồm ``train_loss``, ``val_recall``, ``val_mrr``,
    ``best_epoch``, ``stopped_epoch``, ``early_stopped``, ...

    Cố định seed trước khi khởi tạo model để trọng số ban đầu tái lập.
    """
    if device is None:
        device = get_device()
    if val_sessions is not None and early_stop_metric != "recall":
        raise ValueError(f"early_stop_metric chỉ hỗ trợ 'recall', nhận: {early_stop_metric!r}")
    if val_sessions is not None and patience < 1:
        raise ValueError(f"patience phải >= 1 khi có validation, nhận: {patience}")
    torch.manual_seed(seed)
    if verbose:
        print(f"  Device: {device}")
        print(f"  Số item (vocab): {n_items:,}")

    dataset = SessionDataset(train_sessions, item2idx)
    if len(dataset) == 0:
        raise ValueError("Không có mẫu huấn luyện GRU sau khi tách train/val")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    if verbose:
        print(f"  Số mẫu: {len(dataset):,} | Số batch: {len(loader):,}")

    model = GRU4Rec(n_items, emb_size=emb_size, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if verbose:
        print(f"  Tổng tham số: {sum(p.numel() for p in model.parameters()):,}")
        mode = "early stopping theo val Recall@%d" % top_n if val_sessions is not None else "cố định"
        print(f"\n  Huấn luyện tối đa {n_epochs} epoch ({mode})...")

    history = {
        "train_loss": [], "val_recall": [], "val_mrr": [], "val_n_eval": [],
        "best_epoch": None, "best_val_recall": None, "best_val_mrr": None,
        "stopped_epoch": None, "early_stopped": False,
        "monitor": f"val_recall@{top_n}" if val_sessions is not None else None,
        "patience": patience, "min_delta": min_delta, "max_epochs": n_epochs,
    }

    best_score = -float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()
        for seq, labels in loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        history["train_loss"].append(avg_loss)

        if val_sessions is None:
            if verbose:
                print(f"    Epoch {epoch:2d}/{n_epochs} | Loss={avg_loss:.4f} | {time.time()-epoch_start:.1f}s")
            continue

        # --- Đánh giá trên validation để chọn epoch tốt nhất ---
        val_recall, val_mrr, val_n = evaluate_gru(
            model, val_sessions, item2idx, top_n=top_n,
            max_eval=val_max_eval, device=device, verbose=False)
        history["val_recall"].append(val_recall)
        history["val_mrr"].append(val_mrr)
        history["val_n_eval"].append(val_n)

        improved = val_recall > best_score + min_delta
        marker = ""
        if improved:
            best_score = val_recall
            history["best_epoch"] = epoch
            history["best_val_recall"] = val_recall
            history["best_val_mrr"] = val_mrr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " *best*"
        else:
            epochs_no_improve += 1

        if verbose:
            print(f"    Epoch {epoch:2d}/{n_epochs} | Loss={avg_loss:.4f} | "
                  f"val R@{top_n}={val_recall:.4f} MRR={val_mrr:.4f} | "
                  f"{time.time()-epoch_start:.1f}s{marker}")

        if epochs_no_improve >= patience:
            history["early_stopped"] = True
            history["stopped_epoch"] = epoch
            if verbose:
                print(f"  Early stopping tại epoch {epoch} "
                      f"(không cải thiện sau {patience} epoch). Best: epoch {history['best_epoch']}")
            break

    if history["stopped_epoch"] is None:
        history["stopped_epoch"] = len(history["train_loss"])

    # Khôi phục trọng số tốt nhất theo val.
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    if verbose:
        print("  OK huấn luyện xong!")
    return model, history


def evaluate_gru(model, test_sessions, item2idx, top_n=cfg.TOP_N,
                 max_eval=cfg.MAX_EVAL, device=None, verbose: bool = True):
    """Đánh giá GRU4Rec theo Recall@N / MRR@N (loại item đã xem)."""
    if device is None:
        device = get_device()
    if verbose:
        print("  Đang đánh giá GRU4Rec...")
    model.eval()
    hits, mrr_sum, n_eval = 0, 0.0, 0
    items_list = list(test_sessions.items())
    if max_eval:
        items_list = items_list[:max_eval]
    with torch.no_grad():
        for sid, items in items_list:
            if len(items) < 2:
                continue
            input_seq, label_item = items[:-1], items[-1]
            label_idx = item2idx.get(label_item, 0)
            if label_idx == 0:
                continue
            seq_idx = [item2idx.get(i, 0) for i in input_seq]
            seq_idx = [x for x in seq_idx if x > 0]
            if not seq_idx:
                continue
            seq_t = torch.LongTensor([seq_idx]).to(device)
            logits = model(seq_t).squeeze(0)
            logits[0] = -float("inf")
            for s in seq_idx:
                logits[s] = -float("inf")
            top_indices = logits.topk(top_n).indices.tolist()
            if label_idx in top_indices:
                hits += 1
                mrr_sum += 1.0 / (top_indices.index(label_idx) + 1)
            n_eval += 1
    recall = hits / n_eval if n_eval > 0 else 0
    mrr = mrr_sum / n_eval if n_eval > 0 else 0
    if verbose:
        print(f"  OK đánh giá xong ({n_eval} phiên)")
    return recall, mrr, n_eval
