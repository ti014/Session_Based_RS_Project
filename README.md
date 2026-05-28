# Chủ Đề 9: Session-Based Recommendation System

Hệ thống khuyến nghị dựa trên phiên (Session-Based Recommendation) cho bài toán dự đoán item tiếp theo từ chuỗi click ngắn của người dùng ẩn danh. Dự án triển khai đầy đủ pipeline học thuật: tiền xử lý dữ liệu Yoochoose, huấn luyện/đánh giá Popularity Baseline, SKNN, GRU4Rec, trực quan hóa kết quả, báo cáo LaTeX và slide thuyết trình.

## Thông tin nhóm

| Vai trò | Họ và tên | MSSV |
|---|---|---:|
| Thành viên | Phan Nguyễn Mai Phương | 25002711 |
| Thành viên | Ngô Quốc Hoàng | 25001771 |
| Thành viên | Bùi Thức Nam | 25003881 |
| Giảng viên hướng dẫn | TS. Lê Thị Vĩnh Thanh | - |

## Mục lục

- [Thông tin nhóm](#thông-tin-nhóm)
- [Tính năng chính](#tính-năng-chính)
- [Tech stack](#tech-stack)
- [Cấu trúc project](#cấu-trúc-project)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt nhanh](#cài-đặt-nhanh)
- [Chuẩn bị dataset](#chuẩn-bị-dataset)
- [Cách chạy](#cách-chạy)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Quy trình xử lý dữ liệu](#quy-trình-xử-lý-dữ-liệu)
- [Mô hình](#mô-hình)
- [Giao thức đánh giá](#giao-thức-đánh-giá)
- [Notebook](#notebook)
- [Báo cáo và slide](#báo-cáo-và-slide)
- [Artifact đầu ra](#artifact-đầu-ra)
- [Cấu hình thí nghiệm](#cấu-hình-thí-nghiệm)
- [Tái lập kết quả](#tái-lập-kết-quả)
- [Troubleshooting](#troubleshooting)
- [Hạn chế](#hạn-chế)
- [Hướng phát triển](#hướng-phát-triển)

## Tính năng chính

- Tiền xử lý dataset Yoochoose theo quy trình chống rò rỉ dữ liệu.
- Chia train/test theo thời gian: dùng phiên sớm để dự đoán phiên muộn.
- Lọc item hiếm chỉ dựa trên tập train.
- Triển khai 3 mô hình:
  - Popularity Baseline
  - SKNN (Session K-Nearest Neighbors) với inverted index
  - GRU4Rec bằng PyTorch
- Đánh giá bằng Recall@20 và MRR@20 theo giao thức leave-one-out.
- Loại item đã xuất hiện trong phiên truy vấn khỏi danh sách gợi ý.
- Thí nghiệm ảnh hưởng của K trong SKNN.
- Lưu toàn bộ kết quả vào `output/all_results.pkl` để notebook, biểu đồ, report và slide dùng chung một nguồn số liệu.
- Vẽ biểu đồ có dấu tiếng Việt:
  - So sánh Recall@20 và MRR@20
  - Ảnh hưởng của K trong SKNN
  - Loss huấn luyện GRU4Rec
- Notebook dùng cho báo cáo/trình bày; logic chính nằm trong package `src/`.
- Có báo cáo LaTeX và slide Beamer đã build sẵn.

## Tech stack

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3 |
| Xử lý dữ liệu | NumPy, pandas |
| Deep learning | PyTorch |
| Biểu đồ | Matplotlib |
| Notebook | Jupyter Notebook |
| Báo cáo | LaTeX (`report/main.tex`) |
| Slide | Beamer (`slide/main.tex`) |
| Dataset | Yoochoose - RecSys Challenge 2015 |
| Artifact kết quả | Pickle nhỏ (`output/all_results.pkl`), PNG, PDF |

## Cấu trúc project

```text
Session_Based_RS_Project/
|-- README.md
|-- requirements.txt
|-- run_all.py              <- Orchestrator chạy toàn bộ pipeline
|-- plot_final.py           <- Vẽ lại biểu đồ từ output/all_results.pkl
|-- src/                    <- Package Python chứa logic chính
|   |-- __init__.py
|   |-- config.py           <- Đường dẫn + siêu tham số
|   |-- data.py             <- Đọc dữ liệu, tiền xử lý, chia train/test
|   |-- evaluate.py         <- Recall@N, MRR@N cho Popularity/SKNN
|   |-- plots.py            <- Hàm vẽ biểu đồ từ kết quả đã lưu
|   |-- models/
|   |   |-- __init__.py
|   |   |-- popularity.py   <- Popularity Baseline
|   |   |-- sknn.py         <- SKNN với inverted index
|   |   |-- gru4rec.py      <- GRU4Rec, Dataset, train/evaluate
|-- data/
|   |-- .gitkeep             <- Giữ thư mục data/ trên GitHub
|   |-- yoochoose-clicks.dat <- Tải riêng từ Kaggle, không commit
|   |-- processed_data.pkl   <- Sinh ra khi chạy pipeline, không commit
|-- output/
|   |-- all_results.pkl      <- Kết quả nhỏ, giữ lại để xem biểu đồ/notebook 05
|   |-- so_sanh_3_mo_hinh.png
|   |-- so_sanh_final.png
|   |-- recall_theo_k.png
|   |-- recall_theo_k_final.png
|   |-- gru4rec_loss_curve.png
|-- 01_gioi_thieu.ipynb
|-- 02_tien_xu_ly.ipynb
|-- 03_sknn.ipynb
|-- 04_gru4rec.ipynb
|-- 05_so_sanh_ket_qua.ipynb
|-- report/
|   |-- main.tex
|   |-- main.pdf
|-- slide/
|   |-- main.tex
|   |-- main.pdf
```

## Yêu cầu môi trường

Tối thiểu:

- Python 3.9 trở lên.
- `pip`.
- Jupyter Notebook nếu muốn chạy notebook.
- PyTorch. Có GPU thì GRU4Rec chạy nhanh hơn; không có GPU vẫn chạy được bằng CPU.
- MiKTeX/TeX Live + `latexmk` nếu muốn build lại report/slide.

Thư viện Python nằm trong `requirements.txt`:

```text
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
torch>=1.9.0
jupyter>=1.0.0
```

## Cài đặt nhanh

### 1. Mở terminal tại thư mục project

```bash
cd D:/IUH/ICT/Session_Based_RS_Project
```

Hoặc trên PowerShell:

```powershell
Set-Location "D:\IUH\ICT\Session_Based_RS_Project"
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Cài thư viện

```bash
pip install -r requirements.txt
```

Kiểm tra cài đặt:

```bash
python - <<'PY'
import numpy, pandas, matplotlib, torch
print('NumPy:', numpy.__version__)
print('pandas:', pandas.__version__)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
PY
```

Kỳ vọng:

```text
NumPy: ...
pandas: ...
PyTorch: ...
CUDA: True/False
```

## Chuẩn bị dataset

Dataset dùng trong bài:

- Yoochoose - RecSys Challenge 2015
- Kaggle: https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015

Cần đặt file click vào:

```text
data/yoochoose-clicks.dat
```

File `yoochoose-buys.dat` không bắt buộc vì pipeline hiện tại chỉ dùng click data cho bài toán dự đoán item click tiếp theo.

Trên GitHub, thư mục `data/` chỉ giữ `.gitkeep`. Dataset gốc và file xử lý trung gian bị ignore để tránh đẩy file lớn.

Cấu trúc `data/` sau khi người dùng tự tải/chạy pipeline:

```text
data/
|-- .gitkeep
|-- yoochoose-clicks.dat        <- tải riêng từ Kaggle
|-- yoochoose-buys.dat          <- không bắt buộc cho pipeline hiện tại
|-- processed_data.pkl          <- sinh ra sau khi tiền xử lý
```

## Cách chạy

### Cách 1: Chạy toàn bộ pipeline

```bash
python run_all.py
```

Lệnh này thực hiện toàn bộ quy trình:

1. Đọc `data/yoochoose-clicks.dat`.
2. Lấy mẫu 1/64 theo phiên với seed cố định.
3. Chia train/test theo thời gian.
4. Lọc item hiếm dựa trên train.
5. Lọc độ dài phiên trong `[2, 20]`.
6. Huấn luyện/đánh giá Popularity Baseline.
7. Huấn luyện/đánh giá SKNN.
8. Huấn luyện/đánh giá GRU4Rec.
9. Thử nhiều giá trị K cho SKNN.
10. Lưu kết quả vào `output/all_results.pkl`.
11. Vẽ biểu đồ vào `output/`.

Đầu ra quan trọng:

```text
output/all_results.pkl
output/gru4rec_model.pt        <- sinh ra khi chạy, không commit mặc định
output/so_sanh_final.png
output/recall_theo_k_final.png
output/gru4rec_loss_curve.png
```

### Cách 2: Chạy nhanh để kiểm tra logic

```bash
python run_all.py --smoke
```

Chế độ smoke dùng cấu hình nhỏ hơn:

- Lấy mẫu 1/512.
- GRU4Rec chạy 1 epoch.
- Đánh giá tối đa 300 phiên.
- K values: `[50, 500]`.

Dùng khi muốn kiểm tra code có chạy không, không dùng để lấy số báo cáo cuối.

### Cách 3: Chạy notebook theo thứ tự

```bash
jupyter notebook
```

Sau đó chạy lần lượt:

1. `01_gioi_thieu.ipynb`
2. `02_tien_xu_ly.ipynb`
3. `03_sknn.ipynb`
4. `04_gru4rec.ipynb`
5. `05_so_sanh_ket_qua.ipynb`

Ghi chú:

- Notebook nên mở từ thư mục gốc project.
- Notebook đã có bootstrap `sys.path` để import package `src/`.
- Logic chính không viết lặp trong notebook; notebook chủ yếu dùng để trình bày, chạy minh họa và xuất kết quả.

### Cách 4: Chỉ vẽ lại biểu đồ từ kết quả đã lưu

```bash
python plot_final.py
```

Lệnh này:

- Không huấn luyện lại.
- Không đánh giá lại.
- Chỉ đọc `output/all_results.pkl`.
- Vẽ lại biểu đồ PNG trong `output/`.

## Kết quả thực nghiệm

Kết quả thật đang lưu trong `output/all_results.pkl`:

| Mô hình | Recall@20 | MRR@20 | Ghi chú |
|---|---:|---:|---|
| Popularity Baseline | 0.0187 | 0.0041 | Chỉ dựa trên item phổ biến |
| SKNN (K=500) | 0.2952 | 0.1349 | Tốt nhất trong cấu hình hiện tại |
| GRU4Rec (10 epoch) | 0.2796 | 0.1262 | Gần SKNN nhưng chưa vượt |

Diễn giải nhanh:

- Popularity rất thấp, chứng tỏ chỉ gợi ý item phổ biến không đủ cho session-based recommendation.
- SKNN vượt mạnh Popularity vì tận dụng được item trong phiên hiện tại.
- GRU4Rec học được thứ tự click và đạt kết quả gần SKNN, nhưng trong cấu hình 1/64 dataset + 10 epoch vẫn thấp hơn SKNN.
- Kết luận này chỉ áp dụng cho cấu hình thí nghiệm hiện tại, không nên suy rộng tuyệt đối sang mọi dataset.

### Thống kê thí nghiệm

| Thông tin | Giá trị |
|---|---:|
| Số phiên train | 108,263 |
| Số phiên test | 8,004 |
| Số phiên đánh giá | 8,004 |
| Số item sau xử lý (gồm padding cho GRU) | 10,435 |
| Số epoch GRU4Rec | 10 |
| K của SKNN chính | 500 |
| Split | 90% phiên sớm / 10% phiên muộn |

### Ảnh hưởng của K trong SKNN

| K | Recall@20 |
|---:|---:|
| 50 | 0.2702 |
| 100 | 0.2846 |
| 200 | 0.2951 |
| 500 | 0.2952 |

Nhận xét: Recall tăng từ K=50 đến K=200, sau đó gần bão hòa ở K=500.

### Loss GRU4Rec

Loss theo 10 epoch:

```text
[7.7946, 6.3775, 5.6863, 5.2624, 4.9775, 4.7761, 4.6230, 4.5060, 4.4118, 4.3349]
```

Loss giảm đều, cho thấy mô hình có học. Tuy vậy, việc loss giảm không đồng nghĩa mô hình chắc chắn vượt SKNN về Recall@20/MRR@20.

## Kiến trúc hệ thống

Dự án tách rõ 3 lớp:

1. **Orchestrator**: `run_all.py`
2. **Package logic**: `src/`
3. **Báo cáo/trình bày**: notebook, LaTeX report, slide

### Luồng chạy tổng quát

```text
Yoochoose clicks
      |
      v
src.data.load_clicks()
      |
      v
src.data.preprocess()
      |
      +--> data/processed_data.pkl
      |
      v
train_sessions, test_sessions
      |
      +--> PopularityBaseline --> src.evaluate.evaluate()
      |
      +--> SKNN               --> src.evaluate.evaluate()
      |
      +--> GRU4Rec            --> src.models.gru4rec.evaluate_gru()
      |
      v
output/all_results.pkl
      |
      v
src.plots.plot_all()
      |
      v
PNG charts + report + slide
```

### Vai trò từng file chính

| File | Vai trò |
|---|---|
| `run_all.py` | Điều phối toàn bộ pipeline, không chứa logic thuật toán dài |
| `plot_final.py` | Vẽ lại biểu đồ từ kết quả đã lưu |
| `src/config.py` | Nơi duy nhất chứa đường dẫn và siêu tham số |
| `src/data.py` | Đọc Yoochoose, lấy mẫu, chia train/test, lọc dữ liệu |
| `src/evaluate.py` | Recall@N và MRR@N cho model có hàm `predict()` |
| `src/models/popularity.py` | Popularity Baseline |
| `src/models/sknn.py` | SKNN với inverted index |
| `src/models/gru4rec.py` | GRU4Rec, PyTorch Dataset, training loop, evaluation |
| `src/plots.py` | Vẽ biểu đồ từ dict kết quả |
| `05_so_sanh_ket_qua.ipynb` | Đọc `all_results.pkl`, in bảng, vẽ/hiển thị biểu đồ, phân tích |

## Quy trình xử lý dữ liệu

Pipeline trong `src/data.py` cố tình giữ thứ tự xử lý để tránh data leakage.

### Bước 1: Đọc click data

`yoochoose-clicks.dat` không có header, được đọc với schema:

| Cột | Ý nghĩa |
|---|---|
| `SessionID` | ID phiên |
| `Timestamp` | Thời điểm click |
| `ItemID` | ID item |
| `Category` | Category gốc trong Yoochoose |

### Bước 2: Lấy mẫu theo phiên

Cấu hình mặc định:

```python
SAMPLE_FRACTION = 64
SEED = 42
```

Tức là lấy 1/64 số phiên, cố định seed để tái lập.

### Bước 3: Sắp xếp theo thời gian trong phiên

Dữ liệu được sort theo:

```text
(SessionID, Timestamp)
```

Điều này đảm bảo chuỗi item trong mỗi phiên đúng thứ tự click.

### Bước 4: Chia train/test theo thời gian

Cấu hình:

```python
TRAIN_RATIO = 0.9
```

Cách chia:

- 90% phiên sớm nhất -> train.
- 10% phiên muộn nhất -> test.

Lý do: mô phỏng tình huống thực tế “dùng quá khứ để dự đoán tương lai”, tốt hơn chia random đối với dữ liệu thời gian.

### Bước 5: Lọc item hiếm chỉ trên train

Cấu hình:

```python
MIN_ITEM_FREQ = 5
```

Chỉ đếm tần suất item trong tập train. Sau đó áp cùng tập item hợp lệ này cho train và test.

Lý do học thuật: nếu đếm item hiếm trên cả test thì mô hình đã nhìn thông tin từ tương lai, gây data leakage.

### Bước 6: Lọc độ dài phiên

Cấu hình:

```python
MIN_LEN = 2
MAX_LEN = 20
```

- Phiên ngắn hơn 2 không đánh giá được next-item.
- Phiên quá dài bị cắt/lọc để giữ bài toán ổn định và giảm nhiễu.

## Mô hình

### 1. Popularity Baseline

File:

```text
src/models/popularity.py
```

Ý tưởng:

- Đếm tần suất item trong train.
- Khi dự đoán, trả về top item phổ biến nhất chưa xuất hiện trong phiên hiện tại.

Ưu điểm:

- Rất đơn giản.
- Không cần huấn luyện.
- Là lower bound cần có trong mọi bài so sánh.

Nhược điểm:

- Không dùng thứ tự click.
- Không cá nhân hóa theo phiên.
- Kết quả thấp trong thí nghiệm này.

### 2. SKNN - Session K-Nearest Neighbors

File:

```text
src/models/sknn.py
```

Ý tưởng:

1. Lấy phiên truy vấn hiện tại.
2. Tìm các phiên lịch sử có item trùng với phiên truy vấn.
3. Tính độ tương tự giữa phiên truy vấn và phiên lịch sử.
4. Lấy K phiên gần nhất.
5. Cộng điểm cho các item xuất hiện trong những phiên gần nhất nhưng chưa có trong phiên truy vấn.

Công thức similarity:

```text
sim(Sq, Sn) = |Sq ∩ Sn| / sqrt(|Sq| * |Sn|)
```

Trong đó:

- `Sq`: phiên truy vấn.
- `Sn`: phiên lịch sử.
- `|Sq ∩ Sn|`: số item chung.

Cách tối ưu:

- Dùng inverted index `item -> danh sách session chứa item`.
- Không cần duyệt toàn bộ train session cho mỗi truy vấn.
- Chỉ xét những session có ít nhất một item chung với phiên truy vấn.

Cấu hình chính:

```python
SKNN_K = 500
K_VALUES = [50, 100, 200, 500]
```

Ưu điểm:

- Mạnh trong session-based recommendation.
- Không cần GPU.
- Dễ giải thích.
- Là baseline nghiêm túc, không chỉ là baseline “cho có”.

Nhược điểm:

- Dự đoán có thể chậm nếu dữ liệu rất lớn.
- Phụ thuộc vào overlap item giữa phiên hiện tại và lịch sử.
- Không học representation như mô hình deep learning.

### 3. GRU4Rec

File:

```text
src/models/gru4rec.py
```

Ý tưởng:

GRU4Rec xem chuỗi click như chuỗi thời gian. Mỗi item được ánh xạ thành embedding, sau đó đi qua GRU để học trạng thái ẩn của phiên.

Kiến trúc:

```text
ItemID -> Embedding -> GRU -> Dropout -> Linear -> score cho mọi item
```

Cấu hình mặc định:

| Tham số | Giá trị |
|---|---:|
| `N_EPOCHS` | 10 |
| `EMB_SIZE` | 64 |
| `HIDDEN_SIZE` | 128 |
| `N_LAYERS` | 1 |
| `DROPOUT` | 0.25 |
| `BATCH_SIZE` | 256 |
| `LEARNING_RATE` | 1e-3 |
| `WEIGHT_DECAY` | 1e-5 |
| `GRAD_CLIP` | 5.0 |

Cách tạo mẫu train:

Với session:

```text
[i1, i2, i3, i4]
```

Sinh các mẫu:

```text
[i1]         -> i2
[i1, i2]     -> i3
[i1, i2, i3] -> i4
```

Ưu điểm:

- Học được thứ tự click.
- Có thể mở rộng sang kiến trúc sâu hơn.
- Phù hợp khi có dữ liệu lớn và tune tốt.

Nhược điểm:

- Cần huấn luyện.
- Nhạy với siêu tham số.
- Trong thí nghiệm hiện tại chưa vượt SKNN.

## Giao thức đánh giá

File:

```text
src/evaluate.py
src/models/gru4rec.py
```

Giao thức: leave-one-out next-item prediction.

Với mỗi phiên test:

```text
[item_1, item_2, ..., item_t]
```

Tách thành:

```text
input = [item_1, item_2, ..., item_{t-1}]
label = item_t
```

Mô hình sinh top-N khuyến nghị từ `input`. Nếu `label` nằm trong top-N thì tính là hit.

### Recall@N

```text
Recall@N = số phiên hit / số phiên đánh giá
```

Ý nghĩa: trong bao nhiêu phần trăm phiên, item thật sự tiếp theo xuất hiện trong top-N gợi ý.

### MRR@N

```text
MRR@N = trung bình 1 / rank(label)
```

Nếu item đúng đứng hạng càng cao, MRR càng lớn.

### Loại item đã xem

Khi gợi ý, các item đã xuất hiện trong phiên truy vấn bị loại khỏi danh sách ứng viên.

Lý do: trong khuyến nghị next-item, thường không muốn gợi ý lại item người dùng vừa click trong cùng phiên.

## Notebook

### `01_gioi_thieu.ipynb`

Giới thiệu:

- Bài toán Session-Based Recommendation.
- Khác biệt với Collaborative Filtering truyền thống.
- Dataset Yoochoose.
- Metric Recall@20 và MRR@20.

### `02_tien_xu_ly.ipynb`

Trình bày:

- Dữ liệu mẫu nhỏ.
- Cách đọc Yoochoose.
- Quy trình tiền xử lý chống data leakage.
- Lưu `data/processed_data.pkl`.

### `03_sknn.ipynb`

Trình bày:

- Công thức SKNN.
- Ví dụ tính tay.
- Chạy Popularity Baseline và SKNN trên dữ liệu thật.
- Lưu kết quả vào `output/all_results.pkl`.

### `04_gru4rec.ipynb`

Trình bày:

- Cách đánh số item.
- Dataset PyTorch.
- Kiến trúc GRU4Rec.
- Huấn luyện và đánh giá GRU4Rec.
- Lưu model vào `output/gru4rec_model.pt`.

### `05_so_sanh_ket_qua.ipynb`

Trình bày:

- Đọc kết quả thật từ `output/all_results.pkl`.
- In bảng so sánh.
- Vẽ biểu đồ.
- Phân tích vì sao SKNN tốt hơn Popularity và vì sao GRU4Rec chưa vượt SKNN trong cấu hình hiện tại.

## Báo cáo và slide

### Report

Nguồn LaTeX:

```text
report/main.tex
```

PDF đã build:

```text
report/main.pdf
```

Build lại:

```bash
cd report
latexmk -pdf -interaction=nonstopmode main.tex
```

### Slide

Nguồn LaTeX:

```text
slide/main.tex
```

PDF đã build:

```text
slide/main.pdf
```

Build lại:

```bash
cd slide
latexmk -pdf -interaction=nonstopmode main.tex
```

Ghi chú:

- Report và slide dùng số liệu thống nhất với `output/all_results.pkl`.
- Biểu đồ trong report lấy từ `output/`.
- Nếu chạy lại pipeline, nên chạy `python plot_final.py` hoặc `python run_all.py` trước khi build report.

## Artifact đầu ra

| File | Sinh bởi | Ý nghĩa |
|---|---|---|
| `data/processed_data.pkl` | `src.data.save_processed()` | Train/test session sau tiền xử lý |
| `output/all_results.pkl` | `run_all.py` hoặc notebook 03+04 | Kết quả tổng hợp |
| `output/gru4rec_model.pt` | GRU4Rec training | Trọng số PyTorch của GRU4Rec; sinh ra khi chạy, không commit mặc định |
| `output/so_sanh_3_mo_hinh.png` | `src.plots.plot_comparison()` | Biểu đồ so sánh 3 mô hình |
| `output/so_sanh_final.png` | `src.plots.plot_comparison()` | Bản dùng trong report |
| `output/recall_theo_k.png` | `src.plots.plot_recall_by_k()` | Biểu đồ K của SKNN |
| `output/recall_theo_k_final.png` | `src.plots.plot_recall_by_k()` | Bản dùng trong report |
| `output/gru4rec_loss_curve.png` | `src.plots.plot_loss_curve()` | Loss GRU4Rec theo epoch |
| `report/main.pdf` | LaTeX | Báo cáo cuối |
| `slide/main.pdf` | LaTeX Beamer | Slide thuyết trình |

## Cấu hình thí nghiệm

Tất cả cấu hình chính nằm trong:

```text
src/config.py
```

### Đường dẫn

```python
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CLICKS_PATH = DATA_DIR / "yoochoose-clicks.dat"
PROCESSED_PATH = DATA_DIR / "processed_data.pkl"
RESULTS_PATH = OUTPUT_DIR / "all_results.pkl"
GRU_MODEL_PATH = OUTPUT_DIR / "gru4rec_model.pt"
```

### Tiền xử lý

```python
SEED = 42
SAMPLE_FRACTION = 64
MIN_ITEM_FREQ = 5
MIN_LEN, MAX_LEN = 2, 20
TRAIN_RATIO = 0.9
```

### Đánh giá

```python
TOP_N = 20
MAX_EVAL = None
```

`MAX_EVAL = None` nghĩa là đánh giá toàn bộ test set.

### SKNN

```python
SKNN_K = 500
K_VALUES = [50, 100, 200, 500]
```

### GRU4Rec

```python
N_EPOCHS = 10
EMB_SIZE = 64
HIDDEN_SIZE = 128
N_LAYERS = 1
DROPOUT = 0.25
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
```

## Tái lập kết quả

Để tái lập kết quả cuối:

```bash
python run_all.py
```

Sau khi chạy xong, kiểm tra nhanh kết quả:

```bash
python - <<'PY'
import pickle
from src import config

with open(config.RESULTS_PATH, 'rb') as f:
    r = pickle.load(f)

for name in ['popularity', 'sknn', 'gru4rec']:
    print(name, r[name])
print(r['meta'])
PY
```

Kỳ vọng gần với:

```text
popularity {'recall': 0.018740629685157422, 'mrr': 0.004133761371299721}
sknn {'recall': 0.2952273863068466, 'mrr': 0.13490508093096515}
gru4rec {'recall': 0.27964513307509686, 'mrr': 0.12619739606063746}
```

Lưu ý:

- Popularity và SKNN gần như tái lập hoàn toàn nếu dữ liệu và seed không đổi.
- GRU4Rec có thể lệch rất nhỏ giữa CPU/GPU hoặc phiên bản PyTorch khác nhau.
- Không nên chỉnh số thủ công trong report/slide; nguồn số đúng là `output/all_results.pkl`.

## Kiểm tra nhanh project trước khi nộp

```bash
python - <<'PY'
import ast, json
from pathlib import Path

for path in list(Path('src').rglob('*.py')) + [Path('run_all.py'), Path('plot_final.py')]:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print('PY_SYNTAX_OK')

for path in Path('.').glob('0*.ipynb'):
    json.loads(path.read_text(encoding='utf-8'))
    print('NOTEBOOK_JSON_OK', path.name)

required = [
    'output/all_results.pkl',
    'output/so_sanh_final.png',
    'output/recall_theo_k_final.png',
    'output/gru4rec_loss_curve.png',
    'report/main.pdf',
    'slide/main.pdf',
]
# Các file sau chỉ có sau khi chạy lại pipeline đầy đủ và không commit mặc định:
# data/processed_data.pkl, output/gru4rec_model.pt
missing = [x for x in required if not Path(x).exists()]
if missing:
    raise SystemExit('MISSING: ' + ', '.join(missing))
print('ARTIFACTS_OK')
PY
```

Kỳ vọng:

```text
PY_SYNTAX_OK
NOTEBOOK_JSON_OK 01_gioi_thieu.ipynb
NOTEBOOK_JSON_OK 02_tien_xu_ly.ipynb
NOTEBOOK_JSON_OK 03_sknn.ipynb
NOTEBOOK_JSON_OK 04_gru4rec.ipynb
NOTEBOOK_JSON_OK 05_so_sanh_ket_qua.ipynb
ARTIFACTS_OK
```

## Troubleshooting

### 1. Lỗi không tìm thấy dataset

Lỗi thường gặp:

```text
FileNotFoundError: data/yoochoose-clicks.dat
```

Cách xử lý:

1. Tải dataset Yoochoose từ Kaggle.
2. Giải nén nếu cần.
3. Đặt đúng file vào:

```text
data/yoochoose-clicks.dat
```

4. Chạy lại:

```bash
python run_all.py --smoke
```

### 2. Windows console lỗi tiếng Việt

Nếu thấy lỗi dạng:

```text
UnicodeEncodeError: 'charmap' codec can't encode character
```

Cách xử lý trong PowerShell:

```powershell
$env:PYTHONUTF8=1
python run_all.py --smoke
```

Hoặc chạy một lệnh:

```bash
PYTHONUTF8=1 python run_all.py --smoke
```

Trong project, `src/__init__.py` đã cố gắng cấu hình stdout UTF-8 để log tiếng Việt ổn hơn.

### 3. PyTorch không nhận GPU

Kiểm tra:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

Nếu kết quả là `False`, GRU4Rec vẫn chạy bằng CPU nhưng lâu hơn. Muốn dùng GPU cần cài bản PyTorch phù hợp CUDA trên máy.

### 4. Chạy full quá lâu

Dùng smoke test trước:

```bash
python run_all.py --smoke
```

Nếu smoke chạy ổn mới chạy full:

```bash
python run_all.py
```

### 5. `output/all_results.pkl` chưa tồn tại

Một số notebook hoặc `plot_final.py` cần file này.

Cách tạo:

```bash
python run_all.py
```

Hoặc chạy lần lượt notebook 02, 03, 04 trước khi mở notebook 05.

### 6. Biểu đồ không cập nhật

Nếu đã có `output/all_results.pkl`, chạy:

```bash
python plot_final.py
```

Nếu muốn sinh lại cả kết quả và biểu đồ:

```bash
python run_all.py
```

### 7. LaTeX build lỗi thiếu package

Cần cài MiKTeX hoặc TeX Live. Sau đó build:

```bash
cd report
latexmk -pdf -interaction=nonstopmode main.tex
```

Nếu MiKTeX hỏi cài package, chọn cài tự động.

### 8. Notebook import lỗi `src`

Nên mở Jupyter từ thư mục gốc project:

```bash
cd D:/IUH/ICT/Session_Based_RS_Project
jupyter notebook
```

Notebook đã có đoạn bootstrap để thêm project root vào `sys.path`, nhưng chạy từ root vẫn an toàn nhất.

## Hạn chế

- Thí nghiệm dùng mẫu 1/64 của Yoochoose, nên số tuyệt đối có thể khác khi chạy toàn bộ dataset.
- Chỉ dùng click data, chưa dùng category, price, dwell time hoặc thông tin nội dung item.
- GRU4Rec mới chạy 10 epoch và chưa tune rộng siêu tham số.
- Chưa thử các mô hình mạnh hơn như NARM, STAN, SR-GNN, BERT4Rec.
- Đánh giá offline bằng leave-one-out chưa thay thế được A/B testing trong hệ thống thật.

## Hướng phát triển

Các hướng mở rộng hợp lý:

1. Chạy toàn bộ Yoochoose thay vì mẫu 1/64.
2. Tune GRU4Rec: learning rate, dropout, hidden size, batch size, loss function.
3. Thử NARM để thêm attention vào GRU.
4. Thử SR-GNN để mô hình hóa session dưới dạng đồ thị.
5. Thử V-SKNN hoặc STAN để đưa vị trí/thời gian vào SKNN.
6. Bổ sung metadata item như category hoặc price.
7. Đánh giá thêm Recall@5, Recall@10, NDCG@20, HitRate@20.
8. Xây dựng demo API/web nhỏ để nhập session và trả về top-N item.

## Ghi chú đưa lên GitHub

Repo GitHub nên giữ source code, notebook, biểu đồ nhỏ và PDF cuối; không nên commit dataset gốc hoặc model checkpoint lớn.

Nên commit:

```text
.gitignore
README.md
requirements.txt
run_all.py
plot_final.py
src/
data/.gitkeep
output/all_results.pkl
output/*.png
01_gioi_thieu.ipynb
02_tien_xu_ly.ipynb
03_sknn.ipynb
04_gru4rec.ipynb
05_so_sanh_ket_qua.ipynb
report/main.tex
report/main.pdf
slide/main.tex
slide/main.pdf
```

Không nên commit:

```text
data/yoochoose-clicks.dat       <- rất lớn, tải riêng từ Kaggle
data/yoochoose-buys.dat
data/processed_data.pkl         <- sinh lại được
output/gru4rec_model.pt         <- sinh lại được, binary checkpoint
__pycache__/
*.log
LaTeX auxiliary files (*.aux, *.toc, *.fls, ...)
```

Nếu giảng viên yêu cầu nộp artifact đầy đủ ngoài GitHub, có thể nộp riêng `data/processed_data.pkl` và `output/gru4rec_model.pt` qua Google Drive hoặc file nén riêng.

## Tóm tắt một câu

Trong cấu hình thí nghiệm hiện tại, SKNN là mô hình tốt nhất và thực dụng nhất; GRU4Rec có tiềm năng nhưng cần thêm dữ liệu, epoch và tune siêu tham số để vượt baseline mạnh như SKNN.
