# YOLOv3 from Scratch (PyTorch Implementation)

Dự án tự hiện thực hóa mô hình phát hiện vật thể **YOLOv3 (You Only Look Once v3)** từ số 0 bằng PyTorch dựa trên paper gốc *"YOLOv3: An Incremental Improvement"* của Joseph Redmon & Ali Farhadi. 



## 📑 Mục lục
1. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
2. [Cài đặt & Cách chạy nhanh (Quickstart)](#-cài-đặt--cách-chạy-nhanh-quickstart)
3. [Những kiến thức cốt lõi đúc kết từ dự án](#-những-kiến-thức-cốt-lõi-đúc-kết-từ-dự-án)
   - [1. Xử lý Dữ liệu (`dataset.py`)](#1-xử-lý-dữ-liệu-datasetpy)
   - [2. Hàm Loss (`loss.py`)](#2-hàm-loss-losspy)
   - [3. Kiến trúc Mô hình (`model.py`)](#3-kiến-trúc-mô-hình-modelpy)
   - [4. Huấn luyện Mô hình (`train.py`)](#4-huấn-luyện-mô-hình-trainpy)
   - [5. Tiện ích & Đánh giá (`utils.py`, `config.py`)](#5-tiện-ích--đánh-giá-utilspy-configpy)
   - [6. Chạy Thực Tế (`inference.py`)](#6-chạy-thực-tế-inferencepy)
   - [7. Đúc kết & Hướng phát triển (`note.md`)](#7-đúc-kết--hướng-phát-triển-notemd)
4. [Tác giả](#-tác-giả)

---

## 📂 Cấu trúc thư mục

```text
├── config.py             
├── dataset.py            
├── model.py              
├── loss.py               
├── utils.py              
├── train.py              
├── inference.py          
└── note.md               
└── remote_training.ipynb 
```

---

## 🚀 Cài đặt & Cách chạy nhanh (Quickstart)

### 1. Cài đặt môi trường
```bash
pip install torch torchvision numpy pandas opencv-python albumentations matplotlib tqdm
```

### 2. Chuẩn bị dữ liệu
Tải bộ dữ liệu PASCAL VOC đã được chuẩn hóa cho YOLO tại Kaggle: [PASCAL VOC Dataset (Kaggle)](https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-yolo-works-with-albumentations)

Giải nén và đặt theo cấu trúc thư mục được khai báo trong `config.py`:
```text
PASCAL_VOC/
├── images/
├── labels/
├── train.csv
└── test.csv
```

### 3. Huấn luyện mô hình
```bash
python train.py
```

### 4. Chạy Real-Time Inference với Webcam
```bash
python inference.py
```

---

## 🧠 Những kiến thức cốt lõi đúc kết từ dự án

### 1. Xử lý Dữ liệu (`dataset.py`)
- **Định dạng nhãn gốc:** Dạng `[class, x, y, w, h]` chuẩn hóa về `[0, 1]`. Dùng `np.roll` để đổi thứ tự thành `[x, y, w, h, class]`.
- **Cấu trúc Target:** Gồm 3 tensor tương ứng 3 scale (`S = [13, 26, 52]`), mỗi tensor có kích thước `[num_anchors_per_scale, S, S, 6]` với 6 giá trị là `[objectness, x, y, w, h, class]`.
- **Cơ chế gán Anchor:**
  - Dùng `iou_width_height` để so sánh kích thước box với 9 anchor, sau đó sắp xếp IoU từ cao đến thấp (`argsort`).
  - `scale_idx = anchor_idx // 3`, `anchor_on_scale = anchor_idx % 3`.
  - Mỗi scale chỉ gán tối đa 1 anchor cho 1 vật thể (`has_anchor`).
  - Nếu ô lưới đã có vật thể trước đó chiếm anchor tốt nhất (`anchor_taken`), vòng lặp sẽ tìm anchor tiếp theo phù hợp cho vật thể thứ hai.
- **Quy đổi tọa độ:** Từ tọa độ ảnh sang tọa độ cell:
  - Vị trí cell: `i, j = int(S * y), int(S * x)`
  - Tọa độ tâm trong cell: `x_cell = S * x - j`, `y_cell = S * y - i`
  - Kích thước theo cell: `width_cell = width * S`, `height_cell = height * S`
- **Xử lý Anchor gây nhiễu (`ignore_iou_thresh = 0.5`):**
  - Các anchor có IoU $> 0.5$ với box nhưng không phải anchor tốt nhất sẽ được gán `objectness = -1`.
  - *Mục đích:* Không coi các anchor này là sai (no-object), tránh việc mô hình phạt nhầm dự đoán đúng.
- **Xử lý ảnh:** Bật `LOAD_TRUNCATED_IMAGES = True` để tránh lỗi ảnh bị cụt; chuyển ảnh về RGB; dùng Albumentations để biến đổi đồng thời cả ảnh và bboxes.

---

### 2. Hàm Loss (`loss.py`)
- **Tách nhãn bằng Boolean Mask:**
  - `obj = target[..., 0] == 1`: Lấy cell/anchor có chứa vật thể.
  - `noobj = target[..., 0] == 0`: Lấy cell/anchor không có vật thể (tự động bỏ qua các cell có giá trị `-1`).
- **No-object Loss:** Dùng `BCEWithLogitsLoss` cho các cell/anchor không chứa vật thể (`noobj`).
- **Object Loss:**
  - Target của object loss không phải là 1 cố định, mà là `ious * target[obj]` (IoU giữa box dự đoán và ground-truth).
  - *Mục đích:* Đoán có object phải đi kèm với việc bounding box dự đoán có khớp với vật thể hay không, tránh trường hợp tự tin có vật thể nhưng box lại lệch.
- **Box Coordinate Loss:**
  - Mô hình thực chất dự đoán ra $t_w, t_h$. Kích thước thật là $w = anchor_w \cdot e^{t_w}$.
  - Do đó ground-truth được chuyển đổi ngược: $t_w, t_h = \ln(1e-16 + target / anchors)$.
  - Tâm $x, y$ đi qua hàm Sigmoid để nằm trong khoảng $[0, 1]$.
  - Dùng `MSELoss` để tính sai số giữa dự đoán và nhãn của $(x, y, t_w, t_h)$.
- **Class Loss:** Dùng `CrossEntropyLoss` (predictions có 20 class probabilities, targets là class index).
- **Cân bằng Loss:** Dùng các hệ số `lambda_box = 10`, `lambda_noobj = 10`, `lambda_obj = 1`, `lambda_class = 1` để cân bằng giữa các thành phần loss.

---

### 3. Kiến trúc Mô hình (`model.py`)
- **Cấu hình mạng:**
  - Tuple `(filters, kernel_size, stride)`: Conv layer.
  - List `["B", repeats]`: Khối Residual Block lặp lại.
  - `"S"`: Scale prediction block để đưa ra dự đoán và tính loss.
  - `"U"`: Upsample feature map (tăng resolution x2) và ghép (`torch.cat`) với layer trước đó.
- **CNNBlock:** Nếu có dùng `BatchNorm2d` thì tắt `bias` trong `Conv2d` (`bias=not bn_act`). Dùng activation `LeakyReLU(0.1)`.
- **ResidualBlock:** Giảm channel bằng Conv $1 \times 1$ rồi tăng lại bằng Conv $3 \times 3$; cộng feature cũ với feature mới: `x = x + layer(x)`.
- **ScalePrediction:**
  - Đầu ra có `(num_classes + 5) * 3` kênh cho 3 anchor.
  - Dùng `reshape` và `permute` chuyển từ mảng phẳng thành `[Batch, 3, S, S, num_classes + 5]`.
- **Kết nối Route (FPN):** Lưu feature map từ `ResidualBlock` lặp 8 lần vào `route_connections`, sau khi `Upsample` thì lấy ra nối (`torch.cat`) với feature hiện tại.
- **Kiểm tra kiến trúc:** Dùng `assert` kiểm tra output shape ở 3 scale: `IMAGE_SIZE // 32`, `IMAGE_SIZE // 16`, và `IMAGE_SIZE // 8`.

---

### 4. Huấn luyện Mô hình (`train.py`)
- **Đồng bộ thiết bị:** Toàn bộ tensor đưa vào mô hình (ảnh, nhãn 3 scale, anchors) đều phải đưa về cùng device (`config.DEVICE`).
- **Mixed Precision Training (PyTorch AMP):**
  - Dùng `torch.cuda.amp.autocast()`: Tự động chuyển đổi giữa float16 và float32 để tăng tốc.
  - Dùng `GradScaler()`: Float16 có thể làm gradient quá nhỏ bị coi là 0 (underflow / vanishing gradient), do đó cần scale loss trước khi `backward()` và unscale khi `step()`.
- **Đánh giá trong lúc train:** 
  - Lưu checkpoint (`save_checkpoint`) và tải checkpoint (`load_checkpoint`).
  - Định kỳ kiểm tra độ chính xác theo class (`check_class_accuracy`) và tính chỉ số `mAP` (`mean_average_precision`).

---

### 5. Tiện ích & Đánh giá (`utils.py`, `config.py`)
- **Ngưỡng cấu hình (`config.py`):**
  - `CONF_THRESHOLD = 0.5`: Ngưỡng xác suất để xác nhận trong cell có object hay không.
  - `MAP_IOU_THRESH = 0.5`: Ngưỡng IoU để tính mAP.
  - `NMS_IOU_THRESH = 0.45`: Ngưỡng loại bỏ box trùng (ngưỡng cao dễ trùng nhiều box, ngưỡng thấp ít box trùng hơn).
- **Non-Maximum Suppression (NMS):**
  - Lọc bỏ các box có score $< \text{threshold}$.
  - Sắp xếp giảm dần theo điểm xác suất.
  - Duyệt từng box cao nhất và loại bỏ các box khác cùng class có $\text{IoU} \ge \text{iou\_threshold}$.
- **cells_to_bboxes:** Chuyển đổi tensor dự đoán từ hệ tọa độ cell của từng scale về tọa độ chuẩn hóa toàn ảnh $[x, y, w, h]$ để vẽ hoặc tính mAP.
- **check_class_accuracy:** Đo riêng biệt 3 chỉ số: tỉ lệ đoán đúng class, tỉ lệ đoán đúng cell có object, và tỉ lệ đoán đúng cell không có object (`noobj`).

---

### 6. Chạy Thực Tế (`inference.py`)
- Chuyển mô hình sang `model.eval()` để không tính gradient, giúp tăng tốc độ dự đoán.
- Đọc frame webcam qua OpenCV, resize, chuẩn hóa, đưa qua model, decode bằng `cells_to_bboxes`, lọc qua NMS và vẽ box, label lên màn hình theo thời gian thực.

---

### 7. Đúc kết & Hướng phát triển (`note.md`)
- **Hiện trạng:** Mô hình YOLOv3 còn khá nặng khi chạy trên máy cá nhân (R7 5800H), mới thử nghiệm trên PASCAL_VOC.
- **3 Cấp độ can thiệp khi tối ưu mô hình:**
  1. *Dữ liệu:* Augmentation khác, crop khác, thay đổi image size, cân bằng class (class balancing).
  2. *Tham số cơ bản:* Learning rate, scheduler, batch size, optimizer, weight decay, số epoch, các ngưỡng threshold.
  3. *Kiến trúc:* Thay đổi backbone, freeze/unfreeze backbone, thêm/bớt layer, đổi activation, train from scratch vs pretrained.
- **Phân tích lỗi dự đoán cần theo dõi:** False Positive, False Negative, Wrong class, Bad localization, Small objects, Occlusion, Crowded scene, Low-quality image.
- **Hướng phát triển tiếp theo:** Bài toán Tracking (đếm vật thể, theo dõi xe, đo tốc độ) và Sequence Modeling (phát hiện hành động bất thường, quỹ đạo).

---

## 👨‍💻 Tác giả
* **Nguyễn Tiến Đan**
* Dự án phục vụ mục đích nghiên cứu chuyên sâu về kiến trúc mạng học sâu (Deep Learning) và Thị giác máy tính (Computer Vision).
