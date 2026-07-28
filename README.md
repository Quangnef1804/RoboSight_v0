# RoboSight

Hướng dẫn fine-tune RF-DETR: [docs/RFDETR_TRAINING.md](docs/RFDETR_TRAINING.md).

Đánh giá riêng trên Object_2: [docs/OBJECT2_EXTERNAL_TEST.md](docs/OBJECT2_EXTERNAL_TEST.md).

```powershell
# Tải xuống 2 model RF-DETR và SAM 3 vào thư mục third-party/:
cd third-party/
git clone https://github.com/roboflow/rf-detr RF-DETR
git clone https://github.com/facebookresearch/sam3.git SAM3
cd ..

# Cài dependency của RF-DETR và SAM3
pip install -r requirements.txt

# Cấu hình checkpoint, ảnh đầu vào và output trước khi chạy
notepad configs/sam3.yaml

# Class lấy từ configs/dataset.yaml; đặt ảnh vào data/sam3_trial_v1/images

# Nếu đặt checkpoint: null, đăng nhập Hugging Face trước khi tải pretrained
hf auth login

# Tạo mask proposal tạm thời bằng SAM3 pretrained
python -m src.annotate propose --config configs/sam3.yaml

# Kiểm duyệt accept/edit/reject/missed và xác nhận class
python -m src.annotate review --config configs/sam3.yaml

# Trong cửa sổ: 1-5 chọn class; A/E/R xử lý proposal; M thêm box; D hoàn tất; Q lưu/thoát

# Chỉ xuất annotation đã kiểm duyệt sang dataset mới
python -m src.annotate export --config configs/sam3.yaml

# Dataset chỉ được dùng khi validator báo PASS
python -m src.check_dataset --dataset data/sam3_trial_v1

# Chạy RF-DETR realtime; nhấn Q để dừng và lưu benchmark
python -m src.realtime --config configs/realtime.yaml

# So sánh 720p/1080p hoặc confidence bằng cách sửa configs/realtime.yaml rồi chạy lại
```
