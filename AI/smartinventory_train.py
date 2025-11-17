import os
import random
import shutil
from ultralytics import YOLO

# ==============================
# 1️⃣ HÀM CHIA DỮ LIỆU TRAIN / VAL
# ==============================
def split_dataset(base_dir="dataset", train_ratio=0.8):
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    # Tạo thư mục đầu vào
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for subdir in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, subdir), exist_ok=True)

    def move_files(file_list, subset):
        for img_file in file_list:
            base_name = os.path.splitext(img_file)[0]
            img_src = os.path.join(images_dir, img_file)
            label_src = os.path.join(labels_dir, base_name + ".txt")

            img_dst = os.path.join(images_dir, subset, img_file)
            label_dst = os.path.join(labels_dir, subset, base_name + ".txt")

            if os.path.exists(img_src):
                shutil.move(img_src, img_dst)
            if os.path.exists(label_src):
                shutil.move(label_src, label_dst)

    move_files(train_files, "train")
    move_files(val_files, "val")

    print(f"✅ Đã chia dữ liệu: {len(train_files)} train / {len(val_files)} val")


# ==============================
# 2️⃣ HUẤN LUYỆN YOLOv8
# ==============================
def train_yolov8():
    print("🚀 Bắt đầu huấn luyện YOLOv8...")

    # File cấu hình data.yaml (bạn có thể chỉnh sửa thêm nếu cần)
    data_yaml_content = """train: dataset/images/train
val: dataset/images/val

nc: 7
names:
  [
    'mi_tom_hao_hao',
    'snack_bi_do',
    'snack_bap_ngot',
    'dau_nanh_tuong_an',
    'bia_333',
    'dau_goi_nguyen_xuan',
    'kem_danh_rang_ps'
  ]
"""
    with open("data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    model = YOLO("yolov8n.pt")  # mô hình nền nhỏ (fast)

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        name="smartinventory_vn_v1",
        device='cpu'  # hoặc 'cpu'
    )

    print("🎯 Huấn luyện hoàn tất! Kiểm tra thư mục runs/detect/smartinventory_vn_v1/weights/")


# ==============================
# 3️⃣ CHẠY TOÀN BỘ QUY TRÌNH
# ==============================
if __name__ == "__main__":
    print("🧠 SMART INVENTORY AI - Huấn luyện YOLOv8 tự động")
    split_dataset(base_dir="dataset", train_ratio=0.8)
    train_yolov8()