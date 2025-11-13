import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import json

DATASET_DIR = 'dataset'  # thư mục chứa ảnh đã chụp
IMG_SIZE = (64, 64)      # Kích thước chuẩn cho CNN
MODEL_DIR = 'trainer'
MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_face_model.h5')
LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'label_map.json')

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Đọc ảnh và nhãn ---
images = []
labels = []
label_to_id = {}  # label string -> số
current_id = 0

for root, dirs, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith('.jpg'):
            path = os.path.join(root, f)
            # Chuyển ảnh sang grayscale, resize
            img = Image.open(path).convert('L').resize(IMG_SIZE)
            img_array = np.array(img, dtype='float32') / 255.0
            images.append(img_array)

            # Lấy lớp + ID sinh viên
            folder = os.path.basename(root)
            filename = os.path.basename(path)
            try:
                student_id = filename.split('.')[1]  # User.<id>.<count>.jpg
            except:
                print(f"Filename {filename} không đúng định dạng!")
                continue
            label_str = f"{folder}-{student_id}"

            if label_str not in label_to_id:
                label_to_id[label_str] = current_id
                current_id += 1
            labels.append(label_to_id[label_str])

images = np.expand_dims(np.array(images), axis=-1)
labels = np.array(labels)
NUM_CLASSES = len(label_to_id)

print(f"🔹 Tổng số sinh viên (class) trong dataset: {NUM_CLASSES}")
print(f"🔹 Tổng số ảnh: {len(images)}")

# --- Xây dựng CNN ---
model = models.Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- Huấn luyện CNN ---
EPOCHS = 5
history = model.fit(images, labels, epochs=EPOCHS)

# --- Lưu model và label map ---
model.save(MODEL_PATH)
with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump(label_to_id, f, ensure_ascii=False, indent=4)

print(f"\n✅ Hoàn tất train model với {NUM_CLASSES} sinh viên.")
print(f"📁 Model lưu tại: {MODEL_PATH}")
print(f"📁 Label map lưu tại: {LABEL_MAP_PATH}")
