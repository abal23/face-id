import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json
import os
from datetime import datetime

# --- Cấu hình ---
CONFIG_FILE = "deploy.prototxt.txt"
MODEL_FILE  = "res10_300x300_ssd_iter_140000.caffemodel"
CNN_MODEL_PATH = 'trainer/cnn_face_model.h5'
IMG_SIZE = (64, 64)
INFO_FILE = 'thong tin cac lop.txt'

# --- Load model nhận diện khuôn mặt OpenCV ---
net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)

# --- Load CNN model ---
cnn_model = load_model(CNN_MODEL_PATH)

# --- Load thông tin lớp học ---
if os.path.exists(INFO_FILE):
    with open(INFO_FILE, 'r', encoding='utf-8') as f:
        classes_info = json.load(f)
else:
    print("⚠️ Không tìm thấy file thông tin lớp học!")
    classes_info = {}

# --- Tạo mapping ID chung cho CNN ---
id_to_label = {}
label_to_name = {}
idx = 0
for lop, students in classes_info.items():
    for sid, name in students.items():
        id_to_label[idx] = f"{lop}-{sid}"
        label_to_name[f"{lop}-{sid}"] = name
        idx += 1
NUM_CLASSES = idx

# --- Chọn lớp cần điểm danh ---
print("Các lớp có sẵn:")
for lop in classes_info.keys():
    print("-", lop)

lop_chon = input("\nNhập tên lớp cần điểm danh: ").strip()
if lop_chon not in classes_info:
    print("❌ Lớp không tồn tại!")
    exit()

attendance = {sid: False for sid in classes_info[lop_chon]}

# --- Mở webcam ---
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX

print(f"\n📸 Bắt đầu điểm danh lớp {lop_chon} (Nhấn ESC để kết thúc)\n")

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.85:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)).resize(IMG_SIZE)
            face_array = np.array(face_img, 'float32') / 255.0
            face_array = np.expand_dims(face_array, axis=(0, -1))

            preds = cnn_model.predict(face_array)
            idx_pred = np.argmax(preds)
            conf_cnn = preds[0][idx_pred] * 100

            label = id_to_label[idx_pred]        # Lấy nhãn dạng "Lop-SID"
            lop_id, sid = label.split('-')
            name = label_to_name[label]

            # Chỉ đánh dấu attendance nếu thuộc lớp đang điểm danh
            if lop_id == lop_chon and conf_cnn > 50:
                attendance[sid] = True

            # Vẽ khung và tên + độ chính xác
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}", (x1+5, y1-5), font, 1, (255,255,255), 2)
            cv2.putText(frame, f"Acc: {round(conf_cnn)}%", (x1+5, y2+25), font, 0.7, (255,255,0), 1)

    # Hiển thị số người có mặt trong lớp
    cv2.putText(frame, f"Đã điểm danh: {sum(attendance.values())}/{len(attendance)}", (20,40),
                font, 0.8, (0,255,255), 2)

    cv2.imshow(f"Điểm danh lớp {lop_chon}", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cam.release()
cv2.destroyAllWindows()

# --- Tổng kết ---
present = [classes_info[lop_chon][sid] for sid, checked in attendance.items() if checked]
absent = [classes_info[lop_chon][sid] for sid, checked in attendance.items() if not checked]

print(f"\n✅ Lớp {lop_chon}: {len(present)}/{len(classes_info[lop_chon])} sinh viên có mặt.")
if absent:
    print("❌ Vắng mặt:", ", ".join(absent))

# --- Lưu kết quả ---
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f"diem_danh_{lop_chon}_{date_str}.txt", "w", encoding='utf-8') as f:
    f.write(f"Điểm danh lớp {lop_chon} - {date_str}\n")
    f.write(f"Có mặt ({len(present)}): {', '.join(present)}\n")
    f.write(f"Vắng mặt ({len(absent)}): {', '.join(absent)}\n")

print("\n📁 Kết quả đã được lưu.")
