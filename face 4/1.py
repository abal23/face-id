




















import cv2
import os
import json

# --- Nạp mô hình deep learning của OpenCV ---
configFile = "deploy.prototxt.txt"
modelFile  = "res10_300x300_ssd_iter_140000.caffemodel"
DATASET_DIR = 'dataset'
INFO_FILE = 'thong tin cac lop.txt'

# Tạo thư mục dataset nếu chưa có
os.makedirs(DATASET_DIR, exist_ok=True)

# Nạp model
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# --- Hỏi thông tin sinh viên ---
lop = input("Nhập tên lớp học: ").strip()
face_id = input("Nhập ID sinh viên: ").strip()
name = input("Nhập tên sinh viên: ").strip()

# --- Đảm bảo lớp tồn tại ---
lop_dir = os.path.join(DATASET_DIR, lop)
os.makedirs(lop_dir, exist_ok=True)

# --- Cập nhật file JSON ---
if os.path.exists(INFO_FILE):
    try:
        with open(INFO_FILE, "r", encoding="utf-8") as f:
            classes_info = json.load(f)
    except json.JSONDecodeError:
        classes_info = {}
else:
    classes_info = {}

if lop not in classes_info:
    classes_info[lop] = {}

classes_info[lop][face_id] = name

# --- Ghi lại file ---
with open(INFO_FILE, "w", encoding="utf-8") as f:
    json.dump(classes_info, f, ensure_ascii=False, indent=4)

print(f"✅ Đã cập nhật sinh viên {name} (ID: {face_id}) vào lớp {lop}")

# --- Bắt đầu thu thập ảnh ---
count = 0
MAX_COUNT = 50
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("📷 Bắt đầu chụp ảnh khuôn mặt... (Nhấn 'q' để thoát)")

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
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            face = frame[y1:y2, x1:x2]
            if face.size > 0 and count < MAX_COUNT:
                count += 1
                save_path = os.path.join(lop_dir, f"User.{face_id}.{count}.jpg")
                cv2.imwrite(save_path, face)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Saved: {count}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= MAX_COUNT:
        break

cam.release()
cv2.destroyAllWindows()
print(f"📁 Đã lưu {count} ảnh vào {lop_dir}")
