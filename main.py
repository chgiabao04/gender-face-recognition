import cv2
import os

from gender_classification import predict_gender
from face_localization import detect_faces
from shape_features import extract_hog 
from geometric_features import draw_landmarks
from emotion_recognition import predict_emotion


# ===============================
# Xử lý 1 frame (image hoặc webcam)
# ===============================

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # (1) Face localization
    faces = detect_faces(gray)

    for (x, y, w, h) in faces:
        # tránh crop lỗi
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            continue

        face = frame[y:y+h, x:x+w]

        # ===============================
        # (2) Shape features (HOG - demo)
        # ===============================
        try:
            hog_img = extract_hog(face)
            cv2.imshow("Shape Features (HOG)", hog_img)
        except:
            pass

        # ===============================
        # (3) Geometric features (landmark)
        # ===============================
        face_with_landmarks = draw_landmarks(face)
        frame[y:y+h, x:x+w] = face_with_landmarks

        # ===============================
        # (4) Emotion recognition (CNN)
        # ===============================
        try:
            emotion, emo_conf = predict_emotion(face)
        except:
            emotion, emo_conf = "Unknown", 0.0

        # ===============================
        # (5) Gender classification (CNN)
        # ===============================
        gender = predict_gender(face)

        # ===============================
        # Vẽ bounding box + label
        # ===============================
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        label = f"{gender} | {emotion}"

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        )

        cv2.rectangle(
            frame,
            (x, y-th-10),
            (x+tw+5, y),
            (255, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x+2, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

    return frame



# ===============================
# MENU
# ===============================

print("\n=== CHỌN CHẾ ĐỘ ===")
print("1. Nhận diện từ ảnh")
print("2. Nhận diện từ webcam")
mode = input("Nhập lựa chọn (1/2): ").strip()


# ===============================
# MODE 1: IMAGE
# ===============================

if mode == "1":
    image_path = input("Nhập đường dẫn ảnh (Enter = test.jpg): ").strip()
    if not image_path:
        image_path = "test.jpg"

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        exit()

    result = process_frame(img)

    # Resize ảnh cho vừa màn hình
    h, w = result.shape[:2]
    max_height = 800
    if h > max_height:
        ratio = max_height / h
        result = cv2.resize(
            result,
            (int(w * ratio), max_height),
            interpolation=cv2.INTER_LANCZOS4
        )

    cv2.imshow("Gender Recognition - Image", result)

    print("\n📌 Nhấn 's' để lưu ảnh kết quả")
    print("📌 Nhấn phím bất kỳ để thoát...")

    key = cv2.waitKey(0)

    if key == ord('s'):
        output_path = "result_" + os.path.basename(image_path)
        cv2.imwrite(output_path, result)
        print(f"✅ Đã lưu ảnh: {output_path}")

    cv2.destroyAllWindows()


# ===============================
# MODE 2: WEBCAM
# ===============================

elif mode == "2":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Không mở được webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame)
        cv2.imshow("Gender Recognition - Webcam", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


else:
    print("❌ Lựa chọn không hợp lệ!")
