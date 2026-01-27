import cv2
import numpy as np

try:
    from mediapipe.python.solutions import face_mesh as fm
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe không khả dụng - bỏ qua landmark detection")


def draw_landmarks(face_img):
    """
    Vẽ các điểm đặc trưng hình học (facial landmarks) lên khuôn mặt
    
    Landmarks đại diện cho các đặc điểm hình học của khuôn mặt như:
    - Vị trí mắt, mũi, miệng, hàm
    - Các điểm này có thể dùng để tính khoảng cách, tỷ lệ giữa các bộ phận khuôn mặt
    - Ứng dụng: phân tích cảm xúc, đo độ đối xứng khuôn mặt, nhận dạng sinh trắc học
    """
    if not MEDIAPIPE_AVAILABLE:
        return face_img
    
    try:
        face_mesh = fm.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        h, w, _ = face_img.shape
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            face_mesh.close()
            return face_img

        for lm in result.multi_face_landmarks[0].landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(face_img, (x, y), 1, (0, 255, 0), -1)
        
        face_mesh.close()
        
    except Exception as e:
        print(f"⚠️ Landmark error: {e}")

    return face_img