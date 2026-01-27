import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# nhãn FER2013
EMOTION_LABELS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# load model emotion CNN
emotion_model = keras.models.load_model(
    "models/fer2013_mini_XCEPTION.102-0.66.hdf5",
    compile=False
)

def predict_emotion(face_img):
    """
    Nhận dạng cảm xúc từ khuôn mặt bằng CNN
    
    Phương pháp:
    - Sử dụng Mini-Xception CNN model
    - Dataset: FER-2013 (35,887 ảnh khuôn mặt)
    - Input: 64x64 grayscale image
    - Output: 7 cảm xúc cơ bản
    
    Args:
        face_img: Ảnh khuôn mặt đã crop (BGR format)
    
    Returns:
        emotion: Tên cảm xúc (string)
        confidence: Độ tin cậy (0.0-1.0)
    """
    # chuyển sang grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # resize đúng chuẩn Mini-Xception
    gray = cv2.resize(gray, (64, 64))

    # normalize
    gray = gray.astype("float32") / 255.0

    # reshape: (1, 64, 64, 1)
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)

    # predict
    preds = emotion_model.predict(gray, verbose=0)
    idx = np.argmax(preds)

    emotion = EMOTION_LABELS[idx]
    confidence = preds[0][idx]

    return emotion, confidence