import cv2
import os

# ===============================
# Load Haar Cascade
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

HAAR_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(HAAR_PATH)

if face_cascade.empty():
    raise RuntimeError("❌ Cannot load Haar Cascade model")


# ===============================
# Face localization
# ===============================

def detect_faces(gray_img):
    """
    Detect faces in grayscale image.
    Output: list of (x, y, w, h)
    """
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces
