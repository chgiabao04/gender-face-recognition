import cv2
import numpy as np
import os

# ===============================
# Load model
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")

gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

if gender_net.empty():
    raise RuntimeError("Cannot load gender model")

GENDER_LIST = ["Male", "Female"]


# ===============================
# NMS
# ===============================

def self_nms(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = boxes[:, 2] * boxes[:, 3]
    idxs = np.argsort(area)[::-1]

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / area[idxs[1:]]

        idxs = np.delete(
            idxs,
            np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1))
        )

    return boxes[pick].tolist()


# ===============================
# CORE FUNCTION
# ===============================

def predict_gender(face_img):
    """
    Predict gender from cropped face image.
    """
    face = cv2.resize(face_img, (227, 227))

    blob = cv2.dnn.blobFromImage(
        face,
        1.0,
        (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )

    gender_net.setInput(blob)
    preds = gender_net.forward()

    confidence = preds[0].max()
    return GENDER_LIST[preds[0].argmax()] if confidence > 0.6 else "Unknown"
