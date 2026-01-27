import cv2
import numpy as np
from skimage.feature import hog

def extract_hog(face_img):
    """
    Appearance-based feature extraction using HOG
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )

    hog_image = (hog_image * 255).astype("uint8")
    hog_image = cv2.cvtColor(hog_image, cv2.COLOR_GRAY2BGR)

    return hog_image
