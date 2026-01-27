# Face Recognition System with Gender & Emotion Detection

Hệ thống nhận diện khuôn mặt sử dụng OpenCV, MediaPipe và Deep Learning với khả năng phát hiện giới tính và cảm xúc.

## 📋 Mô tả

Ứng dụng AI phân tích khuôn mặt với các tính năng:

**Tính năng chính:**
- ✅ **Face Localization** - Phát hiện vị trí khuôn mặt (Haar Cascade)
- ✅ **Shape Features** - Trích xuất đặc trưng hình dạng (HOG)
- ✅ **Geometric Features** - Trích xuất 468 điểm đặc trưng hình học (MediaPipe Face Mesh)
- ✅ **Gender Classification** - Nhận diện giới tính (CNN - Caffe Model)
- ✅ **Emotion Recognition** - Nhận diện 7 cảm xúc cơ bản (CNN - FER-2013)
- ✅ Hỗ trợ cả ảnh tĩnh và webcam realtime
- ✅ Lưu kết quả phân tích

## 🎯 Phương pháp sử dụng

### 1. Phát hiện vị trí khuôn mặt (Face Localization)
- **Phương pháp:** Haar Cascade Classifier
- **File:** `face_localization.py`
- **Mô tả:** Phát hiện bounding box (x, y, w, h) của khuôn mặt trong ảnh

### 2. Trích xuất đặc trưng hình dạng (Shape Features)
- **Phương pháp:** HOG (Histogram of Oriented Gradients)
- **File:** `shape_features.py`
- **Mô tả:** Phân tích gradient và hình dạng cục bộ của khuôn mặt
- **Ứng dụng:** Nhận dạng khuôn mặt, phát hiện vật thể

### 3. Trích xuất đặc trưng hình học (Geometric Features)
- **Phương pháp:** MediaPipe Face Mesh
- **File:** `geometric_features.py`
- **Mô tả:** Phát hiện 468 điểm landmark trên khuôn mặt (mắt, mũi, miệng, hàm)
- **Ứng dụng:** 
  - Tính khoảng cách và tỷ lệ giữa các bộ phận khuôn mặt
  - Phân tích cảm xúc
  - Đo độ đối xứng khuôn mặt
  - Nhận dạng sinh trắc học

### 4. Phân loại giới tính (Gender Classification)
- **Phương pháp:** CNN (Convolutional Neural Network)
- **Model:** Caffe Model
- **File:** `gender_classification.py`
- **Output:** Male (Nam) / Female (Nữ)

### 5. Nhận dạng cảm xúc (Emotion Recognition)
- **Phương pháp:** CNN (Mini-Xception Architecture)
- **Dataset:** FER-2013 (35,887 ảnh khuôn mặt)
- **File:** `emotion_recognition.py`
- **Output:** 7 cảm xúc cơ bản
  - 😠 Angry (Tức giận)
  - 🤢 Disgust (Ghê tởm)
  - 😨 Fear (Sợ hãi)
  - 😊 Happy (Vui vẻ)
  - 😢 Sad (Buồn bã)
  - 😲 Surprise (Ngạc nhiên)
  - 😐 Neutral (Bình thường)

## 🛠️ Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.11+
- Webcam (cho chế độ realtime)
- RAM: >= 4GB
- GPU: Không bắt buộc (CPU đủ nhanh)

### 2. Cài đặt thư viện

```bash
pip install opencv-python numpy tensorflow mediapipe
```

### 3. Tải models

**Tạo thư mục `models/` và tải các file sau:**

#### Gender Model (Caffe)
1. **gender_deploy.prototxt** - [Download](https://github.com/smahesh29/Gender-and-Age-Detection/blob/master/gender_deploy.prototxt)
2. **gender_net.caffemodel** - [Download](https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel)

#### Emotion Model (TensorFlow/Keras)
3. **fer2013_mini_XCEPTION.102-0.66.hdf5** - [Download](https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5)

#### Face Detection Model (Haar Cascade)
4. **haarcascade_frontalface_default.xml** - Tự động có sẵn trong OpenCV

**Cấu trúc thư mục:**
```
face_recognition/
├── main.py
├── face_localization.py
├── shape_features.py
├── geometric_features.py
├── gender_classification.py
├── emotion_recognition.py
├── models/
│   ├── haarcascade_frontalface_default.xml
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   └── fer2013_mini_XCEPTION.102-0.66.hdf5
├── test.jpg (ảnh test, không bắt buộc)
└── README.md
```

## 🚀 Sử dụng

### Chạy chương trình

```bash
python main.py
```

### Menu lựa chọn

```
=== CHỌN CHẾ ĐỘ ===
1. Nhận diện từ ảnh
2. Nhận diện từ webcam
```

### Chế độ 1: Nhận diện từ ảnh

```bash
Nhập đường dẫn ảnh (Enter = test.jpg): path/to/image.jpg
```

**Phím tắt:**
- `s` - Lưu ảnh kết quả
- Phím bất kỳ - Thoát

### Chế độ 2: Nhận diện từ webcam

**Phím tắt:**
- `q` - Thoát webcam

## 📊 Kết quả

### Hiển thị trên ảnh:
- **Bounding box** màu xanh dương quanh khuôn mặt
- **Label** hiển thị: `Gender | Emotion`
  - Ví dụ: `Male | Happy`, `Female | Neutral`
- **Facial landmarks** (468 điểm màu xanh lá)
- **HOG features** (cửa sổ riêng)

### Ví dụ output:
```
Male | Happy
Female | Sad
Male | Neutral
```