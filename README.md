# 🎨 Gesture-Controlled Virtual Paint App

Control your screen like a magic wand! This real-time virtual paint application lets you **draw, erase, and switch colors** using just your hand gestures — powered by **MediaPipe**, **OpenCV**, and **Machine Learning (Random Forest & CNN)**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-RealTime-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-red?logo=google)
![ML Model](https://img.shields.io/badge/Model-RandomForest/CNN-yellow)

---

## 🧠 Features

- ✋ **Hand Gesture Recognition** with MediaPipe
- 🧠 **Dual Model Support**: Toggle between Random Forest and CNN
- 🎨 **Draw, Erase, Clear Canvas**
- 🟥 **Color Options**: Red, Green, Blue
- ⚡ **Smooth Drawing & UI Overlay**
- 🔁 **Real-Time Gesture Prediction**
- 📸 **Live Webcam Support**
- 🎛️ **Intuitive GUI with Gesture-based Controls**

---

## 🖼️ Demo (Add your GIF or image here)

> _Place a GIF or screenshot here showing how the gesture UI works with drawing and color switching._

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenCV**
- **MediaPipe**
- **Scikit-learn**
- **TensorFlow (for CNN model)**
- **NumPy, Pandas, Matplotlib**

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/gesture-virtual-paint.git
cd gesture-virtual-paint
2. Install dependencies
pip install -r requirements.txt
3. Collect gesture data (optional)

python data_collection.py
4. Train a model
For ML (Random Forest):

b
python train_model.py
For CNN:


python train_cnn.py
5. Launch the App


python virtual_paint_ml.py
Choose between ML or CNN when prompted.



📁 Project Structure

gesture-virtual-paint/
├── data_collection.py
├── train_model.py
├── train_cnn.py
├── virtual_paint_ml.py
├── gesture_model.pkl
├── cnn_gesture_model.h5
├── gesture_images/ (if using CNN)
└── README.md
