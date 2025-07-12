
🎨 Gesture-Controlled Virtual Paint App

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

## 🖼️ Demo
> 

https://github.com/user-attachments/assets/6cb637e0-bf50-4df6-885a-ebe53698c997





---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenCV**
- **MediaPipe**
- **Scikit-learn**
- **TensorFlow (for CNN model)**
- **NumPy, Pandas, Matplotlib**

---
## 📁 Project Structure

```bash
gesture-virtual-paint/
├── data_collection.py         # Collects gesture data using webcam
├── train_model.py             # Trains Random Forest classifier
├── train_cnn.py               # Trains CNN model on gesture images
├── virtual_paint_ml.py        # Main application (choose ML or CNN at runtime)
├── gesture_model.pkl          # Saved Random Forest model
├── cnn_gesture_model.h5       # Saved CNN model
├── gesture_images/            # Folder of gesture images (used for CNN training)
└── README.md                  # Project documentation
```
---
## 🚀 Setup Instructions
### 1. Clone the repository
```bash 
git clone https://github.com/yourusername/gesture-virtual-paint.git
```
```bash
cd gesture-virtual-paint
```

### 2. Install dependencies
```bash
 pip install -r requirements.txt
```

### 3. Collect gesture data (optional)
```bash
python data_collection.py
```
### 4. Train a model
For ML (Random Forest):
```bash
python train_model.py
```
For CNN:
```bash
python train_cnn.py
```

### 5. Launch the App
```bash
python virtual_paint_ml.py
```
Choose between ML or CNN when prompted.

