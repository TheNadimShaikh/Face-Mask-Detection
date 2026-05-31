# Face-Mask-Detection
Real-time face mask detection using Random Forest and OpenCV


# 🎭 Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-RandomForest-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-88.5%25-brightgreen)

A real-time face mask detection system built using **Random Forest Classifier** 
and **OpenCV**, developed during my Data Science internship at 
**Softcrowd Technologies, Nashik**.

---

## 📌 Project Overview

This system detects whether a person is wearing a face mask or not 
using a webcam in real-time. It was trained on approximately 
**20,000 images** and achieves an accuracy of **~88.5%**.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| OpenCV | Image processing & webcam feed |
| Scikit-learn | Random Forest model training |
| NumPy 1.26.4 | Numerical computations |
| Pandas | Data handling |
| Flask | Web framework (future deployment) |
| Jupyter Notebook | Development & testing |

---

## 📁 Project Structure
Face-Mask-Detection/
│
├── train_model.py          # Model training script
├── detect_mask_webcam.py   # Real-time webcam detection
├── analyze_results.py      # Model evaluation & analysis
├── detection_log.csv       # Detection log data
├── model_accuracy.png      # Accuracy comparison chart
└── README.md

--------

## ⚙️ How It Works

1. **Data Collection** — ~20,000 images (with mask / without mask)
2. **Preprocessing** — Image resizing, grayscale conversion using OpenCV
3. **Feature Extraction** — HOG features extracted from images
4. **Model Training** — Random Forest Classifier trained using Scikit-learn
5. **Real-time Detection** — Webcam feed processed frame by frame
6. **Result Display** — Bounding box with "Mask" or "No Mask" label

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~88.5% |
| Classifier | Random Forest |
| Dataset Size | ~20,000 images |

---

## 🚀 How to Run

**1. Install dependencies:**
```bash
pip install opencv-python scikit-learn numpy==1.26.4 pandas
```

**2. Train the model:**
```bash
py -3.11 train_model.py
```

**3. Run real-time detection:**
```bash
py -3.11 detect_mask_webcam.py
```

---

## 👨‍💻 Developer

**Nadim Shaikh**  
Computer Engineering — 3rd Year  
Savitribai Phule Pune University (SPPU)  
Internship at Softcrowd Technologies, Nashik  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/nadeem-shaikh-926912346)
[![GitHub](https://img.shields.io/badge/GitHub-TheNadimShaikh-black)](https://github.com/TheNadimShaikh)
