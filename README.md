# FinDocAI – Financial Document Tampering Detection

AI-powered financial document analysis system that detects suspicious modifications in receipts and invoices using deep learning.

## 📌 Overview

FinDocAI is a web-based application designed to identify potentially tampered financial documents such as receipts and invoices.

The system uses **EfficientNetV2B0** to analyze document images and **Grad-CAM** to provide a visual explanation of the regions that contributed to the prediction.

Users can upload a document or capture an image using their device camera and receive an AI-based analysis.

---

## ✨ Features

- 📄 Upload receipt or invoice images
- 📷 Capture documents using a camera
- 🤖 AI-based tampering detection
- 🧠 EfficientNetV2B0 deep-learning model
- 🔍 Document region / tile analysis
- 🌡️ Grad-CAM visualization
- 📊 Prediction confidence
- 🖥️ Interactive Flask web interface
- 📱 Responsive web design
- 📑 Separate pages for detection, methodology, performance and project information

---

## 🧠 Technologies Used

### Frontend

- HTML5
- CSS3
- JavaScript

### Backend

- Python
- Flask

### Machine Learning

- TensorFlow
- Keras
- EfficientNetV2B0
- Grad-CAM
- NumPy

### Image Processing

- Pillow (PIL)
- Matplotlib

---

## 🏗️ System Architecture

```text
                ┌─────────────────────┐
                │       User          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Upload / Camera     │
                │ Document Image      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Image Preprocessing │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Document Tiling     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ EfficientNetV2B0    │
                │ Classification      │
                └──────────┬──────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
        ┌─────────────────┐  ┌─────────────────┐
        │ Prediction      │  │ Grad-CAM        │
        │ Probability     │  │ Explanation     │
        └────────┬────────┘  └────────┬────────┘
                 │                    │
                 └──────────┬─────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Final Result        │
                 │ Genuine / Tampered  │
                 └─────────────────────┘
