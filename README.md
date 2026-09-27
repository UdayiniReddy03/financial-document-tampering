# FinDocAI – Financial Document Tampering Detection

AI-powered web application for detecting potential tampering in financial documents such as receipts and invoices using **EfficientNetV2B0** and **Grad-CAM**.

## 🚀 About the Project

FinDocAI analyzes financial document images and identifies regions that may contain suspicious modifications.

The application uses a trained **EfficientNetV2B0** deep-learning model for image classification and **Grad-CAM** to visualize the regions that contributed to the prediction.

Users can upload a document or capture an image using a camera and view the analysis through a web interface.

## ✨ Features

- 📄 Upload receipt and invoice images
- 📷 Capture documents using a camera
- 🤖 AI-based tampering detection
- 🧠 EfficientNetV2B0 deep-learning model
- 🔍 Document region analysis
- 🌡️ Grad-CAM visualization
- 📊 Prediction confidence
- 🖥️ Flask-based web application
- 📱 Responsive HTML & CSS interface
- 📑 Multiple application pages

## 🛠️ Tech Stack

**Frontend**
- HTML5
- CSS3
- JavaScript

**Backend**
- Python
- Flask

**Machine Learning**
- TensorFlow
- Keras
- EfficientNetV2B0
- Grad-CAM
- NumPy

**Image Processing**
- Pillow
- Matplotlib

## 🧠 How It Works

```text
Upload / Capture Document
          ↓
    Image Preprocessing
          ↓
      Document Tiling
          ↓
     EfficientNetV2B0
          ↓
   Tampering Prediction
          ↓
      Grad-CAM
          ↓
 Suspicious Region + Result
