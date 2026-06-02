# 🧠 Brain Tumor Classification

> **Deep Learning** based brain tumor classification from MRI images using **VGG16 CNN**.
> Final project untuk thesis S1 di Universitas Sriwijaya — Teknik Informatika (IPK 3.55).

![Tech](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.x-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Accuracy](https://img.shields.io/badge/Accuracy-95%2B%25-brightgreen)

## ✨ Features

- 🎯 **95%+ accuracy** dengan VGG16 transfer learning
- ⚡ **< 3 detik** waktu prediksi per gambar
- 🎨 **Modern UI** — dark theme dengan cyan accent, drag & drop upload
- 📱 **Responsive** — bekerja di desktop & mobile
- 🔒 **Privacy** — gambar upload tidak disimpan permanen

## 🚀 Quick Start

```bash
# Setup
python -m venv venv
source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run
python app.py
# → buka http://localhost:8000
```

## 📂 Project Structure

```
brain-tumor-classification/
├── app.py                    # Flask backend + model loader
├── helper.py                 # Image preprocessing
├── models/
│   └── vgg16_model_scen2confv1.h5    # Trained VGG16 weights
├── templates/
│   ├── index.html            # Landing page (modern dark theme)
│   └── predict.html          # Upload + classify page (drag & drop)
└── static/
    └── img_uploaded/         # Temporary upload storage (gitignored)
```

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **ML:** TensorFlow / Keras, VGG16 (transfer learning)
- **Frontend:** HTML5, CSS3 (custom dark theme), Vanilla JS
- **Image processing:** Pillow, OpenCV

## 📊 Model Details

- **Architecture:** VGG16 (ImageNet pretrained) + custom dense layers
- **Training:** Transfer learning + fine-tuning
- **Classes:** Tumor / No Tumor
- **Dataset:** Brain MRI Images for Brain Tumor Detection (Kaggle)
- **Research paper:** "Improving the performance for automated brain tumor classification on magnetic resonance imaging deep learning-based"

## 🔄 Recent Updates (Juni 2026)

- ✨ **Total redesign** UI dengan modern dark theme + cyan accent
- ✨ **Drag & drop upload** dengan file preview & validation
- ✨ **Hero landing page** dengan stats cards & features grid
- ✨ Color-coded result badge (red = tumor, green = no tumor)

## 👤 Author

**Jerrel Adriel A Hutahaean**
Fullstack Developer · Jakarta, Indonesia

- 🌐 Portfolio: [jerreladriel.github.io](https://jerreladriel.github.io)
- 💼 LinkedIn: [jerrelhutahaean](https://www.linkedin.com/in/jerrelhutahaean)
- ✉️ Email: jerreladriel@gmail.com

---

📚 **Full documentation** untuk semua project saya tersedia di [Portfolio Documentation](https://github.com/JerrelAdriel/JerrelAdriel.github.io/blob/main/DOCUMENTATION.md).
