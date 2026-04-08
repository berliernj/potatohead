# 🎭 Real-Time Face Feature Mapping (Eyes & Mouth Tracking)

This project demonstrates a **real-time face tracking system** that extracts your **eyes and mouth from a live webcam feed** and maps them onto a **moving model face**.

The model dynamically follows your **head movement and rotation**, creating a simple face-driven avatar effect.

---

## 🚀 Features

* 📷 Live webcam capture using OpenCV
* 🧠 Facial landmark detection via MediaPipe (468 landmarks)
* 👁️ Eye and mouth region extraction
* 🎭 Model face rendering (ellipse-based avatar)
* 🔄 Head rotation tracking and alignment
* 🧩 Real-time feature overlay onto the model

---

## 🧠 How It Works

The pipeline:

```
Webcam → Face Landmarks → Head Pose Estimation
                                ↓
                  Feature Extraction (Eyes + Mouth)
                                ↓
                  Model Transformation (Rotation)
                                ↓
                         Feature Overlay
```

### Key Concepts

* **Face Mesh**: Uses MediaPipe to detect detailed facial landmarks
* **Head Pose Approximation**: Uses eye-to-nose vector to estimate tilt
* **Image Patching**: Extracts regions (eyes/mouth) from the real face
* **Affine Transformation**: Rotates the model to match head movement
* **Overlay Rendering**: Places extracted features onto the model
