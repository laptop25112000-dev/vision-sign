---
title: Visual Voice — ISL Translator
emoji: 🤟
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.34.2"
app_file: app.py
pinned: true
license: mit
short_description: Indian Sign Language A-Z alphabet recognition
---

# 🤟 Visual Voice — ISL Translator

Indian Sign Language A-Z alphabet recognition using MediaPipe hand landmarks and an SVM classifier.

**Made by Divyansh** | Made under Navin Sir's guidance

## How it works
1. Webcam captures your hand
2. MediaPipe extracts 21 hand landmarks (x, y, z)
3. Landmarks are normalized into a 128-feature vector
4. SVM classifier predicts the alphabet letter (A-Z)

## Model Details
- **Accuracy:** 99.8% (offline test set)
- **Features:** 128 normalized landmark coordinates
- **Training Samples:** 16,016 balanced samples
- **Classes:** 26 (A-Z)
