# 🚗 Vehicle Observer — Vehicle Detection System

A YOLOv12-powered vehicle detection web application built for Capstone Project Module 4 at Purwadhika AI Engineer Bootcamp.

Upload a road or traffic image and the system will detect all vehicles, draw bounding boxes, and count each vehicle class automatically.

---

## 🎯 What it does

- Detects **cars**, **vans**, and **buses** in uploaded images
- Draws bounding boxes with class label and confidence score
- Displays a count summary per vehicle class + total vehicle count

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| Object Detection | YOLOv12 (ultralytics) |
| Annotation | supervision |
| Image Handling | Pillow |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```
VehicleObserver-App/
├── main.py           # Streamlit UI
├── observer.py       # Detection logic (model loader + inference pipeline)
├── best_vehicle2.pt  # Fine-tuned YOLOv12n weights
├── requirements.txt  # Dependencies
└── .gitignore
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Blatopio/Vehicle_Observer.git
cd Vehicle_Observer

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run main.py
```

---

## 📊 Model Details

| Attribute | Value |
|---|---|
| Architecture | YOLOv12n (nano) |
| Input size | 640 x 640 px |
| Classes | car, bus, van |
| Training environment | Google Colab (T4 GPU) |
| Augmentation | Non-geometric only (hue, saturation, brightness) |

---

## 👤 Author

**Muhammad Fachreza Alghifari**  
Purwadhika AI Engineer Bootcamp — Module 4 Capstone Project