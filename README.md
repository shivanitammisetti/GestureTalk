# GestureTalk — Real-Time Sign Language to Text System

## Overview

GestureTalk converts hand gestures (ASL letters) into text in real time using computer vision and machine learning.
It supports live prediction, sentence formation, and backend API integration for UI applications.

## Features

* Real-time hand gesture detection (MediaPipe)
* ASL letter prediction using ML model
* Sentence formation
* Word suggestions
* Text-to-speech output
* FastAPI backend for UI integration
* Modular production-style architecture

## Project Structure

```
GestureTalk/
│
├── src/
│   ├── api.py          # FastAPI backend
│   ├── live_camera.py  # Camera + prediction UI
│   ├── predict.py      # Model inference logic
│   ├── trainModel.py
│   └── dataCollection.py
│
├── models/
├── data/
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone repo

```
git clone https://github.com/YOUR_USERNAME/GestureTalk.git
cd GestureTalk
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

## Usage

### Run live gesture recognition

```
python -m src.live_camera
```

### Run backend API

```
uvicorn src.api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

## Tech Stack

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* FastAPI
* NumPy / Pandas

## Future Improvements

* Full ASL word recognition
* Language model based sentence correction
* Web UI integration
* Cloud deployment

## Author

Shivani T
