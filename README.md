# ✋ Hand Tracking & Gesture Control System (Python)

A real-time **computer vision–driven hand tracking and gesture interaction system** built using **OpenCV** and **MediaPipe**.  
This project explores how natural hand movements can be translated into intuitive, touch-free controls for system interaction and interactive visual demos using nothing more than a standard webcam.

The repository includes multiple demos ranging from low-level hand landmark visualization to practical system controls and experimental spatial interaction concepts.

---

## 📌 Project Overview

Modern user interfaces are moving beyond keyboards and touchscreens toward **natural, gesture-based interaction**.  
This project is a hands-on exploration of that shift, focusing on:

- Real-time hand detection and tracking
- Gesture recognition through geometric relationships
- Mapping gestures to meaningful system-level actions
- Building a foundation for future AR/VR and spatial computing applications

All processing runs locally and in real time, making the system lightweight, responsive, and easy to extend.

---

## 🚀 Features

### 🖐️ Real-Time Hand Landmark Tracking
- Detects and tracks hands using **MediaPipe Hands**
- Renders 21 hand landmarks and skeletal connections
- Smooth and low-latency performance
- Works with a regular webcam (no depth sensors required)

### 🔊 Gesture-Based Volume Control (Windows Only)
- Control system volume using **pinch gestures**
- Measures the distance between the **thumb tip and index finger tip**
- Maps pinch distance linearly to system speaker volume
- Uses **Pycaw** to interface directly with the Windows audio subsystem
- Provides visual feedback for gesture intensity

### 🧊 Hand Blocks 3D (Experimental Demo)
- Interactive pseudo-3D environment driven by hand gestures
- Pinch to:
  - Spawn wireframe cubes
  - Grab and move objects in camera space
- Smooth object tracking with basic depth cues
- Designed as an experimental step toward:
  - Spatial UIs
  - Gesture-controlled 3D scenes
  - AR-style interactions

---

## 🛠️ Tech Stack

| Category | Tools |
|-------|------|
| Language | Python |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Math & Utilities | NumPy |
| System Audio | Pycaw (Windows) |

---

## Usage

Run the desired module depending on the demo you want to explore:

1️⃣ Hand Tracking Visualizer

Visualizes hand landmarks and finger connections in real time.

python hand_tracking.py

2️⃣ Gesture-Based Volume Control (Windows)

Control your system volume using thumb–index finger pinch distance.

python volume_control.py

3️⃣ Hand Blocks 3D (Experimental)

Interact with wireframe cubes using pinch-and-drag gestures.

python hand_blocks_3d.py

---

## 🎯Learning Outcomes

This project helped in developing hands-on understanding of:

Real-time computer vision pipelines

Hand landmark detection and tracking

Gesture recognition using geometric relationships

Mapping physical gestures to system-level actions

Camera-space interaction and basic depth approximation

Writing modular, experiment-friendly Python code

---

## ⚠️ Limitations

Volume control is Windows-only

Performance depends on lighting and camera quality

The 3D interaction demo is experimental and not physics-accurate

---

## 🔮 Roadmap & Future Enhancements

Multi-hand gesture support

Gesture classification using ML models

Cross-platform audio control

Improved depth estimation

Gesture-controlled UI components

AR / VR integration

Robotics or IoT gesture control

---

##🤝 Contributions

Contributions are welcome!

If you’d like to contribute:

Fork the repository

Create a new branch (feature/your-feature)

Commit your changes

Open a Pull Request with a clear description

Bug fixes, feature ideas, and optimizations are all appreciated.

---


## 📌 Why This Project?

This project was built to explore human-computer interaction beyond keyboards and touchscreens.
It focuses on clarity, performance, and experimentation, making it suitable for learning, demos, and future research-oriented extensions.


---


## 📜 License

This project is open-source and available under the MIT License.


----

## 🙌 Acknowledgements

MediaPipe for efficient hand tracking models

OpenCV for real-time computer vision utilities

Pycaw for Windows audio control

---


