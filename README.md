# 🚗 Smart Parking Space Occupancy Detection using OpenCV

An interactive computer vision project that detects whether a selected parking space is **Occupied** or **Empty** using **OpenCV**, **NumPy**, **Canny Edge Detection**, and **Edge Density Analysis**.

Users can click on any parking space in the image, and the system analyzes the selected region to determine its occupancy status.

---

## 📌 Features

- Interactive parking space selection using mouse clicks
- Automatic parking occupancy detection
- Works with different vehicle colors using edge detection
- Uses Canny Edge Detection for robust contour extraction
- Displays:
  - 🟢 Empty Parking Space
  - 🔴 Occupied Parking Space
- Real-time visualization with OpenCV
- Clear all detected parking spots with a single key press

---

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy

---

## 📂 Project Structure

```
Parking-Space-Occupancy-Detection/
│── parking_detector.py
│── parking_lot.png
│── README.md
│── requirements.txt
└── screenshots/
```

---

## ⚙️ How It Works

1. Loads the parking lot image.
2. Converts the image to grayscale.
3. Applies Gaussian Blur to reduce noise.
4. Detects edges using Canny Edge Detection.
5. User clicks on a parking space.
6. The program extracts a Region of Interest (ROI).
7. Counts edge pixels inside the ROI.
8. Calculates the edge density ratio.
9. Classifies the parking space:
   - **Edge Ratio > 8% → Occupied**
   - **Edge Ratio ≤ 8% → Empty**
10. Displays the result with a colored bounding box and status label.

---

## 🧠 Detection Algorithm

The system determines occupancy using **edge density**.

```
Edge Ratio = Edge Pixels / Total ROI Pixels
```

Decision Rule:

- **Edge Ratio > 0.08** → 🔴 Occupied
- **Edge Ratio ≤ 0.08** → 🟢 Empty

---

## ▶️ Installation

### Clone the repository

```bash
git clone https://github.com/Fasee9831/Smart-Parking-Space-Detection-using-OpenCV.git
cd Parking-Space-Occupancy-Detection
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python parking_detector.py
```

---

## 🎮 Usage

- **Left Mouse Click** → Select a parking space
- **C** → Clear all selected parking spaces
- **ESC** → Exit the application

---

## 📷 Sample Output

| Empty Parking Space | Occupied Parking Space |
|---------------------|------------------------|
| 🟢 Green Bounding Box | 🔴 Red Bounding Box |

> Add screenshots inside the `screenshots/` folder and update this section with actual images.

---

## 📦 Requirements

```
opencv-python
numpy
```

---

## 💡 Future Improvements

- Automatic parking space detection
- Real-time video support
- Live CCTV integration
- Deep Learning-based vehicle detection (YOLO)
- Parking occupancy statistics
- Parking availability counter
- Web dashboard for monitoring

---

## 🎯 Learning Outcomes

This project demonstrates:

- Image Processing with OpenCV
- Computer Vision Fundamentals
- Edge Detection Techniques
- Region of Interest (ROI)
- Mouse Event Handling
- NumPy Array Slicing
- Threshold-Based Classification
- Interactive GUI Development with OpenCV

---

## 👨‍💻 Author

**Faseeha Zulaika N**

If you found this project helpful, consider ⭐ starring the repository.

---

## 📄 License

This project is licensed under the MIT License.
