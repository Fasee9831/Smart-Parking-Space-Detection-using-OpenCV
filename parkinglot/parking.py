import cv2
import numpy as np

# 1. Load Image
img = cv2.imread('parking_lot.png')
if img is None:
    raise FileNotFoundError("Could not find 'parking_lot.png'. Please save the generated image in the script folder.")

# 2. Pre-process Image for Universal Detection (Any Car Color)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Canny Edge Detection highlights vehicle contours regardless of car color
edges = cv2.Canny(blurred, 50, 150)

display = img.copy()

# Store marked spots: [(x1, y1, x2, y2, status_text, color), ...]
marked_spots = []

# Adjustable ROI Size (Tweak based on how far back in perspective you click)
SPOT_W, SPOT_H = 90, 60  

def check_spot_edge_density(x, y):
    """Calculates edge density within the selected bounding box."""
    x1, y1 = max(0, x - SPOT_W // 2), max(0, y - SPOT_H // 2)
    x2, y2 = min(img.shape[1], x1 + SPOT_W), min(img.shape[0], y1 + SPOT_H)
    
    # Extract ROI from Canny Edge Mask
    roi_edges = edges[y1:y2, x1:x2]
    edge_pixel_count = cv2.countNonZero(roi_edges)
    
    # Calculate area to normalize edge ratio
    roi_area = (x2 - x1) * (y2 - y1)
    edge_ratio = edge_pixel_count / float(roi_area) if roi_area > 0 else 0

    # Threshold: Smooth asphalt usually has < 5% edge pixels, cars have > 10-15%
    if edge_ratio > 0.08:
        status = f"Occupied ({int(edge_ratio*100)}%)"
        color = (0, 0, 255) # Red for Occupied
    else:
        status = f"Empty ({int(edge_ratio*100)}%)"
        color = (0, 255, 0) # Green for Empty

    marked_spots.append((x1, y1, x2, y2, status, color))

def redraw():
    """Redraws all spots and UI overlay."""
    canvas = img.copy()
    for x1, y1, x2, y2, status, color in marked_spots:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, status, (x1, max(15, y1 - 6)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    
    # UI Instructions overlay
    cv2.putText(canvas, "Left Click: Select Spot | 'c': Clear | 'ESC': Exit", 
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        check_spot_edge_density(x, y)

cv2.namedWindow("Smart Parking Detection System", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Smart Parking Detection System", on_mouse)

print("--- Instructions ---")
print("1. Click on empty spots or parked cars in the image.")
print("2. Press 'c' to clear all selections.")
print("3. Press 'ESC' to exit.")

while True:
    cv2.imshow("Smart Parking Detection System", redraw())
    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c'):
        marked_spots.clear()

cv2.destroyAllWindows()