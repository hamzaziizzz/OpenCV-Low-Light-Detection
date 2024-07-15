import time

import cv2
import numpy as np


def roi_coordinates(roi_width: int, roi_height: int, frame_width: int, frame_height: int, x: int = 0, y: int = 0):
    frame_x_center = frame_width // 2
    roi_width_half = roi_width // 2

    left_coordinate = max(0, frame_x_center - roi_width_half + x)
    right_coordinate = min(frame_width, frame_x_center + roi_width_half + x)

    bottom_coordinate = min(frame_height, frame_height - y)
    top_coordinate = max(0, bottom_coordinate - (roi_height + y))

    return left_coordinate, top_coordinate, right_coordinate, bottom_coordinate


def create_mask(cctv_frame: np.ndarray, left: int, top: int, right: int, bottom: int):
    mask_region = np.zeros_like(cctv_frame)
    mask_region[top:bottom, left:right] = 1
    return mask_region


def analyze_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    return edge_density


def analyze_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness


def analyze_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    return contrast


def dynamic_thresholds(image, base_edge_density=0.02, base_brightness=100, base_contrast=50):
    ambient_brightness = analyze_brightness(image)
    ambient_contrast = analyze_contrast(image)

    edge_density_threshold = base_edge_density * (ambient_brightness / 128.0)
    brightness_threshold = base_brightness * (ambient_contrast / 64.0)
    contrast_threshold = base_contrast * (ambient_brightness / 120.0)

    return edge_density_threshold, brightness_threshold, contrast_threshold


def is_well_lit(image):
    # edge_density_threshold, brightness_threshold, contrast_threshold = dynamic_thresholds(image)
    edge_density = analyze_edges(image)
    mean_brightness = analyze_brightness(image)
    contrast = analyze_contrast(image)

    # print(f"Edge Density: {edge_density}")
    # print(f"Mean Brightness: {mean_brightness}")
    # print(f"Contrast: {contrast}")
    # print(f"Dynamic Thresholds - Edge Density: {edge_density_threshold}, Brightness: {brightness_threshold}, Contrast: {contrast_threshold}")

    well_lit = (
            edge_density > 0.2 and
            mean_brightness > 80 and
            contrast > 30
    )

    return well_lit


# Initialize webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("rtsp://grilsquad:grilsquad@192.168.5.1:554/stream1")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


detections = 0
low_intensity = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.resize(frame, (640, 360))

    roi_left, roi_top, roi_right, roi_bottom = roi_coordinates(
        200,
        360,
        frame.shape[1],
        frame.shape[0]
    )
    mask = create_mask(frame, roi_left, roi_top, roi_right, roi_bottom)

    frame = frame * mask

    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (255, 0, 0), 2, cv2.LINE_AA)

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    if is_well_lit(roi):
        cv2.putText(frame, "Normal Light Condition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)
    else:
        print("Inside Else Block")
        detections += 1
        print(f"detections: {detections}")
        if detections > 32:
            low_intensity = True
            print(f"Low Intensity: {low_intensity}")
        if low_intensity:
            print("Low Light Intensity. Wait for 5 minutes")
            detections = 0
            low_intensity = False
            time.sleep(30)
        # cv2.putText(frame, "Low Light Condition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
