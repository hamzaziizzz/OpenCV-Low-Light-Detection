import cv2
import numpy as np


def roi_coordinates(roi_width: int, roi_height: int, frame_width: int, frame_height: int, x: int = 0, y: int = 0):
    """
    This function returns the coordinates of the region of interest for face detection.
        Face detection will be performed on the region of interest only.

    Args:
        roi_width (int): width of the region of interest
        roi_height (int): height of the region of interest
        frame_width (int): width of the frame
        frame_height (int): height of the frame
        x (int): x coordinate of the middle of the region of interest
        y (int): y coordinate of the bottom of the region of interest

    Returns:
        tuple(roi_left, roi_top, roi_right, roi_bottom): coordinates of the ROI (region of interest)
    """

    frame_x_center = frame_width // 2
    roi_width_half = roi_width // 2

    left_coordinate = frame_x_center - roi_width_half + x
    right_coordinate = frame_x_center + roi_width_half + x

    bottom_coordinate = frame_height - y
    top_coordinate = bottom_coordinate - (roi_height + y)

    return left_coordinate, top_coordinate, right_coordinate, bottom_coordinate


def create_mask(cctv_frame: np.ndarray, left: int, top: int, right: int, bottom: int):
    """
    This function creates a mask for the frame so that face detection is performed only on the region of interest

    Args:
        cctv_frame (numpy.ndarray): frame on which the mask will be applied
        left (int): left coordinate of the region of interest
        top (int): top coordinate of the region of interest
        right (int): right coordinate of the region of interest
        bottom (int): bottom coordinate of the region of interest

    Returns:
        numpy.ndarray: mask for the frame
    """
    mask_region = np.zeros_like(cctv_frame)
    mask_region[top:bottom, left:right] = 1
    return mask_region


def is_low_light(image, threshold=80):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average brightness
    average_brightness = np.mean(gray_image)

    print(average_brightness)
    # Determine if the image is low light
    return average_brightness < threshold


# Initialize webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("rtsp://grilsquad:grilsquad@192.168.5.1:554/stream1")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

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

    # Extract ROI from frame
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    if is_low_light(roi):
        cv2.putText(frame, "Low Light Condition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Normal Light Condition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with the light condition text
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
