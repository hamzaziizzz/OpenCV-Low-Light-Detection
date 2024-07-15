import cv2
import numpy as np


def is_too_dark(frame, adaptive=False, threshold=50):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if adaptive:
        # Use adaptive thresholding to determine darkness
        mean_intensity = cv2.mean(gray)[0]
        return mean_intensity < threshold
    else:
        # Calculate the average pixel intensity
        avg_intensity = np.mean(gray)
        # Check if the average intensity is below the threshold
        return avg_intensity < threshold


def reduce_noise(frame):
    # Apply Gaussian blur to reduce noise
    return cv2.GaussianBlur(frame, (5, 5), 0)


def main():
    # Initialize the camera
    cap = cv2.VideoCapture("rtsp://grilsquad:grilsquad@192.168.5.1:554/stream1")  # Use 0 for default camera, or specify your CCTV camera source

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    frame_count = 0
    dark_frames = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1

        # Reduce noise in the frame
        frame = reduce_noise(frame)

        # Check if the frame is too dark
        if is_too_dark(frame, adaptive=True):
            dark_frames += 1

            # Display message on the frame
            cv2.putText(frame, "Too dark for attendance", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"Frame {frame_count}: Too dark for attendance")

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Make decision based on the analysis of the last 60 frames
        if frame_count == 60:
            if dark_frames > 30:
                print("Overall, too dark for attendance")
            else:
                print("Lighting conditions acceptable")
            # Reset counters
            frame_count = 0
            dark_frames = 0

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

