import cv2

def list_available_cameras(max_devices=10):
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW often works better on Windows
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

cameras = list_available_cameras()
print("Available camera indexes:", cameras)

# Try to open the default camera (device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open camera. Device 0 may not exist or is in use.")
else:
    print("Camera opened successfully!")

    # Capture one frame
    ret, frame = cap.read()
    if ret:
        print("Frame captured successfully.")
        # Display the frame
        cv2.imshow("Test Frame", frame)
        cv2.waitKey(0)  # Wait for any key press
    else:
        print("Failed to capture frame.")

    cap.release()
    cv2.destroyAllWindows()
