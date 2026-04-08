"""
Real-Time Face Feature Mapping onto a Moving Model

This script:
1. Captures webcam video
2. Detects facial landmarks using MediaPipe
3. Extracts eyes and mouth regions from the real face
4. Estimates head rotation based on landmark geometry
5. Draws a simple "model face"
6. Rotates the model to match head movement
7. Places (overlays) the extracted features onto the model

This is a foundational version of a face-driven avatar system.
"""

import cv2
import mediapipe as mp
import numpy as np


# -----------------------------
# INITIALIZE MEDIAPIPE
# -----------------------------
# MediaPipe Face Mesh gives 468 3D facial landmarks
mp_face_mesh = mp.solutions.face_mesh

# refine_landmarks=True improves accuracy for eyes and lips
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


# -----------------------------
# START WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)


# -----------------------------
# LANDMARK INDICES (KEY POINTS)
# -----------------------------
# These are indices from MediaPipe's 468-point face model
# We only use a few points for simplicity and speed

LEFT_EYE = [33, 133]     # approximate eye region
RIGHT_EYE = [362, 263]
MOUTH = [61, 291]
NOSE = 1                 # used for head direction


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_point(landmark, w, h):
    """
    Convert a normalized landmark (0–1 range) into pixel coordinates.

    MediaPipe gives coordinates relative to image size.
    We scale them to actual pixel positions.
    """
    return np.array([
        int(landmark.x * w),
        int(landmark.y * h)
    ], dtype=np.float32)


def extract_patch(frame, center, size=60):
    """
    Extract a square patch (ROI) around a feature.

    Used to grab:
    - left eye
    - right eye
    - mouth

    This is a simple crop (no masking or blending).
    """
    x, y = int(center[0]), int(center[1])
    half = size // 2

    return frame[y-half:y+half, x-half:x+half]


def overlay(dst, src, center):
    """
    Place (overlay) a feature patch onto the destination image.

    NOTE:
    - No blending is used (hard edges)
    - If the patch goes out of bounds, we skip it
    """
    h, w = src.shape[:2]

    # Compute top-left corner
    x = int(center[0] - w // 2)
    y = int(center[1] - h // 2)

    # Boundary check to avoid crashes
    if y < 0 or x < 0 or y + h > dst.shape[0] or x + w > dst.shape[1]:
        return

    dst[y:y+h, x:x+w] = src


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Convert BGR (OpenCV) → RGB (MediaPipe requirement)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run face landmark detection
    results = face_mesh.process(rgb)

    # Create a blank "model face" canvas
    canvas = np.zeros_like(frame)

    # If a face is detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # All 468 landmarks
            lm = face_landmarks.landmark

            # -----------------------------
            # EXTRACT KEY LANDMARK POSITIONS
            # -----------------------------
            # Convert normalized coordinates → pixel positions

            nose = get_point(lm[NOSE], w, h)
            left_eye = get_point(lm[33], w, h)
            right_eye = get_point(lm[263], w, h)
            mouth = get_point(lm[13], w, h)

            # -----------------------------
            # HEAD ROTATION ESTIMATION
            # -----------------------------
            # We approximate head tilt using:
            # vector from eye center → nose

            eye_center = (left_eye + right_eye) / 2

            dx = nose[0] - eye_center[0]
            dy = nose[1] - eye_center[1]

            # Angle of head tilt in degrees
            angle = np.degrees(np.arctan2(dy, dx))

            # -----------------------------
            # MODEL FACE SETUP
            # -----------------------------
            # Define where the model sits (center of screen)

            model_center = (int(w / 2), int(h / 2))

            # Create rotation matrix for the entire model
            # This ensures the model rotates with your head
            M = cv2.getRotationMatrix2D(model_center, angle, 1.0)

            # Draw a simple face (ellipse)
            # This acts as our "avatar base"
            cv2.ellipse(
                canvas,
                model_center,
                (120, 160),   # width, height
                angle,        # rotation
                0, 360,
                (50, 50, 50),
                -1
            )

            # -----------------------------
            # EXTRACT REAL FACE FEATURES
            # -----------------------------
            # These are patches from the webcam image

            left_eye_patch = extract_patch(frame, left_eye, 60)
            right_eye_patch = extract_patch(frame, right_eye, 60)
            mouth_patch = extract_patch(frame, mouth, 80)

            # -----------------------------
            # DEFINE MODEL FEATURE POSITIONS
            # -----------------------------
            # These are relative positions ON the model face

            model_left_eye = np.array([w * 0.4, h * 0.45])
            model_right_eye = np.array([w * 0.6, h * 0.45])
            model_mouth = np.array([w * 0.5, h * 0.65])

            # -----------------------------
            # APPLY SAME ROTATION TO FEATURES
            # -----------------------------
            # Critical concept:
            # We transform feature positions using the SAME
            # rotation matrix as the face → keeps alignment correct

            def transform_point(p):
                px, py = p

                x = M[0, 0] * px + M[0, 1] * py + M[0, 2]
                y = M[1, 0] * px + M[1, 1] * py + M[1, 2]

                return np.array([x, y])

            model_left_eye = transform_point(model_left_eye)
            model_right_eye = transform_point(model_right_eye)
            model_mouth = transform_point(model_mouth)

            # -----------------------------
            # OVERLAY FEATURES ON MODEL
            # -----------------------------
            # Place real eyes and mouth onto the animated face

            if left_eye_patch.size > 0:
                overlay(canvas, left_eye_patch, model_left_eye)

            if right_eye_patch.size > 0:
                overlay(canvas, right_eye_patch, model_right_eye)

            if mouth_patch.size > 0:
                overlay(canvas, mouth_patch, model_mouth)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    cv2.imshow("Original", frame)
    cv2.imshow("Tracked Model", canvas)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
