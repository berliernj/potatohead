"""
Real-Time Face Feature Mapping with Proper Segmentation

Upgrades in this version:
- Extracts ONLY the eye and mouth shapes (no square patches)
- Uses polygon masks based on facial landmarks
- Blends features smoothly onto the model face
"""

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# -----------------------------
# LANDMARK INDICES (FULL SHAPES)
# -----------------------------
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
NOSE = 1


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_point(landmark, w, h):
    return np.array([int(landmark.x * w), int(landmark.y * h)], dtype=np.float32)


def extract_feature(frame, landmarks, indices, w, h):
    """
    Extract a feature using a polygon mask (eye or mouth).
    Returns cropped image + mask.
    """
    pts = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in indices
    ], dtype=np.int32)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Extract region
    extracted = cv2.bitwise_and(frame, frame, mask=mask)

    # Bounding box
    x, y, w_box, h_box = cv2.boundingRect(pts)

    cropped = extracted[y:y+h_box, x:x+w_box]
    cropped_mask = mask[y:y+h_box, x:x+w_box]

    # Feather edges (important for smooth blending)
    cropped_mask = cv2.GaussianBlur(cropped_mask, (15, 15), 0)

    return cropped, cropped_mask


def overlay_masked(dst, src, mask, center):
    """
    Overlay using mask blending (smooth edges).
    """
    h, w = src.shape[:2]
    x = int(center[0] - w // 2)
    y = int(center[1] - h // 2)

    if y < 0 or x < 0 or y+h > dst.shape[0] or x+w > dst.shape[1]:
        return

    roi = dst[y:y+h, x:x+w]

    mask_f = mask.astype(float) / 255.0
    mask_f = np.expand_dims(mask_f, axis=-1)

    blended = (roi * (1 - mask_f) + src * mask_f).astype(np.uint8)
    dst[y:y+h, x:x+w] = blended


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    canvas = np.zeros_like(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # Key points
            nose = get_point(lm[NOSE], w, h)
            left_eye_center = get_point(lm[33], w, h)
            right_eye_center = get_point(lm[263], w, h)
            mouth_center = get_point(lm[13], w, h)

            # -----------------------------
            # HEAD ROTATION
            # -----------------------------
            eye_center = (left_eye_center + right_eye_center) / 2
            dx = nose[0] - eye_center[0]
            dy = nose[1] - eye_center[1]
            angle = np.degrees(np.arctan2(dy, dx))

            model_center = (int(w / 2), int(h / 2))
            M = cv2.getRotationMatrix2D(model_center, angle, 1.0)

            # Draw base face
            cv2.ellipse(canvas, model_center, (120, 160), angle, 0, 360, (50, 50, 50), -1)

            # -----------------------------
            # EXTRACT SEGMENTED FEATURES
            # -----------------------------
            left_eye_img, left_eye_mask = extract_feature(frame, lm, LEFT_EYE_IDX, w, h)
            right_eye_img, right_eye_mask = extract_feature(frame, lm, RIGHT_EYE_IDX, w, h)
            mouth_img, mouth_mask = extract_feature(frame, lm, MOUTH_IDX, w, h)

            # -----------------------------
            # MODEL FEATURE POSITIONS
            # -----------------------------
            model_left_eye = np.array([w * 0.4, h * 0.45])
            model_right_eye = np.array([w * 0.6, h * 0.45])
            model_mouth = np.array([w * 0.5, h * 0.65])

            def transform_point(p):
                px, py = p
                x = M[0, 0] * px + M[0, 1] * py + M[0, 2]
                y = M[1, 0] * px + M[1, 1] * py + M[1, 2]
                return np.array([x, y])

            model_left_eye = transform_point(model_left_eye)
            model_right_eye = transform_point(model_right_eye)
            model_mouth = transform_point(model_mouth)

            # -----------------------------
            # OVERLAY WITH MASKS
            # -----------------------------
            if left_eye_img.size > 0:
                overlay_masked(canvas, left_eye_img, left_eye_mask, model_left_eye)

            if right_eye_img.size > 0:
                overlay_masked(canvas, right_eye_img, right_eye_mask, model_right_eye)

            if mouth_img.size > 0:
                overlay_masked(canvas, mouth_img, mouth_mask, model_mouth)

    cv2.imshow("Original", frame)
    cv2.imshow("Segmented Model", canvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
