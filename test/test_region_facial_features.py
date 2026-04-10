import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Landmark groups
LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

MOUTH = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 78, 95, 88, 178,
    87, 14, 317, 402, 318, 324, 415
]

def get_points(landmarks, indices, w, h):
    pts = []
    for i in indices:
        lm = landmarks[i]
        pts.append((int(lm.x * w), int(lm.y * h)))
    return np.array(pts)

def create_region_mask(frame_shape, points, padding=10):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)

    # Get tight shape
    hull = cv2.convexHull(points)

    # Fill region
    cv2.fillConvexPoly(mask, hull, 255)

    # Expand region outward
    if padding > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding, padding))
        mask = cv2.dilate(mask, kernel)

    return mask

def apply_overlay(frame, mask, color, alpha=0.3):
    overlay = frame.copy()
    overlay[mask == 255] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# Webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # 👁️ LEFT EYE
                left_pts = get_points(landmarks, LEFT_EYE, w, h)
                left_mask = create_region_mask(frame.shape, left_pts, padding=15)
                frame = apply_overlay(frame, left_mask, (255, 0, 0))

                # 👁️ RIGHT EYE
                right_pts = get_points(landmarks, RIGHT_EYE, w, h)
                right_mask = create_region_mask(frame.shape, right_pts, padding=15)
                frame = apply_overlay(frame, right_mask, (0, 255, 0))

                # 👄 MOUTH
                mouth_pts = get_points(landmarks, MOUTH, w, h)
                mouth_mask = create_region_mask(frame.shape, mouth_pts, padding=20)
                frame = apply_overlay(frame, mouth_mask, (0, 0, 255))

        cv2.imshow("Eyes + Mouth Regions", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
