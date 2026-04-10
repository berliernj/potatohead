import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Landmark groups (used to compute ellipses)
LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 78, 95, 88, 178,
    87, 14, 317, 402, 318, 324, 415
]

NOSE = [1, 2, 98, 327, 168, 197, 195, 5, 4]

def draw_ellipse(image, landmarks, indices, color, w, h):
    pts = []
    for i in indices:
        lm = landmarks[i]
        pts.append((int(lm.x * w), int(lm.y * h)))

    pts = np.array(pts)

    if len(pts) >= 5:  # required for ellipse fitting
        ellipse = cv2.fitEllipse(pts)
        cv2.ellipse(image, ellipse, color, 2)

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

                # 👁️ Eyes (full region feel)
                draw_ellipse(frame, landmarks, LEFT_EYE, (255, 0, 0), w, h)
                draw_ellipse(frame, landmarks, RIGHT_EYE, (0, 255, 0), w, h)

                # 👄 Lips
                draw_ellipse(frame, landmarks, LIPS, (0, 0, 255), w, h)

                # 👃 Nose
                draw_ellipse(frame, landmarks, NOSE, (0, 255, 255), w, h)

        cv2.imshow("Ellipse Face Regions", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
