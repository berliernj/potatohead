import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define landmark groups
FACE_OUTLINE = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127
]

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
    87, 14, 317, 402, 318, 324, 308, 415
]

NOSE = [1, 2, 98, 327, 168, 197, 195, 5, 4]

# Colors (BGR)
COLORS = {
    "outline": (255, 255, 255),  # white
    "left_eye": (255, 0, 0),     # blue
    "right_eye": (0, 255, 0),    # green
    "lips": (0, 0, 255),         # red
    "nose": (0, 255, 255)        # yellow
}

def draw_feature(image, landmarks, indices, color, w, h):
    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 2, color, -1)

# Start webcam
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

        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                draw_feature(frame, landmarks, FACE_OUTLINE, COLORS["outline"], w, h)
                draw_feature(frame, landmarks, LEFT_EYE, COLORS["left_eye"], w, h)
                draw_feature(frame, landmarks, RIGHT_EYE, COLORS["right_eye"], w, h)
                draw_feature(frame, landmarks, LIPS, COLORS["lips"], w, h)
                draw_feature(frame, landmarks, NOSE, COLORS["nose"], w, h)

        cv2.imshow("Face Feature Overlay", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
