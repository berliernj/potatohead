import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

# -----------------------------
# LANDMARKS
# -----------------------------
UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# -----------------------------
# SMOOTHING BUFFERS
# -----------------------------
def make_buffer():
    return deque(maxlen=5)

mouth_w = make_buffer()
mouth_h = make_buffer()
mouth_c = make_buffer()

leye_w = make_buffer()
leye_h = make_buffer()
leye_c = make_buffer()

reye_w = make_buffer()
reye_h = make_buffer()
reye_c = make_buffer()

# -----------------------------
# HELPERS
# -----------------------------
def pt(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h])

def dist(a, b):
    return np.linalg.norm(a - b)

def smooth(buf):
    return np.mean(buf, axis=0)

# -----------------------------
# DRAW FUNCTIONS
# -----------------------------
def draw_region(frame, center, axis_x, axis_y, color, alpha=0.25):
    overlay = frame.copy()

    cv2.ellipse(
        overlay,
        tuple(center.astype(int)),
        (int(axis_x), int(axis_y)),
        0,
        0,
        360,
        color,
        -1
    )

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# -----------------------------
# MOUTH
# -----------------------------
def update_mouth(frame, lm, w, h):
    up = pt(lm, UPPER_LIP, w, h)
    low = pt(lm, LOWER_LIP, w, h)
    left = pt(lm, MOUTH_LEFT, w, h)
    right = pt(lm, MOUTH_RIGHT, w, h)

    center = (up + low + left + right) / 4
    width = dist(left, right)
    height = dist(up, low)

    mouth_c.append(center)
    mouth_w.append(width)
    mouth_h.append(height)

    c = smooth(mouth_c)
    w_sm = smooth(mouth_w)
    h_sm = smooth(mouth_h)

    return draw_region(
        frame,
        c,
        w_sm * 0.55,
        max(18, h_sm * 3.2),
        (0, 0, 255),
        0.25
    )

# -----------------------------
# EYES
# -----------------------------
def update_eye(frame, lm, top, bottom, left, right, w, h, buffers, color):
    t = pt(lm, top, w, h)
    b = pt(lm, bottom, w, h)
    l = pt(lm, left, w, h)
    r = pt(lm, right, w, h)

    center = (t + b + l + r) / 4
    width = dist(l, r)
    height = dist(t, b)

    buffers["c"].append(center)
    buffers["w"].append(width)
    buffers["h"].append(height)

    c = smooth(buffers["c"])
    w_sm = smooth(buffers["w"])
    h_sm = smooth(buffers["h"])

    return draw_region(
        frame,
        c,
        w_sm * 0.6,
        max(10, h_sm * 2.5),
        color,
        0.25
    )

# -----------------------------
# MAIN
# -----------------------------
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    left_eye_buf = {"c": make_buffer(), "w": make_buffer(), "h": make_buffer()}
    right_eye_buf = {"c": make_buffer(), "w": make_buffer(), "h": make_buffer()}

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
                lm = face_landmarks.landmark

                # 👄 Mouth
                frame = update_mouth(frame, lm, w, h)

                # 👁️ Left eye
                frame = update_eye(
                    frame, lm,
                    LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                    LEFT_EYE_LEFT, LEFT_EYE_RIGHT,
                    w, h,
                    left_eye_buf,
                    (255, 0, 0)
                )

                # 👁️ Right eye
                frame = update_eye(
                    frame, lm,
                    RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                    RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT,
                    w, h,
                    right_eye_buf,
                    (0, 255, 0)
                )

        cv2.imshow("Smooth Eyes + Mouth AR Regions", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
