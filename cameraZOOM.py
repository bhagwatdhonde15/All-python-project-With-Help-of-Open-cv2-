import cv2
import mediapipe as mp
import math

# --- Settings you can tweak ---
MIN_DIST = 20      # distance (in pixels) for "no zoom" (fingers almost touching)
MAX_DIST = 200     # distance (in pixels) for "full zoom"
MAX_ZOOM = 3.0     # maximum zoom level (3x)

# Init MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

current_zoom = 1.0  # start with no zoom

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera")
        break

    h, w, _ = frame.shape

    # Convert BGR -> RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand landmarks
    results = hands.process(rgb)

    distance = None

    if results.multi_hand_landmarks:
        # Use first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks on frame (just for visualization)
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        # Thumb tip = id 4, Index finger tip = id 8
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        # Convert normalized [0,1] coords to pixel coords
        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

        # Draw circles on thumb and index tips
        cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Euclidean distance between fingers
        distance = math.hypot(x2 - x1, y2 - y1)

        # --- Map distance -> zoom level ---
        # Clamp distance between MIN_DIST and MAX_DIST
        d = max(MIN_DIST, min(MAX_DIST, distance))

        # Normalize to 0..1
        t = (d - MIN_DIST) / (MAX_DIST - MIN_DIST)  # 0 when close, 1 when far

        # More distance -> more zoom
        target_zoom = 1.0 + t * (MAX_ZOOM - 1.0)

        # Smooth zoom a bit (so it doesn't jump too much)
        alpha = 0.2
        current_zoom = (1 - alpha) * current_zoom + alpha * target_zoom

    else:
        # If no hand detected, slowly go back to no zoom
        alpha = 0.1
        current_zoom = (1 - alpha) * current_zoom + alpha * 1.0

    # --- Apply zoom by cropping center ---
    zoom = max(1.0, min(MAX_ZOOM, current_zoom))  # keep in [1, MAX_ZOOM]
    new_w = int(w / zoom)
    new_h = int(h / zoom)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Safety clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(cropped, (w, h))

    # --- Show info text ---
    if distance is not None:
        cv2.putText(zoomed_frame, f"Dist: {int(distance)} px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(zoomed_frame, f"Zoom: {zoom:.2f}x", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(zoomed_frame, "Press 'q' to quit", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Finger Zoom Camera", zoomed_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
