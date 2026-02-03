import cv2
import mediapipe as mp
from picamera2 import Picamera2
import time
from gpiozero import AngularServo



servo_x = AngularServo(23, min_pulse_width=0.0006, max_pulse_width=0.0024)
servo_y = AngularServo(24, min_pulse_width=0.0006, max_pulse_width=0.0024)

pos_x = 0
pos_y = 0



err_x_prev = 0
err_y_prev = 0

last_control_time = time.time()
CONTROL_DT = 0.1  # 10 Hz


# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def pid_controller(error, kp, ki, kd, previous_error, integral, dt):
    integral += error * dt
    derivative = (error - previous_error) / dt
    control = kp * error + ki * integral + kd * derivative
    return control, integral


prev_err_x = 0
prev_err_y = 0

integral_x = 0
integral_y = 0




# Camera setup
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "XRGB8888"}
    )
)
picam2.start()
time.sleep(2)

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # shoulder center (primary)
        left_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if left_sh.visibility > 0.5 and right_sh.visibility > 0.5:
            cx = int((left_sh.x + right_sh.x) * 0.5 * w)
            cy = int((left_sh.y + right_sh.y) * 0.5 * h)
        else:
            # hip fallback
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            cx = int((left_hip.x + right_hip.x) * 0.5 * w)
            cy = int((left_hip.y + right_hip.y) * 0.5 * h)

        # Draw target
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

        # Draw skeleton (optional) TURN ON IF WANT THE LINES
        #mp.solutions.drawing_utils.draw_landmarks(
        #   frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        #)

        # Error signal (for pan-tilt)
        err_x = (cx - w//2) / (w//2)   # range ~[-1, 1]
        err_y = (cy - h//2) / (h//2)

        now = time.time()



        if now - last_control_time >= CONTROL_DT:
            dt = now - last_control_time
            last_control_time = now

            # Deadzone
            DEADZONE = 0.05
            if abs(err_x) < DEADZONE: err_x = 0
            if abs(err_y) < DEADZONE: err_y = 0

            # Low-pass filter
            ALPHA = 0.8
            err_x = ALPHA * err_x_prev + (1 - ALPHA) * err_x
            err_y = ALPHA * err_y_prev + (1 - ALPHA) * err_y
            err_x_prev = err_x
            err_y_prev = err_y

            # PID gains
            kp = 0.4
            ki = 0.02
            kd = 0.0

            x_cor, integral_x = pid_controller(err_x, kp, ki, kd, prev_err_x, integral_x, dt)
            y_cor, integral_y = pid_controller(err_y, kp, ki, kd, prev_err_y, integral_y, dt)

            prev_err_x = err_x
            prev_err_y = err_y

            # Clamp integral (anti-windup)
            integral_x = max(-1.0, min(1.0, integral_x))
            integral_y = max(-1.0, min(1.0, integral_y))

            # Limit step size
            MAX_STEP = 2
            x_cor = max(-MAX_STEP, min(MAX_STEP, x_cor))
            y_cor = max(-MAX_STEP, min(MAX_STEP, y_cor))

            # Update servo positions
            pos_x += x_cor
            pos_y += y_cor

            pos_x = max(-60, min(60, pos_x))
            pos_y = max(-45, min(45, pos_y))

            servo_x.angle = pos_x
            servo_y.angle = pos_y


        cv2.putText(frame,
                    f"err_x={err_x}, err_y={err_y}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    cv2.imshow("Body Tracking (Pose)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
'''

import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

# ---------------- Camera setup ----------------
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "XRGB8888"}
    )
)
picam2.start()
time.sleep(2)

# ---------------- Load YOLO ONCE ----------------
model = YOLO("yolo11n-pose.pt")

# ---------------- Main loop ----------------
while True:
    # Capture frame
    frame = picam2.capture_array()

    # Convert BGRA → BGR for OpenCV / YOLO
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Run YOLO pose inference on the frame
    results = model(frame, conf=0.5, verbose=False)

    # Draw results
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Pose (Picamera2)", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
'''