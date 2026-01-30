from picamera2 import Picamera2
import cv2
import numpy as np

# -----------------------------
# Camera setup
# -----------------------------
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
)
picam2.start()

# -----------------------------
# HSV Skin Mask (YOUR RANGES)
# -----------------------------
def skin_mask_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)

    lower = np.array([0, 70, 200], dtype=np.uint8)
    upper = np.array([30, 90, 235], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

# -----------------------------
# Keep largest component
# -----------------------------
def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out

# -----------------------------
# Skeletonization
# -----------------------------
def skeletonize(mask):
    return cv2.ximgproc.thinning(mask)

# -----------------------------
# Find skeleton endpoints
# -----------------------------
def find_endpoints(skel):
    sk = (skel > 0).astype(np.uint8)

    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    conv = cv2.filter2D(sk, -1, kernel)

    # Endpoint = center + exactly one neighbor
    pts = np.argwhere(conv == 11)
    return [(int(x), int(y)) for (y, x) in pts]

# -----------------------------
# Finger counting
# -----------------------------
def count_fingers(endpoints, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, []

    y_min, y_max = ys.min(), ys.max()
    height = y_max - y_min

    # Reject wrist endpoints
    fingertip_y_thresh = y_min + int(0.65 * height)

    tips = [p for p in endpoints if p[1] < fingertip_y_thresh]

    # Merge nearby endpoints
    filtered = []
    min_dist = 25
    for p in tips:
        if all((p[0]-q[0])**2 + (p[1]-q[1])**2 > min_dist**2 for q in filtered):
            filtered.append(p)

    return len(filtered), filtered

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        rgb = picam2.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # ROI for stability
        cv2.rectangle(bgr, (80, 80), (560, 460), (0, 255, 0), 2)
        roi = bgr[80:460, 80:560]

        # 1) HSV mask
        mask = skin_mask_hsv(roi)
        mask = largest_component(mask)

        if mask is None:
            cv2.imshow("Frame", bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # 2) Skeleton
        skel = skeletonize(mask)

        # 3) Endpoints
        endpoints = find_endpoints(skel)

        # 4) Finger count
        count, tips = count_fingers(endpoints, mask)

        # Visualization
        skel_vis = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

        for (x, y) in endpoints:
            cv2.circle(skel_vis, (x, y), 4, (0, 255, 255), -1)
        for (x, y) in tips:
            cv2.circle(skel_vis, (x, y), 6, (0, 0, 255), -1)

        cv2.putText(bgr, f"Fingers: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("HSV Mask", mask)
        cv2.imshow("Skeleton + Fingertips", skel_vis)
        cv2.imshow("Frame", bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
