import math
import numpy as np
import pygame
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ============================================================
# Configuration
# ============================================================

SCREEN_W = 1400
SCREEN_H = 900
FPS = 60

PIXELS_PER_UNIT = 35.0
ORIGIN_X = SCREEN_W // 2 + 150
ORIGIN_Y = SCREEN_H // 2 - 50

WHITE = (250, 250, 250)
BLACK = (20, 20, 20)
GRAY = (140, 140, 140)
LIGHT_GRAY = (225, 225, 225)
RED = (210, 60, 60)
GREEN = (50, 170, 70)
BLUE = (60, 110, 220)
ORANGE = (230, 140, 30)
PURPLE = (150, 70, 180)

JOINT_RADIUS_PX = 7
TARGET_RADIUS_PX = 10

# ============================================================
# Mechanism dimensions
# ============================================================

AB = 7.0
AY = 3.25
YZ = -1.5          # keep magnitude positive
ZF = 0.845
EF = 3.946
BE = 2.25
BD = 10.0
CD = 4.0
ANGLE_BCD_DEG = 146.0

# Flip this if Z appears on wrong side of AB
Z_OFFSET_SIGN = -1.0

# Angle bounds
THETA1_MIN_DEG = 120.0
THETA1_MAX_DEG = 260.0
THETA2_MIN_DEG = -170.0
THETA2_MAX_DEG = 170.0

# Good starting guess
DEFAULT_GUESS = np.radians([179.5, -92.0])

# ============================================================
# Manual sweep settings
# ============================================================

MAX_MANUAL_POINTS = 500
MIN_POINT_SPACING = 0.08      # minimum D movement to record a new point
AUTO_CLOSE_ON_MAX_POINTS = True
SAVE_PLOTS_TO_FILES = True

# ============================================================
# Utility
# ============================================================

def deg(rad):
    return math.degrees(rad)

def world_to_screen(x, y):
    sx = ORIGIN_X + x * PIXELS_PER_UNIT
    sy = ORIGIN_Y - y * PIXELS_PER_UNIT
    return int(round(sx)), int(round(sy))

def screen_to_world(sx, sy):
    x = (sx - ORIGIN_X) / PIXELS_PER_UNIT
    y = (ORIGIN_Y - sy) / PIXELS_PER_UNIT
    return x, y

def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector encountered.")
    return v / n

def wrap_to_180(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0

def angle_of_vector_deg(vx, vy):
    return math.degrees(math.atan2(vy, vx))

def angle_wrt_negative_x_deg(P_from, P_to):
    """
    Angle of vector P_from -> P_to relative to negative x-axis.
    0 deg means pointing exactly left.
    """
    vx = P_to[0] - P_from[0]
    vy = P_to[1] - P_from[1]
    theta_global = angle_of_vector_deg(vx, vy)
    return wrap_to_180(theta_global - 180.0)

def angle_wrt_negative_y_deg(P_from, P_to):
    """
    Angle of vector P_from -> P_to relative to negative y-axis.
    0 deg means pointing straight down.
    """
    vx = P_to[0] - P_from[0]
    vy = P_to[1] - P_from[1]
    theta_global = angle_of_vector_deg(vx, vy)
    return wrap_to_180(theta_global - (-90.0))

def draw_text(screen, text, x, y, font, color=BLACK):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))

def circle_intersections(c0, r0, c1, r1):
    """
    Returns intersection points of two circles.
    Output: list of 0, 1, or 2 numpy arrays.
    """
    x0, y0 = c0
    x1, y1 = c1

    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy)

    if d > r0 + r1 + 1e-12:
        return []
    if d < abs(r0 - r1) - 1e-12:
        return []
    if d < 1e-12:
        return []

    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h_sq = r0 * r0 - a * a
    if h_sq < 0:
        if h_sq > -1e-10:
            h_sq = 0.0
        else:
            return []
    h = math.sqrt(h_sq)

    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    rx = -dy * (h / d)
    ry = dx * (h / d)

    p1 = np.array([xm + rx, ym + ry])
    p2 = np.array([xm - rx, ym - ry])

    if np.linalg.norm(p1 - p2) < 1e-10:
        return [p1]
    return [p1, p2]

def point_on_link_with_offset(A, B, along_from_A, offset_perp):
    """
    Point defined in local frame of link AB.
    """
    u = unit(B - A)
    n = np.array([-u[1], u[0]])
    return A + along_from_A * u + offset_perp * n

def compute_BC_length():
    """
    From rigid triangle B-C-D:
      BD^2 = BC^2 + CD^2 - 2*BC*CD*cos(angle BCD)
    Solve for BC.
    """
    theta = math.radians(ANGLE_BCD_DEG)
    a = 1.0
    b = -2.0 * CD * math.cos(theta)
    c = CD * CD - BD * BD

    disc = b * b - 4 * a * c
    if disc < 0:
        raise ValueError("No real BC length from given dimensions.")

    x1 = (-b + math.sqrt(disc)) / (2 * a)
    x2 = (-b - math.sqrt(disc)) / (2 * a)

    roots = [x for x in (x1, x2) if x > 0]
    if not roots:
        raise ValueError("No positive BC length found.")

    return max(roots)

BC = compute_BC_length()

# ============================================================
# Forward kinematics
# ============================================================

def forward_kinematics(theta1, theta2):
    """
    theta1: angle of link AB about fixed point A
    theta2: angle of crank ZF about point Z

    Returns:
      A, B, Y, Z, F, E, C, D
    or None if infeasible.
    """
    A = np.array([0.0, 0.0])

    # B rotates around A
    B = A + np.array([
        AB * math.cos(theta1),
        AB * math.sin(theta1)
    ])

    # Y and Z attached to AB
    Y = point_on_link_with_offset(A, B, along_from_A=AY, offset_perp=0.0)
    Z = point_on_link_with_offset(A, B, along_from_A=AY, offset_perp=Z_OFFSET_SIGN * YZ)

    # F rotates around Z
    F = Z + np.array([
        ZF * math.cos(theta2),
        ZF * math.sin(theta2)
    ])

    # E from circles around B and F
    e_candidates = circle_intersections(B, BE, F, EF)
    if len(e_candidates) == 0:
        return None

    # Pick lower candidate
    E = min(e_candidates, key=lambda p: p[1])

    # C lies on line from B through E
    dir_BE = unit(E - B)
    C = B + BC * dir_BE

    # D from circles around B and C
    d_candidates = circle_intersections(B, BD, C, CD)
    if len(d_candidates) == 0:
        return None

    D = min(d_candidates, key=lambda p: p[1])

    return {
        "A": A,
        "B": B,
        "Y": Y,
        "Z": Z,
        "F": F,
        "E": E,
        "C": C,
        "D": D
    }

# ============================================================
# IK
# ============================================================

def ik_residual(angles, x_target, y_target):
    theta1, theta2 = angles
    joints = forward_kinematics(theta1, theta2)

    if joints is None:
        return np.array([100.0, 100.0])

    D = joints["D"]
    return np.array([
        D[0] - x_target,
        D[1] - y_target
    ])

def solve_ik_numerical(x_target, y_target, initial_guess):
    lower_bounds = np.radians([THETA1_MIN_DEG, THETA2_MIN_DEG])
    upper_bounds = np.radians([THETA1_MAX_DEG, THETA2_MAX_DEG])

    # make sure initial guess is within bounds
    x0 = np.array(initial_guess, dtype=float)
    x0[0] = np.clip(x0[0], lower_bounds[0] + 1e-6, upper_bounds[0] - 1e-6)
    x0[1] = np.clip(x0[1], lower_bounds[1] + 1e-6, upper_bounds[1] - 1e-6)

    result = least_squares(
        ik_residual,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        args=(x_target, y_target),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=500
    )

    theta1, theta2 = result.x
    err = np.linalg.norm(ik_residual(result.x, x_target, y_target))
    success = result.success and err < 1e-3
    return success, theta1, theta2, err

def joint_angles_to_motor_angles(theta1, theta2):
    motor1_deg = deg(theta1)
    motor2_deg = deg(theta2)
    return motor1_deg, motor2_deg

# ============================================================
# Data extraction
# ============================================================

def compute_pose_features(joints, theta1, theta2):
    A = joints["A"]
    B = joints["B"]
    Z = joints["Z"]
    F = joints["F"]
    D = joints["D"]

    x_left = A[0] - D[0]
    ad_distance = np.linalg.norm(D - A)
    ab_angle_neg_x = angle_wrt_negative_x_deg(A, B)
    zf_angle_neg_y = angle_wrt_negative_y_deg(Z, F)

    return {
        "D_x": D[0],
        "D_y": D[1],
        "x_left": x_left,
        "ad_distance": ad_distance,
        "ab_angle_neg_x_deg": ab_angle_neg_x,
        "zf_angle_neg_y_deg": zf_angle_neg_y,
        "theta1_deg": deg(theta1),
        "theta2_deg": deg(theta2),
    }

def should_record_point(rows, joints, min_spacing=MIN_POINT_SPACING):
    """
    Record a new point only if D has moved enough from last saved point.
    """
    if len(rows) == 0:
        return True

    D = joints["D"]
    last_x = rows[-1]["D_x"]
    last_y = rows[-1]["D_y"]

    dist = math.hypot(D[0] - last_x, D[1] - last_y)
    return dist >= min_spacing

def print_sweep_table(rows):
    if len(rows) == 0:
        print("No rows collected.")
        return

    print("\nCollected sweep data:")
    print("-" * 110)
    print(f"{'idx':>3s} {'D_x':>10s} {'D_y':>10s} {'x_left':>10s} {'AD_dist':>10s} {'AB<-x(deg)':>14s} {'ZF<-y(deg)':>14s} {'theta1':>10s} {'theta2':>10s}")
    print("-" * 110)

    for i, r in enumerate(rows):
        print(
            f"{i:3d} "
            f"{r['D_x']:10.3f} "
            f"{r['D_y']:10.3f} "
            f"{r['x_left']:10.3f} "
            f"{r['ad_distance']:10.3f} "
            f"{r['ab_angle_neg_x_deg']:14.3f} "
            f"{r['zf_angle_neg_y_deg']:14.3f} "
            f"{r['theta1_deg']:10.3f} "
            f"{r['theta2_deg']:10.3f}"
        )
    print("-" * 110)

def save_rows_to_csv(rows, filename="manual_sweep_data.csv"):
    if len(rows) == 0:
        return

    header = [
        "D_x",
        "D_y",
        "x_left",
        "ad_distance",
        "ab_angle_neg_x_deg",
        "zf_angle_neg_y_deg",
        "theta1_deg",
        "theta2_deg",
        "target_x",
        "target_y",
        "ik_error"
    ]

    with open(filename, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in header) + "\n")

    print(f"Saved CSV: {filename}")

def plot_sweep_results(rows):
    print(f"\nNumber of valid points collected: {len(rows)}")

    if len(rows) == 0:
        print("No sweep data to plot.")
        return

    ad_dist = np.array([r["ad_distance"] for r in rows])
    ab_ang = np.array([r["ab_angle_neg_x_deg"] for r in rows])
    zf_ang = np.array([r["zf_angle_neg_y_deg"] for r in rows])

    plt.figure(figsize=(8, 5))
    plt.plot(ad_dist, ab_ang, marker="o")
    plt.xlabel("Distance AD")
    plt.ylabel("Angle of AB w.r.t. negative x-axis (deg)")
    plt.title("AB angle vs AD distance")
    plt.grid(True)
    plt.tight_layout()
    if SAVE_PLOTS_TO_FILES:
        plt.savefig("ab_angle_vs_ad.png", dpi=200)

    plt.figure(figsize=(8, 5))
    plt.plot(ad_dist, zf_ang, marker="o")
    plt.xlabel("Distance AD")
    plt.ylabel("Angle of ZF w.r.t. negative y-axis (deg)")
    plt.title("ZF angle vs AD distance")
    plt.grid(True)
    plt.tight_layout()
    if SAVE_PLOTS_TO_FILES:
        plt.savefig("zf_angle_vs_ad.png", dpi=200)

    if SAVE_PLOTS_TO_FILES:
        print("Saved plots:")
        print("  ab_angle_vs_ad.png")
        print("  zf_angle_vs_ad.png")

    plt.show()

# ============================================================
# Drawing
# ============================================================

def draw_grid(screen):
    step = int(PIXELS_PER_UNIT)
    for sx in range(0, SCREEN_W, step):
        pygame.draw.line(screen, LIGHT_GRAY, (sx, 0), (sx, SCREEN_H), 1)
    for sy in range(0, SCREEN_H, step):
        pygame.draw.line(screen, LIGHT_GRAY, (0, sy), (SCREEN_W, sy), 1)

    pygame.draw.line(screen, GRAY, (0, ORIGIN_Y), (SCREEN_W, ORIGIN_Y), 2)
    pygame.draw.line(screen, GRAY, (ORIGIN_X, 0), (ORIGIN_X, SCREEN_H), 2)

def draw_mechanism(screen, joints, target_world, reachable):
    if joints is None:
        return

    A = joints["A"]
    B = joints["B"]
    Y = joints["Y"]
    Z = joints["Z"]
    F = joints["F"]
    E = joints["E"]
    C = joints["C"]
    D = joints["D"]

    A_s = world_to_screen(*A)
    B_s = world_to_screen(*B)
    Y_s = world_to_screen(*Y)
    Z_s = world_to_screen(*Z)
    F_s = world_to_screen(*F)
    E_s = world_to_screen(*E)
    C_s = world_to_screen(*C)
    D_s = world_to_screen(*D)
    T_s = world_to_screen(*target_world)

    target_color = GREEN if reachable else RED
    pygame.draw.circle(screen, target_color, T_s, TARGET_RADIUS_PX, 3)
    pygame.draw.line(screen, target_color, (T_s[0] - 8, T_s[1]), (T_s[0] + 8, T_s[1]), 2)
    pygame.draw.line(screen, target_color, (T_s[0], T_s[1] - 8), (T_s[0], T_s[1] + 8), 2)

    pygame.draw.line(screen, BLACK, A_s, B_s, 5)      # AB
    pygame.draw.line(screen, GRAY, Y_s, Z_s, 2)       # YZ
    pygame.draw.line(screen, PURPLE, Z_s, F_s, 5)     # ZF
    pygame.draw.line(screen, BLUE, B_s, E_s, 5)       # BE
    pygame.draw.line(screen, GREEN, E_s, F_s, 5)      # EF
    pygame.draw.line(screen, BLUE, B_s, C_s, 4)       # BC
    pygame.draw.line(screen, RED, C_s, D_s, 5)        # CD
    pygame.draw.line(screen, GRAY, B_s, D_s, 2)       # BD

    points = [
        ("A", A_s, BLACK),
        ("B", B_s, BLACK),
        ("Y", Y_s, ORANGE),
        ("Z", Z_s, PURPLE),
        ("F", F_s, PURPLE),
        ("E", E_s, ORANGE),
        ("C", C_s, RED),
        ("D", D_s, RED),
    ]

    font_small = pygame.font.SysFont("arial", 18)
    for name, p, color in points:
        pygame.draw.circle(screen, color, p, JOINT_RADIUS_PX)
        draw_text(screen, name, p[0] + 10, p[1] - 18, font_small)

def draw_info_panel(screen, theta1, theta2, motor1, motor2, joints, target_world, err, reachable, collected_count):
    panel_x = 25
    panel_y = 25
    panel_w = 560
    panel_h = 520

    pygame.draw.rect(screen, (245, 245, 245), (panel_x, panel_y, panel_w, panel_h))
    pygame.draw.rect(screen, BLACK, (panel_x, panel_y, panel_w, panel_h), 2)

    title_font = pygame.font.SysFont("arial", 28, bold=True)
    font = pygame.font.SysFont("consolas", 20)

    draw_text(screen, "Crab Linkage IK", panel_x + 20, panel_y + 18, title_font)

    y = panel_y + 70
    line = 36

    draw_text(screen, f"theta1 (AB angle)              : {deg(theta1):8.2f} deg", panel_x + 20, y, font)
    y += line
    draw_text(screen, f"theta2 (ZF angle)              : {deg(theta2):8.2f} deg", panel_x + 20, y, font)
    y += line + 6

    draw_text(screen, f"motor1 angle                   : {motor1:8.2f} deg", panel_x + 20, y, font, PURPLE)
    y += line
    draw_text(screen, f"motor2 angle                   : {motor2:8.2f} deg", panel_x + 20, y, font, PURPLE)
    y += line + 6

    if joints is not None:
        feats = compute_pose_features(joints, theta1, theta2)
        D = joints["D"]
        draw_text(screen, f"D x                            : {D[0]:8.3f}", panel_x + 20, y, font)
        y += line
        draw_text(screen, f"D y                            : {D[1]:8.3f}", panel_x + 20, y, font)
        y += line
        draw_text(screen, f"distance AD                    : {feats['ad_distance']:8.3f}", panel_x + 20, y, font)
        y += line
        draw_text(screen, f"AB angle wrt -x                : {feats['ab_angle_neg_x_deg']:8.3f} deg", panel_x + 20, y, font)
        y += line
        draw_text(screen, f"ZF angle wrt -y                : {feats['zf_angle_neg_y_deg']:8.3f} deg", panel_x + 20, y, font)
        y += line
    else:
        draw_text(screen, "Pose invalid", panel_x + 20, y, font, RED)
        y += line

    draw_text(screen, f"Target x                       : {target_world[0]:8.3f}", panel_x + 20, y, font)
    y += line
    draw_text(screen, f"Target y                       : {target_world[1]:8.3f}", panel_x + 20, y, font)
    y += line
    draw_text(screen, f"IK error norm                  : {err:8.5f}", panel_x + 20, y, font)
    y += line

    reach_text = "YES" if reachable else "NO / approx"
    reach_color = GREEN if reachable else RED
    draw_text(screen, f"Target reachable               : {reach_text}", panel_x + 20, y, font, reach_color)
    y += line + 10

    draw_text(screen, f"Collected points               : {collected_count:8d}/{MAX_MANUAL_POINTS}", panel_x + 20, y, font, PURPLE)

    help_font = pygame.font.SysFont("arial", 18)
    draw_text(screen, "Controls:", panel_x + 20, panel_y + panel_h - 120, help_font)
    draw_text(screen, "Left click / drag -> manually sweep point D", panel_x + 20, panel_y + panel_h - 90, help_font)
    draw_text(screen, f"Auto-record valid points up to {MAX_MANUAL_POINTS}", panel_x + 20, panel_y + panel_h - 65, help_font)
    draw_text(screen, "R -> reset    ESC -> quit", panel_x + 20, panel_y + panel_h - 40, help_font)

# ============================================================
# Main
# ============================================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Interactive Crab Linkage IK - Manual Sweep")
    clock = pygame.time.Clock()

    current_guess = DEFAULT_GUESS.copy()

    init_joints = forward_kinematics(current_guess[0], current_guess[1])
    if init_joints is None:
        raise RuntimeError("Initial guess produces invalid geometry. Change DEFAULT_GUESS.")

    target_world = init_joints["D"].copy()

    dragging = False
    running = True

    collected_rows = []

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    current_guess = DEFAULT_GUESS.copy()
                    init_joints = forward_kinematics(current_guess[0], current_guess[1])
                    if init_joints is not None:
                        target_world = init_joints["D"].copy()
                    collected_rows = []
                    print("\nManual sweep data reset.\n")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    mx, my = pygame.mouse.get_pos()
                    target_world = np.array(screen_to_world(mx, my), dtype=float)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = pygame.mouse.get_pos()
                target_world = np.array(screen_to_world(mx, my), dtype=float)

        success, theta1, theta2, err = solve_ik_numerical(
            target_world[0],
            target_world[1],
            current_guess
        )

        current_guess = np.array([theta1, theta2], dtype=float)
        joints = forward_kinematics(theta1, theta2)
        motor1_deg, motor2_deg = joint_angles_to_motor_angles(theta1, theta2)

        # automatic manual-sweep data collection while dragging
        if dragging and success and joints is not None:
            if should_record_point(collected_rows, joints, MIN_POINT_SPACING):
                features = compute_pose_features(joints, theta1, theta2)
                features["target_x"] = target_world[0]
                features["target_y"] = target_world[1]
                features["ik_error"] = err
                collected_rows.append(features)

                print(
                    f"Recorded point {len(collected_rows):02d}/{MAX_MANUAL_POINTS} | "
                    f"D=({features['D_x']:.3f}, {features['D_y']:.3f}) | "
                    f"AD={features['ad_distance']:.3f} | "
                    f"AB<-x={features['ab_angle_neg_x_deg']:.3f} deg | "
                    f"ZF<-y={features['zf_angle_neg_y_deg']:.3f} deg"
                )

                if AUTO_CLOSE_ON_MAX_POINTS and len(collected_rows) >= MAX_MANUAL_POINTS:
                    print("\nReached maximum manual sweep points.")
                    running = False

        screen.fill(WHITE)
        draw_grid(screen)
        draw_mechanism(screen, joints, target_world, success)
        draw_info_panel(
            screen,
            theta1,
            theta2,
            motor1_deg,
            motor2_deg,
            joints,
            target_world,
            err,
            success,
            len(collected_rows)
        )
        pygame.display.flip()

    pygame.quit()

    print_sweep_table(collected_rows)
    save_rows_to_csv(collected_rows)
    plot_sweep_results(collected_rows)

if __name__ == "__main__":
    main()
