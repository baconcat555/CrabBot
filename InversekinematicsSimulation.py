import tkinter as tk
import math
# I'm sad to say I used the lying machine for this only god truly understands the inner workings of this code
# --- Scale factor ---
multiplier = 30

# --- Link lengths (scaled) ---
L1 = 7.0 * multiplier
L2 = 1.2 * multiplier
L3 = 3.939 * multiplier
L4 = 9.151 * multiplier

# --- Adjustable parameters ---
L_DC = 4.5 * multiplier
offset_AD_to_E = L2
pos_E_on_AD = 0.5

# canvas
width, height = 800, 600
origin = (width//2, height//3)

window = tk.Tk()
window.title("Inverse Kinematics Test")
canvas = tk.Canvas(window, width=width, height=height, bg="white")
canvas.pack()

# origin definition
Ax_origin = origin[0] - L1/2
Ay_origin = origin[1]
Dx_origin = origin[0] + L1/2
Dy_origin = origin[1]

def rotate_point(x, y, cx, cy, angle):
    s, c = math.sin(angle), math.cos(angle)
    x -= cx
    y -= cy
    xr = x*c - y*s
    yr = x*s + y*c
    return xr + cx, yr + cy


# I stole this so hard
def circle_circle_intersection(x0, y0, r0, x1, y1, r1):
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d > r0 + r1 or d < abs(r0 - r1):
        return None
    a = (r0**2 - r1**2 + d**2) / (2*d)
    h = math.sqrt(max(r0**2 - a**2, 0))
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    rx = -dy * (h / d)
    ry = dx * (h / d)
    return (xm + rx, ym + ry), (xm - rx, ym - ry)

# Drawing function
def draw_crank_rocker(theta_crank_relative, theta_black):
    Ax, Ay = Ax_origin, Ay_origin
    Dx, Dy = rotate_point(Dx_origin, Dy_origin, Ax_origin, Ay_origin, theta_black)

    Ex = Ax + (Dx - Ax) * pos_E_on_AD
    Ey = Ay + (Dy - Ay) * pos_E_on_AD + offset_AD_to_E

    theta_crank_abs = theta_black + theta_crank_relative
    Bx = Ex + L2 * math.cos(theta_crank_abs)
    By = Ey + L2 * math.sin(theta_crank_abs)

    points = circle_circle_intersection(Bx, By, L3, Dx, Dy, L_DC)
    if points is None:
        Cx_mid, Cy_mid = Dx, Dy + L_DC
    else:
        if points[0][1] > Dy:
            Cx_mid, Cy_mid = points[0]
        else:
            Cx_mid, Cy_mid = points[1]

    L_CF = L4 - L_DC
    Cx_full = Cx_mid + (Cx_mid - Dx) * (L_CF / L_DC)
    Cy_full = Cy_mid + (Cy_mid - Dy) * (L_CF / L_DC)

    canvas.delete("all")
    canvas.create_line(Ex, Ey, Bx, By, fill="blue", width=3)
    canvas.create_line(Bx, By, Cx_mid, Cy_mid, fill="green", width=3)
    canvas.create_line(Dx, Dy, Cx_mid, Cy_mid, fill="red", width=3)
    canvas.create_line(Cx_mid, Cy_mid, Cx_full, Cy_full, fill="red", width=3)
    canvas.create_line(Ax, Ay, Dx, Dy, fill="black", width=3)

    joints = {"A": (Ax, Ay), "B": (Bx, By), "C": (Cx_mid, Cy_mid),
              "D": (Dx, Dy), "E": (Ex, Ey), "F": (Cx_full, Cy_full)}
    for label, point in joints.items():
        x, y = point
        canvas.create_oval(x-5, y-5, x+5, y+5, fill="orange")
        canvas.create_text(x + 10, y - 10, text=label, font=("Arial", 12, "bold"))

# Inverse Kinematics
def inverse_kinematics(Fx, Fy):

    best = None
    best_dist = float('inf')

    # black link limits
    min_theta_black = -math.pi/8
    max_theta_black = math.pi/8

    steps = 40

    for i in range(steps+1):

        theta_black = min_theta_black + (max_theta_black-min_theta_black)*i/steps

        Ax, Ay = Ax_origin, Ay_origin
        Dx, Dy = rotate_point(Dx_origin, Dy_origin, Ax_origin, Ay_origin, theta_black)

        # Compute C_mid from F
        vx = Fx - Dx
        vy = Fy - Dy
        dist = math.hypot(vx, vy)

        if dist == 0:
            continue

        scale = L_DC / dist
        Cx = Dx + vx * scale
        Cy = Dy + vy * scale

        # Solve B
        points = circle_circle_intersection(Cx, Cy, L3,
                                            Ax + (Dx-Ax)*pos_E_on_AD,
                                            Ay + (Dy-Ay)*pos_E_on_AD + offset_AD_to_E,
                                            L2)

        if points is None:
            continue

        for Bx, By in points:

            Ex = Ax + (Dx-Ax)*pos_E_on_AD
            Ey = Ay + (Dy-Ay)*pos_E_on_AD + offset_AD_to_E

            theta_crank_abs = math.atan2(By-Ey, Bx-Ex)
            theta_crank_rel = theta_crank_abs - theta_black

            if math.pi/6 <= theta_crank_rel <= math.pi:

                dist_err = math.hypot(Fx-(Dx + (Cx-Dx)*(L4/L_DC)),
                                      Fy-(Dy + (Cy-Dy)*(L4/L_DC)))

                if dist_err < best_dist:
                    best_dist = dist_err
                    best = (theta_crank_rel, theta_black)

    return best

# Mouse tracking
mouse_pos = (origin[0], origin[1])

def on_mouse_move(event):
    global mouse_pos
    mouse_pos = (event.x, event.y)

canvas.bind("<Motion>", on_mouse_move)

# Animation
def animate():

    Fx, Fy = mouse_pos

    result = inverse_kinematics(Fx, Fy)

    if result:
        theta_cr, theta_blk = result

        draw_crank_rocker(theta_cr, theta_blk)

        # --- Print values ---
        print(
            f"Mouse: ({Fx/multiplier:.3f}, {Fy/multiplier:.3f})   "
            f"Theta crank relative: {theta_cr:.4f} rad   "
            f"Theta black: {theta_blk:.4f} rad"
        )

    window.after(20, animate)

animate()
window.mainloop()
