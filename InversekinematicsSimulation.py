import tkinter as tk
import math

# --- Scale factor ---
multiplier = 30

# --- Link lengths ---
L1 = 7.0 * multiplier
L2 = 1.5 * multiplier
L3 = 3.946 * multiplier
L4 = 10 * multiplier

# --- Adjustable parameters ---
L_DC = 2.25 * multiplier
offset_AD_to_E = L2
pos_E_on_AD = 0.5
C_offset = 0.25 * multiplier

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


# --- Drawing function ---
def draw_crank_rocker(theta_crank_relative, theta_black):

    Ax, Ay = Ax_origin, Ay_origin
    Dx, Dy = rotate_point(Dx_origin, Dy_origin, Ax_origin, Ay_origin, theta_black)

    # E point
    Ex = Ax + (Dx - Ax) * pos_E_on_AD
    Ey = Ay + (Dy - Ay) * pos_E_on_AD + offset_AD_to_E

    # crank
    theta_crank_abs = theta_black + theta_crank_relative
    Bx = Ex + L2 * math.cos(theta_crank_abs)
    By = Ey + L2 * math.sin(theta_crank_abs)

    # solve C_base (true joint)
    points = circle_circle_intersection(Bx, By, L3, Dx, Dy, L_DC)

    if points is None:
        Cx_base, Cy_base = Dx, Dy + L_DC
    else:
        if points[0][1] > Dy:
            Cx_base, Cy_base = points[0]
        else:
            Cx_base, Cy_base = points[1]

    # DF direction
    vx = Cx_base - Dx
    vy = Cy_base - Dy
    dist = math.hypot(vx, vy)

    ux = vx / dist
    uy = vy / dist

    # perpendicular right
    px = uy
    py = -ux

    # visible C
    Cx = Cx_base + px * C_offset
    Cy = Cy_base + py * C_offset

    # full red link endpoint F
    Fx = Dx + ux * L4
    Fy = Dy + uy * L4

    # --- Clear only mechanism lines, keep trail ---
    canvas.delete("mechanism")

    # real mechanism
    canvas.create_line(Ex, Ey, Bx, By, fill="blue", width=3, tags="mechanism")
    canvas.create_line(Bx, By, Cx, Cy, fill="green", width=3, tags="mechanism")
    canvas.create_line(Dx, Dy, Cx, Cy, fill="red", width=3, tags="mechanism")
    canvas.create_line(Cx, Cy, Fx, Fy, fill="red", width=3, tags="mechanism")
    canvas.create_line(Ax, Ay, Dx, Dy, fill="black", width=3, tags="mechanism")

    # visual-only structure
    canvas.create_line(Ax, Ay, Ex, Ey, dash=(4,2), fill="gray", tags="mechanism")
    canvas.create_line(Ex, Ey, Dx, Dy, dash=(4,2), fill="gray", tags="mechanism")
    canvas.create_line(Dx, Dy, Fx, Fy, dash=(4,2), fill="gray", tags="mechanism")

    joints = {
        "A": (Ax, Ay),
        "B": (Bx, By),
        "C": (Cx, Cy),
        "D": (Dx, Dy),
        "E": (Ex, Ey),
        "F": (Fx, Fy)
    }

    for label, point in joints.items():
        x, y = point
        canvas.create_oval(x-5, y-5, x+5, y+5, fill="orange", tags="mechanism")
        canvas.create_text(x + 10, y - 10, text=label, font=("Arial", 12, "bold"), tags="mechanism")

    # --- Draw persistent dot for F trail ---
    r = 3
    canvas.create_oval(Fx - r, Fy - r, Fx + r, Fy + r, fill="red", outline="", tags="trail")


# --- Inverse Kinematics ---
def inverse_kinematics(Fx, Fy):

    best = None
    best_dist = float('inf')

    min_theta_black = -math.pi/8
    max_theta_black = math.pi/8

    steps = 40

    for i in range(steps + 1):

        theta_black = min_theta_black + (max_theta_black-min_theta_black)*i/steps

        Ax, Ay = Ax_origin, Ay_origin
        Dx, Dy = rotate_point(Dx_origin, Dy_origin, Ax_origin, Ay_origin, theta_black)

        vx = Fx - Dx
        vy = Fy - Dy
        dist = math.hypot(vx, vy)

        if dist == 0:
            continue

        ux = vx / dist
        uy = vy / dist

        # true pivot on DF
        Cx_base = Dx + ux * L_DC
        Cy_base = Dy + uy * L_DC

        # perpendicular offset
        px = uy
        py = -ux

        # use the true pivot for IK
        Cx = Cx_base
        Cy = Cy_base

        Ex = Ax + (Dx-Ax)*pos_E_on_AD
        Ey = Ay + (Dy-Ay)*pos_E_on_AD + offset_AD_to_E

        points = circle_circle_intersection(Cx, Cy, L3, Ex, Ey, L2)

        if points is None:
            continue

        for Bx, By in points:

            theta_crank_abs = math.atan2(By-Ey, Bx-Ex)
            theta_crank_rel = theta_crank_abs - theta_black

            if math.pi/6 <= theta_crank_rel <= math.pi:

                Fx_est = Dx + ux * L4
                Fy_est = Dy + uy * L4

                dist_err = math.hypot(Fx - Fx_est, Fy - Fy_est)

                if dist_err < best_dist:
                    best_dist = dist_err
                    best = (theta_crank_rel, theta_black)

    return best


# --- Mouse tracking ---
mouse_pos = (origin[0], origin[1])


def on_mouse_move(event):
    global mouse_pos
    mouse_pos = (event.x, event.y)


canvas.bind("<Motion>", on_mouse_move)


# --- Animation ---
def animate():
    Fx, Fy = mouse_pos

    result = inverse_kinematics(Fx, Fy)

    if result:
        theta_cr, theta_blk = result
        draw_crank_rocker(theta_cr, theta_blk)

        # --- Draw a small dot at the tip (F) for the trail ---
        # Get the current F position from the last draw
        Ax, Ay = Ax_origin, Ay_origin
        Dx, Dy = rotate_point(Dx_origin, Dy_origin, Ax_origin, Ay_origin, theta_blk)

        # direction of DF
        vx = Dx - Dx  # not used for F position here
        # compute DF direction from draw_crank_rocker logic
        # we can just compute F
        points = circle_circle_intersection(
            Ax + (Dx-Ax)*pos_E_on_AD + L2 * math.cos(theta_blk + theta_cr), 
            Ay + (Dy-Ay)*pos_E_on_AD + offset_AD_to_E + L2 * math.sin(theta_blk + theta_cr),
            L3, Dx, Dy, L_DC
        )

        if points is None:
            Cx_base, Cy_base = Dx, Dy + L_DC
        else:
            if points[0][1] > Dy:
                Cx_base, Cy_base = points[0]
            else:
                Cx_base, Cy_base = points[1]

        # DF direction
        vx = Cx_base - Dx
        vy = Cy_base - Dy
        dist = math.hypot(vx, vy)
        ux = vx / dist
        uy = vy / dist

        Fx_pos = Dx + ux * L4
        Fy_pos = Dy + uy * L4

        # Draw small dot for trail
        r = 3  # radius
        canvas.create_oval(Fx_pos - r, Fy_pos - r, Fx_pos + r, Fy_pos + r, fill="red", outline="")

        # --- Print values relative to A with down negative ---
        Fx_rel = (Fx - Ax_origin) / multiplier
        Fy_rel = (Ay_origin - Fy) / multiplier
        print(
            f"Mouse relative to A: ({Fx_rel:.3f}, {Fy_rel:.3f})   "
            f"Theta crank relative: {theta_cr:.4f} rad   "
            f"Theta black: {theta_blk:.4f} rad"
        )

    # --- Draw scale bar at bottom ---
    scale_length = 1 * multiplier  # 1 inch
    scale_x = 50
    scale_y = height - 50

    canvas.create_line(scale_x, scale_y, scale_x + scale_length, scale_y, fill="gray", width=3)
    canvas.create_line(scale_x, scale_y-5, scale_x, scale_y+5, fill="gray", width=2)
    canvas.create_line(scale_x + scale_length, scale_y-5, scale_x + scale_length, scale_y+5, fill="gray", width=2)
    canvas.create_text(scale_x + scale_length/2, scale_y + 15, text="1 in", font=("Arial", 12))

    window.after(20, animate)


animate()
window.mainloop()
