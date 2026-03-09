import math
import numpy


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

width, height = 800, 600
origin = (width//2, height//3)

# origin definition
Ax_origin = origin[0] - L1/2
Ay_origin = origin[1]
Dx_origin = origin[0] + L1/2
Dy_origin = origin[1]


def rotate_point(x, y, cx, cy, angle):
    s = math.sin(angle)
    c = math.cos(angle)
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



# --- Inverse Kinematics ---
def inverse_kinematics(Fx, Fy):

    best = float('NaN'),float('NaN')
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

        #avoid divide by 0
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


# --- Drawing function ---
def actualposition(theta_crank_relative, theta_black):

    Ax, Ay = Ax_origin, Ay_origin
    test = rotate_point(Dx_origin, Dy_origin, Ax_origin, Ay_origin, theta_black)
    Dx = test[0]
    Dy = test[1]
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

    # full red link endpoint F
    Fx = Dx + ux * L4
    Fy = Dy + uy * L4

    position =  Fx,Fy
    return position



#xbounds = 326,774
#ybounds = 272,590
xbounds = 0,1000
ybounds = 0,1000
width, height = xbounds[1]-xbounds[0], ybounds[1]-ybounds[0]
table_resolution = 50

Theta_Crank = numpy.zeros([table_resolution, table_resolution])
Theta_Main = numpy.zeros([table_resolution, table_resolution])
xs = numpy.zeros([table_resolution, table_resolution])
ys = numpy.zeros([table_resolution, table_resolution])
validyn = numpy.empty([table_resolution, table_resolution], dtype=str, order='C')

for i in range(table_resolution):
    for j in range(table_resolution):
        x = (xbounds[0]+(i*width/table_resolution))
        y = (ybounds[0]+(j*height/table_resolution))
        result = inverse_kinematics(x,y)
        theta_cr, theta_blk = result
        Theta_Crank[i][j] = theta_cr
        Theta_Main[i][j]= theta_blk
        x1, y1= actualposition(theta_cr,theta_blk)
        xs[i][j] = x
        ys[i][j] = y


        if abs(x-x1)>5:
            validyn[i][j] = '.'
        elif result[0] != result[0]:
            validyn[i][j] = '.'
        else:
            validyn[i][j] = '@'




numpy.savetxt("Theta_Crank.csv", Theta_Crank, delimiter=",")
numpy.savetxt("Theta_Main.csv", Theta_Main, delimiter=",")
numpy.savetxt("xs.csv", xs, delimiter=",")
numpy.savetxt("ys.csv", ys, delimiter=",")
numpy.savetxt("valid.txt", validyn,fmt='%s', delimiter=" ")
print("done")