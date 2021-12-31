try:
    import cupy as xp
except ImportError:
    import numpy as xp
try:
    from cupyx.scipy.ndimage import rotate
except ImportError:
    from scipy.ndimage import rotate
import lines
import importlib

'''

This does the job of lines.py, except in the corotating frame (much simpler).

'''

#tested. Must pass transposed rho.
def ROTATE(rho, x1, y1, Order):
    angle = xp.arctan2(y1,x1) * 180.0 / xp.pi
    return rotate(rho, angle, reshape=False, order=Order)

#tested
def minidisk_separator(N, length):
    return xp.array([0.0 for i in range(N)]),\
           xp.array([-0.5 * length + length/(N-1.0) * i for i in range(N)])

#tested
def circle(N, xc, yc, r):
    angles = [2.0 * xp.pi / (N-1.0) * i for i in range(N)]
    return xc + r * xp.cos(angles), yc + r * xp.sin(angles)

#tested
def circle_unitnormal(x, y, xc, yc):
    xvec = x - xc
    yvec = y - yc
    mag  = xp.sqrt(xvec**2 + yvec**2)
    return xvec/mag, yvec/mag

def RHAT(xx, yy, xc, yc):
    xvec = xx - xc
    yvec = yy - yc
    mag  = xp.sqrt(xvec**2 + yvec**2)
    return xvec/mag, yvec/mag

def THETAHAT(xx, yy, xc, yc):
    xx2, yy2 = xx - xc, yy - yc
    xvec = -yy2
    yvec = xx2
    mag = xp.sqrt(xvec**2 + yvec**2)
    return xvec/mag, yvec/mag

def v_transform_to_corotating_frame(vx, vy, xx, yy):
    rr = xp.sqrt(xx**2 + yy**2)
    rhatx, rhaty = RHAT(xx, yy, 0.0, 0.0)
    thetahatx, thetahaty = THETAHAT(xx, yy, 0.0, 0.0)
    vr = vx * rhatx + vy * rhaty
    vtheta = vx * thetahatx + vy * thetahaty
    vtheta = vtheta - 1.0
    xhatr, xhattheta = xx / rr, -yy / rr
    yhatr, yhattheta = yy / rr,  xx / rr
    return vr * xhatr + vtheta * xhattheta, vr * yhatr + vtheta * yhattheta

def project_along_rhat(vx, vy, xx, yy, xc, yc):
    rhatx, rhaty = RHAT(xx, yy, xc, yc)
    return vx * rhatx + vy * rhaty

#tested
def project_along_bhs(vx, vy, x1, y1, x2, y2):
    xvec = x1 - x2
    yvec = y1 - y2
    mag  = xp.sqrt(xvec**2 + yvec**2)
    return vx * xvec / mag + vy * yvec / mag

#need to transform velocities into corotating frame
def mdot_minidisk_separator(N, length, rho, vx, vy, x1, y1, x2, y2, xx, yy):
    vx, vy = v_transform_to_corotating_frame(vx, vy, xx, yy)
    momproj = project_along_bhs(rho*vx, rho*vy, x1, y1, x2, y2)
    mom  = ROTATE( momproj.T, x1, y1, 3)
    del rho, vx, vy, momproj
    xmd, ymd = minidisk_separator(N, length)
    dx = length/(N-1.0)
    mom_md = xp.array([lines.bilinear_interp(mom, xx, yy, xmd[i], ymd[i]) for i in range(N)])
    mom_md_p = mom_md*1
    mom_md_m = mom_md*1
    mom_md_p[xp.where(mom_md_p<0)] = 0.0
    mom_md_m[xp.where(mom_md_m>0)] = 0.0
    return xp.trapz(mom_md_p, dx=dx), xp.trapz(mom_md_m, dx=dx)

def mdot_circle(N, rho, vx, vy, xc, yc, xc2, yc2, r, xx, yy, x1, y1):
    dx = 2*xp.pi*r/(N-1.0)
    vx, vy = v_transform_to_corotating_frame(vx, vy, xx, yy)
    momproj = project_along_rhat(rho*vx, rho*vy, xx, yy, xc, yc)
    mom = ROTATE( momproj.T, x1, y1, 3)
    del rho, vx, vy, momproj
    xcirc, ycirc = circle(N, xc2, yc2, r)
    mom_circ = xp.array([lines.bilinear_interp(mom, xx, yy, xcirc[i], ycirc[i]) for i in range(N)])
    mom_circ_p = mom_circ*1
    mom_circ_m = mom_circ*1
    mom_circ_p[xp.where(mom_circ_p<0)] = 0.0
    mom_circ_m[xp.where(mom_circ_m>0)] = 0.0
    return xp.trapz(mom_circ_p, dx=dx), xp.trapz(mom_circ_m, dx=dx)
