import numpy as np
from scipy.ndimage import rotate
import lines
import importlib

'''

This does the job of lines.py, except in the corotating frame (much simpler).

'''

#tested
def ROTATE(rho, x1, y1):
    angle = np.arctan2(y1,x1) * 180.0 / np.pi
    return rotate(rho, angle, reshape=False, order=3)

#tested
def minidisk_separator(N, length):
    return np.array([0.0 for i in range(N)]),\
           np.array([-0.5 * length + length/(N-1.0) * i for i in range(N)])

#tested
def circle(N, xc, yc, r):
    angles = [2.0 * np.pi / (N-1.0) * i for i in range(N)]
    return xc + r * np.cos(angles), yc + r * np.sin(angles)

#tested
def circle_unitnormal(x, y, xc, yc):
    xvec = x - xc
    yvec = y - yc
    mag  = np.sqrt(xvec**2 + yvec**2)
    return xvec/mag, yvec/mag

#tested
def project_along_bhs(vx, vy, x1, y1, x2, y2):
    xvec = x1 - x2
    yvec = y1 - y2
    mag  = np.sqrt(xvec**2 + yvec**2)
    return vx * xvec / mag + vy * yvec / mag

#tested on single snapshot.
#need to compare time series with corotating movie
def mdot_minidisk_separator(N, length, rho, vx, vy, x1, y1, x2, y2, xx, yy):
    momproj = project_along_bhs(rho*vx, rho*vy, x1, y1, x2, y2)
    mom  = ROTATE( momproj, x1, y1)
    del rho, vx, vy, momproj
    xmd, ymd = minidisk_separator(N, length)
    dx = length/(N-1.0)
    mom_md = np.array([lines.bilinear_interp(mom, xx, yy, xmd[i], ymd[i]) for i in range(N)])
    mom_md_p = mom_md*1
    mom_md_m = mom_md*1
    mom_md_p[np.where(mom_md_p<0)] = 0.0
    mom_md_m[np.where(mom_md_m>0)] = 0.0
    return np.trapz(mom_md_p, dx=dx), np.trapz(mom_md_m, dx=dx)

#needs testing, can't do it on a single snapshot.
#should produce a roughly time-lagged Mdot when computed
#around a single minidisk.
def mdot_circle(N, rho, vx, vy, xc, yc, r, xx, yy):
    xcirc, ycirc = circle(N, xc, yc, r)
    dx = 2*np.pi*r/(N-1.0)
    rho_circ = np.array([lines.bilinear_interp(rho, xx, yy, xcirc[i], ycirc[i]) for i in range(N)])
    vx_circ  = np.array([lines.bilinear_interp( vx, xx, yy, xcirc[i], ycirc[i]) for i in range(N)])
    vy_circ  = np.array([lines.bilinear_interp( vy, xx, yy, xcirc[i], ycirc[i]) for i in range(N)])
    momx_circ = rho_circ * vx_circ
    momy_circ = rho_circ * vy_circ
    momr_circ = momx_circ*0
    for i in range(N):
        rhatx, rhaty = circle_unitnormal(xcirc[i], ycirc[i], xc, yc)
        momr_circ[i] = momx_circ[i]*rhatx + momy_circ[i]*rhaty
    momr_circ_p = momx_circ*1
    momr_circ_m = momx_circ*1
    momr_circ_p[np.where(momr_circ_p<0)] = 0.0
    momr_circ_m[np.where(momr_circ_m>0)] = 0.0
    return np.trapz(momr_circ_p, dx=dx), np.trapz(momr_circ_m, dx=dx)