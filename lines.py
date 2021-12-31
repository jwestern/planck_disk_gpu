try:
    import cupy as xp
except ImportError:
    import numpy as xp

'''
Note: when testing with matplotlib plots, use

imshow(rho, extent=(-DR,DR,-DR,DR), origin='bottomleft')

otherwise the x,y coordinates will be messed up.

Tests used coordinate arrays defined via

x     = xp.arange((N))*dx - 2*DR/2. + dx/2.
xx,yy = xp.zeros((N,N)),xp.zeros((N,N))
for i in range(N):
    xx[:,i] = x*1
    yy[i,:] = x*1

where DR is the domain radius, dx is resolution, N is grid dimension.

'''

def unitvec(x, y):
    mag = xp.sqrt(x**2 + y**2)
    return x/mag, y/mag

def rotate90(x, y):
    return -y, x

def bh_separation_unitvec(x1, y1, x2, y2):
    return unitvec(x2-x1, y2-y1)

#tested
def bh_separator(N, x1, y1, x2, y2, length):
    dx = length / N
    xhat, yhat = bh_separation_unitvec(x1, y1, x2, y2)
    xsep, ysep = [], []
    for i in range(N+1):
        xsep.append((-0.5 * length + i * dx) * xhat)
        ysep.append((-0.5 * length + i * dx) * yhat)
    return xsep, ysep

def minidisk_separator_unitvec(x1, y1, x2, y2):
    vec = bh_separation_unitvec(x1, y1, x2, y2)
    return rotate90(vec[0], vec[1])

#tested
def minidisk_separator(N, x1, y1, x2, y2, length):
    dx = length / N
    xhat, yhat = minidisk_separator_unitvec(x1, y1, x2, y2)
    xsep, ysep = [], []
    for i in range(N+1):
        xsep.append((-0.5 * length + i * dx) * xhat)
        ysep.append((-0.5 * length + i * dx) * yhat)
    return xsep, ysep

def semicircle(N, x1, y1, radius, theta_start, theta_stop):
    x, y = [], []
    dtheta = (theta_stop - theta_start) / N
    for i in range(N+1):
        x.append(x1 + radius * xp.cos(dtheta * i))
        y.append(y1 + radius * xp.sin(dtheta * i))
    return x, y

#tested
def circle(N, x1, y1, radius):
    return semicircle(N, x1, y1, radius, 0.0, 2*xp.pi)

def outward_unitnormal_vec_circle(x, y, x1, y1):
    return unitvec(x - x1, y - y1)

#tested
def bilinear_interp(data, xx, yy, x, y):
    x, y = y, x #need to swap these, don't know why
    il = xp.where(xx[:,0] < x)[0][-1]
    ih = xp.where(xx[:,0] > x)[0][0]
    jl = xp.where(yy[0,:] < y)[0][-1]
    jh = xp.where(yy[0,:] > y)[0][0]
    ll = (x - xx[il,jl]) * (y - yy[il,jl])
    hh = (xx[ih,jh] - x) * (yy[ih,jh] - y)
    lh = (x - xx[il,jh]) * (yy[il,jh] - y)
    hl = (xx[ih,jl] - x) * (y - yy[ih,jl])
    return data[il,jl]*hh + data[ih,jh]*ll + data[il,jh]*hl + data[ih,jl]*lh

def project_normal_to_minidisk_separator(vx, vy, x1, y1, x2, y2):
    xhat, yhat = bh_separation_unitvec(x1, y1, x2, y2)
    return xhat * vx + yhat * vy

def project_normal_to_circle(vx, vy, x, y, x1, y1):
    xhat, yhat = outward_unitnormal_vec_circle(x, y, x1, y1)
    return xhat * vx, yhat * vy
