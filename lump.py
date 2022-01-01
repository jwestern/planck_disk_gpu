import numpy as np
try:
    import cupy as xp
except ImportError:
    import numpy as xp
import importlib

'''

Lump diagnostics.

'''

def AsNumpy(x):
    if xp.__name__ == 'cupy':
        return xp.asnumpy(x)
    elif xp.__name__ == 'numpy':
        return x
    else:
        print("Error: xp must be cupy or numpy")

def lump_weight(rho, xx, yy, r1, r2, dx):
    theta = xp.arctan2(yy, xx)
    wreal = np.trapz(np.trapz( AsNumpy(rho*xp.cos(theta)), dx=dx), dx=dx)
    wimag = np.trapz(np.trapz( AsNumpy(rho*xp.sin(theta)), dx=dx), dx=dx)
    wmag  = np.sqrt(wreal**2 + wimag**2)
    wphase= np.arctan2(wimag, wreal)
    return wmag, wphase
