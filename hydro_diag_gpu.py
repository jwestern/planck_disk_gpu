import pdb
import numpy as np
import msgpack
import re
import matplotlib.pyplot as plt
try:
    import cupy as xp
except ImportError:
    import numpy as xp
from sailfish.kernel.library import Library
import corotate

def AsNumpy(x):
    if xp.__name__ == 'cupy':
        return xp.asnumpy(x)
    elif xp.__name__ == 'numpy':
        return x
    else:
        print("Error: xp must be cupy or numpy")

exec(open('constants3.py').read())

def reconstitute(filename, fieldnum, numfields):
    chkpt = msgpack.load(open(filename, 'rb'), raw=False)
    mesh = chkpt['mesh']
    cfl  = chkpt['command_line']['cfl_number']
    tnew = chkpt['time']
    x1new= chkpt['masses'][0]['x']
    y1new= chkpt['masses'][0]['y']
    x2new= chkpt['masses'][1]['x']
    y2new= chkpt['masses'][1]['y']
    primitive = xp.zeros([mesh['ni'], mesh['nj'], numfields])
    for patch in chkpt['primitive_patches']:
        i0 = patch['rect'][0]['start']
        j0 = patch['rect'][1]['start']
        i1 = patch['rect'][0]['end']
        j1 = patch['rect'][1]['end']
        local_prim = xp.array(np.frombuffer(patch['data'])).reshape([i1 - i0, j1 - j0, numfields])
        primitive[i0:i1, j0:j1] = local_prim
    rho, vx, vy, pres = primitive[:,:,0], primitive[:,:,1], primitive[:,:,2], primitive[:,:,3]
    if fieldnum==-1:
        par = re.search('gamma_law_index=(.+?):', chkpt['parameters'])
        if par==None:
            gamma = 1.666666666666666
        else:
            gamma = xp.float(par.group(1))
        return rho, vx, vy, pres, pres / rho / (gamma - 1.), cfl, tnew, x1new, y1new, x2new, y2new
    if fieldnum==0:
        return rho
    if fieldnum==1:
        return vx
    if fieldnum==2:
        return vy
    if fieldnum==3:
        return pres
    if fieldnum==4:
        return pres / rho / (gamma - 1.)
    if fieldnum==5:
        cs = xp.sqrt(gamma*pres/rho)
        v  = xp.sqrt(vx**2 + vy**2)
        return v/cs

fn        = '/scratch1/wester5/jwestern/sailfish/data/p/M11e0/1.5k/csoft/ii/up3/SS/fast/'
Nchkpts   = range(3250,3650,1) #range(3200,10000,1)
nstr      = str(np.char.zfill(str(Nchkpts[0]),4))
nstrs     = []
for i in range(len(Nchkpts)):
    nstrs.append(str(np.char.zfill(str(Nchkpts[i]),4)))
nstr      = nstrs[0]
#Nextra    = range(10000,13201,2)
#Nchkpts   = list(Nchkpts) + list(Nextra)
#for i in range(len(Nextra)):
#    nstrs.append(str(Nextra[i]))
print(nstr)
d         = msgpack.load(open(fn+'chkpt.'+nstr+'.sf','rb'), raw=False)
d['parameters'] = ':'+d['parameters']+':' #this allows parameter reads at beginning and end of the parameter string
DR        = xp.float(re.search('domain_radius=(.+?):', d['parameters']).group(1))
N         = d['mesh']['ni']
dx        = d['mesh']['dx']
rs        = xp.float(re.search('sink_radius=(.+?):', d['parameters']).group(1))
cfl       = d['command_line']['cfl_number']
alpha     = xp.float(re.search('alpha=(.+?):', d['parameters']).group(1))
gamma     = xp.float(re.search('gamma_law_index=(.+?):', d['parameters']).group(1))

mach_ceiling = xp.float(re.search('mach_ceiling=(.+?):', d['parameters']).group(1))
buffer_scale = 0.1
sink_rate    = xp.float(re.search('sink_rate=(.+?):', d['parameters']).group(1))
eccentricity = xp.float(re.search(':e=(.+?):', d['parameters']).group(1))
pressure_floor=xp.float(re.search('pressure_floor=(.+?):', d['command_line']['setup']).group(1))

### Corotate computation parameters
save_corotation_data = 1
mx,my = 0.08, 0.1
Nh = int(N/2)
Mx = int(mx*Nh)
My = int(my*Nh) #can zoom on on array by selecting [Nh-Mx:Nh+Mx,Nh-My:Nh+My]
###

x         = xp.arange((N))*dx - 2*DR/2. + dx/2.
xx,yy     = xp.zeros((N,N)),xp.zeros((N,N))
for i in range(N):
	xx[:,i] = x*1
	yy[i,:] = x*1
rr        = xp.sqrt(xx**2 + yy**2)
r         = xp.arange(int(N/2))*dx

a         = pc*0.0009691860239254802
M         = Msol*8*10**6
M1        = M/2
M2        = M/2

m0        = M*1.                 #code mass   unit in cgs
l0        = a*1.                 #code length unit in cgs
t0        = 1./xp.sqrt(G*M/a**3) #code time   unit in cgs

mp_code   = mp    / m0
kb_code   = kb    / (m0*l0**2/t0**2)
sigma_code= sigma / (m0/t0**3)
kappa_code= kappa / (l0**2/m0)
A         = (8./3) * sigma_code * (mp_code/kb_code)**4 * (gamma-1.)**4 / kappa_code
A_half    = A/2
rp_code   = (1-eccentricity)
rp_cgs    = rp_code*l0

time               = []
x1,y1,x2,y2        = [],[],[],[]
M1dot, M2dot       = [],[]

Mom_md_p, Mom_md_m = [],[]

for i in range(len(Nchkpts)):
    n         = Nchkpts[i]
    nstr      = nstrs[i]
    print('Reading checkpoint '+nstr)
    rho_code, vx_code, vy_code, pres_code, eps_code2, cfl, tnew, x1new, y1new, x2new, y2new = reconstitute(fn+'chkpt.'+nstr+'.sf',-1,4)
    time.append(tnew)
    x1.append( x1new)
    y1.append( y1new)
    x2.append( x2new)
    y2.append( y2new)
    pres_code[xp.where(pres_code<0)] = pressure_floor
    eps_code  = pres_code/rho_code/(gamma-1.)
    if xp.amax(abs(eps_code-eps_code2))!=0:
        print("Warning: eps does not agree with pres/rho/(gamma-1).")
    cs_code   = xp.sqrt(gamma*pres_code/rho_code)
    sigspeeds = xp.array([abs(vx_code) + cs_code, abs(vy_code) + cs_code])
    maxspeed  = xp.amax(sigspeeds)
    dt        = cfl*dx/maxspeed
    del eps_code2, cs_code, sigspeeds, maxspeed

    #Get mass flux between minidisks
    #mom_md_p, mom_md_m = corotate.mdot_minidisk_separator(int(1.0/dx), 1.0, AsNumpy(rho_code), AsNumpy(vx_code), AsNumpy(vy_code), x1[-1], y1[-1], x2[-1], y2[-1], AsNumpy(xx), AsNumpy(yy))
    mom_md_p, mom_md_m = corotate.mdot_minidisk_separator(int(1.0/dx), 1.0, rho_code, vx_code, vy_code, x1[-1], y1[-1], x2[-1], y2[-1], xx, yy)
    Mom_md_p.append(mom_md_p)
    Mom_md_m.append(mom_md_m)
    xp.save('Mom_md_p' ,xp.array(Mom_md_p))
    xp.save('Mom_md_m' ,xp.array(Mom_md_m))
    plt.figure(figsize=[10, 8])
    plt.subplot(211)
    plt.plot(np.array(time)/2/np.pi, np.array(Mom_md_p))
    plt.plot(np.array(time)/2/np.pi, np.array(Mom_md_m))
    plt.xlim(3250,3254)

    #Compute BH separation, for when e!=0
    d_bh = np.sqrt((x1[-1] - x2[-1])**2 + (y1[-1] - y2[-1])**2)

    #Put density into corotating frame
    #rho_rotated = corotate.ROTATE(AsNumpy(rho_code.T), x1[-1], y1[-1], 1)[Nh-Mx:Nh+Mx,Nh-My:Nh+My]
    rho_rotated = corotate.ROTATE(rho_code.T, x1[-1], y1[-1], 1)[Nh-Mx:Nh+Mx,Nh-My:Nh+My]
    xp.save(fn+'rho_rotated_'+str(n), rho_rotated)
    xmd, ymd = corotate.minidisk_separator(100, 1.0)
    xcirc1, ycirc1 = corotate.circle(100, -0.5, 0.0, 0.5)
    xcirc2, ycirc2 = corotate.circle(100,  0.5, 0.0, 0.5)
    xcirc3, ycirc3 = corotate.circle(100,  0.0, 0.0, 1.0)
    plt.subplot(212)
    plt.imshow(AsNumpy(rho_rotated)**0.25, origin='lower', cmap='plasma', extent=(-DR*my,DR*my,-DR*mx,DR*mx))
    plt.plot(AsNumpy(xmd), AsNumpy(ymd), color='white')
    #plt.plot(xcirc1, ycirc1, color='b')
    #plt.plot(xcirc2, ycirc2, color='r')
    #plt.plot(xcirc3, ycirc3, color='g')
    plt.show()
    plt.savefig("rho_"+str(n)+".png")
    plt.close()
    del rho_rotated

    del rho_code, vx_code, vy_code

    '''
    #Save stuff
    xp.save('lum_resolved_vis' ,xp.array(lum_resolved_vis ))
    xp.save('lum_resolved_inf' ,xp.array(lum_resolved_inf ))
    xp.save('time'             ,xp.array(time             ))
    xp.save('x1'               ,x1                         )
    xp.save('y1'               ,y1                         )
    xp.save('x2'               ,x2                         )
    xp.save('y2'               ,y2                         )
    xp.save('M1dot'            ,xp.array(M1dot            ))
    xp.save('M2dot'            ,xp.array(M2dot            ))

    xp.save('lum_BH1_vis' ,xp.array(lum_BH1_vis ))
    xp.save('lum_BH1_inf' ,xp.array(lum_BH1_inf ))

    xp.save('lum_BH2_vis' ,xp.array(lum_BH2_vis ))
    xp.save('lum_BH2_inf' ,xp.array(lum_BH2_inf ))

    xp.save(fn+                  'Teff_'+str(n),Teff                  )
    xp.save(fn+'lum_resolved_cell_vis_' +str(n),lum_resolved_cell_vis )
    xp.save(fn+'lum_resolved_cell_inf_' +str(n),lum_resolved_cell_inf )
    '''
