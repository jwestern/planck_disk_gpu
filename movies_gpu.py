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
import lump
from scipy.ndimage import gaussian_filter1d

def AsNumpy(x):
    if xp.__name__ == 'cupy':
        return xp.asnumpy(x)
    elif xp.__name__ == 'numpy':
        return x
    else:
        print("Error: xp must be cupy or numpy")

exec(open('/home/wester5/jwestern/planck_disk_gpu/constants3.py').read())

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

PLOT_WHOLE_SYSTEM_DENSITY = False
PLOT_LUMP_PHASE           = False
PLOT_COROTATE_RHO         = False
PLOT_MD_LUMINOSITY_TIMING = True

fn        = '/scratch1/wester5/jwestern/sailfish/data/p/M11e0/1.5k/csoft/ii/up3/SSeb/fast/'
Nchkpts   = range(3250,4251,2) #range(3250,8251,5) #range(3250,4251,1) #range(3200,10000,1) 
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
mx,my = 0.028125, 0.11
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

timeh              = np.load('time.npy')
x1,y1,x2,y2        = np.load('x1.npy'), np.load('y1.npy'), np.load('x2.npy'), np.load('y2.npy')
M1dot, M2dot       = np.load('../M1dot.npy'), np.load('../M2dot.npy')

Mdot_md_p  , Mdot_md_m   = np.load('Mdot_md_p.npy'),  np.load('Mdot_md_m.npy')
Mdot_r08_p , Mdot_r08_m  = np.load('Mdot_r08_p.npy'), np.load('Mdot_r08_p.npy')
Lump_amp, Lump_phase = np.load('Lump_amp.npy'), np.load('Lump_phase.npy')
lump_mask = rr**0
lump_mask[xp.where(rr<1.5)]= 0.0
lump_mask[xp.where(rr>10)]= 0.0
lump_mask = AsNumpy(lump_mask)

lum_total_vis = np.load('../lum_total_vis.npy')
lum_BH1_vis   = np.load('../lum_BH1_vis.npy')
lum_BH2_vis   = np.load('../lum_BH2_vis.npy')
M1dot         = np.load('../M1dot.npy')
M2dot         = np.load('../M2dot.npy')
time          = np.load('../time.npy')

M1dotp = M1dot - gaussian_filter1d(M1dot,300)
M1dotp = M1dotp/np.sqrt(np.average(M1dotp**2))

M2dotp = M2dot - gaussian_filter1d(M2dot,300)
M2dotp = M2dotp/np.sqrt(np.average(M2dotp**2))

for i in range(len(Nchkpts)):
    n         = Nchkpts[i]
    nstr      = nstrs[i]

    plt.figure(figsize=[10, 5.626]) #These dimensions allow upsampling by ffmpeg,
                                    #and are approximately the dimensions of a Google slide.
    
    if PLOT_COROTATE_RHO:
        print('Reading data '+nstr)
        rho_code  = np.load(fn+'rho_'+str(n)+'.npy')

        #Plot mass fluxes
        plt.subplot(211)
        plt.subplots_adjust(left=0.01, right=1-0.01, top=1-0.01, bottom=0.01)
        plt.plot((timeh-timeh[0])/2/np.pi, Mdot_md_p /np.average(Mdot_md_p)  , 'b', label='mass flux from left to right')
        plt.plot((timeh-timeh[0])/2/np.pi,-Mdot_r08_m/np.average(-Mdot_r08_m), 'r', label='mass flux into r=0.8*a')
        plt.plot((timeh[i]-timeh[0])/2/np.pi*np.ones((2)), [0,7.2],'k--')
        plt.ylim(0,7.2)
        plt.yticks([])
        plt.xticks(range(0,11))
        plt.grid(axis='x')
        plt.xlim(0,10)
        plt.xlabel('time [orbits]')
        plt.legend(loc=1)

        #imshow corotating minidisks
        #Put density into corotating frame
        rho_rotated = corotate.ROTATE(xp.asarray(rho_code).T, x1[i], y1[i], 1)[Nh-Mx:Nh+Mx,Nh-My:Nh+My]
        xmd, ymd = corotate.minidisk_separator(100, 1.0)
        #xcirc1, ycirc1 = corotate.circle(100, -0.5, 0.0, 0.5)
        #xcirc2, ycirc2 = corotate.circle(100,  0.5, 0.0, 0.5)
        xcirc3, ycirc3 = corotate.circle(100,  0.0, 0.0, 0.8)
        plt.subplot(212)
        plt.imshow(AsNumpy(rho_rotated)**0.5, origin='lower', cmap='inferno', extent=(-DR*my,DR*my,-DR*mx,DR*mx))
        plt.plot(AsNumpy(xmd), AsNumpy(ymd), color='b')
        #plt.plot(xcirc1, ycirc1, color='b')
        #plt.plot(xcirc2, ycirc2, color='r')
        plt.plot(AsNumpy(xcirc3), AsNumpy(ycirc3), color='r')
        plt.ylim(-mx*DR, mx*DR)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.savefig("corotate_rho_"+str(i+1000)+".png",dpi=400)
        plt.close()
        del rho_rotated
        del rho_code
    
    if PLOT_LUMP_PHASE:
        print('Reading data '+nstr)
        rho_code  = np.load(fn+'rho_'+str(n)+'.npy')
        #Plot lump phase
        plt.subplot(211)
        plt.subplots_adjust(left=0.075, right=1-0.01, top=1-0.01, bottom=0.01)
        plt.plot((timeh-timeh[0])/2/np.pi, Lump_phase,'k',label='lump phase (m=1 component of density)')
        plt.plot((timeh[int(i*5)]-timeh[0])/2/np.pi*np.ones((2)), [0.053,0.256],'k--')
        plt.ylim(0.053,0.256)
        plt.xticks(range(0,51,10))
        plt.xlim(0,50)
        plt.grid(axis='x')
        plt.xlabel('time [orbits]')
        plt.ylabel('phase [radians]')
        plt.legend(loc=2)

        #imshow lump
        plt.subplot(212)
        mlump= 0.553
        rho2 = (rho_code*lump_mask).T[4180:7500,:]*1
        plt.imshow(AsNumpy(rho2), origin='lower', cmap='inferno', extent=(-DR,DR,-4.5,3.8),vmax=0.356,vmin=0.0)
        xcirc_small, ycirc_small = corotate.circle(100,  0.0, 0.0, 1.5)
        xcirc_big,   ycirc_big   = corotate.circle(100,  0.0, 0.0, 10.)
        plt.plot(AsNumpy(xcirc_small), AsNumpy(ycirc_small), lw=1, color='white')
        plt.plot(AsNumpy(xcirc_big),   AsNumpy(ycirc_big),   lw=1, color='white')
        plt.plot(x1[int(i*5)], y1[int(i*5)], 'ro')
        plt.plot(x2[int(i*5)], y2[int(i*5)], 'ro')
        plt.xlim(-15,15)
        plt.ylim(-4.5,3.8)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.savefig("lump_"+str(i+1000)+".png",dpi=400)
        plt.close()
        del rho2
        del rho_code
    
    if PLOT_WHOLE_SYSTEM_DENSITY:
        print('Reading data '+nstr)
        rho_code  = np.load(fn+'rho_'+str(n)+'.npy')
        #Plot whole system density
        plt.subplot(111)
        plt.imshow(AsNumpy(rho_code.T[4180:7500,3800:9702]**0.4), origin='lower', cmap='inferno',vmax=2.05,vmin=0.0)
        #print(xp.amax(rho_code.T**0.4))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
        plt.savefig("rho_"+str(i+1000)+".png", dpi=400)
        plt.close()
        del rho_code
    
    if PLOT_MD_LUMINOSITY_TIMING:
        print('Reading data '+nstr)
        lum  = np.load(fn+'lum_resolved_cell_vis_'+str(n)+'.npy')
        #Teff = np.load(fn+'Teff_'+str(n)+'.npy')
        #print(np.amax(Teff), np.amin(Teff))
        #
        plt.subplot(211)
        lum_plot = lum_total_vis
        #lum_plot = lum_plot/np.sqrt(np.average(lum_plot**2))
        plt.subplots_adjust(left=0.065, right=1-0.01, top=1-0.05, bottom=0.005)
        plt.plot((time-time[0])/2/np.pi, lum_plot, 'b', label='thermal optical emission')
        #plt.plot((time-time[0])/2/np.pi, lum_BH1_vis, 'r', label='BH1')
        #plt.plot((time-time[0])/2/np.pi, lum_BH2_vis, 'k', label='BH2')
        #up = np.amax(lum_BH1_vis)*1.025
        #down = np.amin(lum_BH1_vis)*(1-0.025)
        up = np.amax(lum_plot)*1.025
        down = np.amin(lum_plot)*(1-0.025)
        plt.plot((time[i]-time[0])/2/np.pi*np.ones((2)), [down,up],'k--')
        plt.ylim(down,up)
        plt.xticks(range(0,11,1))
        plt.xlim(0,10)
        plt.grid(axis='x')
        plt.xlabel('time [orbits]')
        plt.ylabel('luminosity [erg/s]')
        plt.legend(loc=2)

        #imshow corotating minidisks
        #Put luminosity map into corotating frame
        lum_rotated = corotate.ROTATE(xp.asarray(lum).T, x1[int(i*2)], y1[int(i*2)], 1)[Nh-Mx:Nh+Mx,Nh-My:Nh+My]
        plt.subplot(212)
        lum_max = 1e13
        lum_min = 0.0
        plt.imshow(AsNumpy(lum_rotated), origin='lower', cmap='inferno', extent=(-DR*my,DR*my,-DR*mx,DR*mx), vmin=lum_min, vmax=lum_max)
        plt.ylim(-mx*DR, mx*DR)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.savefig("corotate_lum_vis_"+str(i+1000)+".png",dpi=400)
        plt.close()
        del lum_rotated
        del lum
