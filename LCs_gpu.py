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

#Bands in units of centimeters
m_to_cm   = 1e2
cm_x      = [ 10e-9 * m_to_cm      ]
cm_UV     = [ 10e-9 * m_to_cm, 400e-9 * m_to_cm]
cm_vis    = [400e-9 * m_to_cm, 700e-9 * m_to_cm]
cm_inf    = [700e-9 * m_to_cm,   1e-3 * m_to_cm]
cm_mw     = [  1e-3 * m_to_cm,   1e-1 * m_to_cm]

fn        = '/scratch1/wester5/jwestern/sailfish/data/p/M11e0/1.5k/csoft/ii/up2/SS/fast/'
Nchkpts   = range(3200,10000,2)
nstr      = str(np.char.zfill(str(Nchkpts[0]),4))
nstrs     = []
for i in range(len(Nchkpts)):
    nstrs.append(str(np.char.zfill(str(Nchkpts[i]),4)))
nstr      = nstrs[0]
Nextra    = range(10000,13201,2)
Nchkpts   = list(Nchkpts) + list(Nextra)
for i in range(len(Nextra)):
    nstrs.append(str(Nextra[i]))
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
##################################
#Mdot_boost= 2.3e4/10  ############## Mach 21
Mdot_boost= 8.15e5/10  ############## Mach 11
#Mdot_boost= 6.7e6/10  ############## Mach 7
##################################

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

erg_to_eV = 624150686952.7291

m0        = M*1.                 #code mass   unit in cgs
l0        = a*1.                 #code length unit in cgs
t0        = 1./xp.sqrt(G*M/a**3) #code time   unit in cgs

sigma     = 5.6705119e-5 #Stefan-Boltzmann in cgs units
kb        = 1.38065812e-16 #Boltzmann  in cgs units
kappa     = 0.4

mp_code   = mp    / m0
kb_code   = kb    / (m0*l0**2/t0**2)
sigma_code= sigma / (m0/t0**3)
kappa_code= kappa / (l0**2/m0)
A         = (8./3) * sigma_code * (mp_code/kb_code)**4 * (gamma-1.)**4 / kappa_code
A_half    = A/2
rp_code   = (1-eccentricity)
rp_cgs    = rp_code*l0

lum_resolved_bol   = []
lum_resolved_x     = []
lum_resolved_UV    = []
lum_resolved_vis   = []
lum_resolved_inf   = []
lum_resolved_mw    = []
time               = []
x1,y1,x2,y2        = [],[],[],[]
M1dot, M2dot       = [],[]

lum_subsink_bol_old= []
lum_subsink_M1_bol    = []
lum_subsink_M1_x      = []
lum_subsink_M1_UV     = []
lum_subsink_M1_vis    = []
lum_subsink_M1_inf    = []
lum_subsink_M1_mw     = []
lum_subsink_M2_bol    = []
lum_subsink_M2_x      = []
lum_subsink_M2_UV     = []
lum_subsink_M2_vis    = []
lum_subsink_M2_inf    = []
lum_subsink_M2_mw     = []

lum_BH1_bol, lum_BH1_x, lum_BH1_UV, lum_BH1_vis, lum_BH1_inf, lum_BH1_mw = [],[],[],[],[],[]
lum_BH2_bol, lum_BH2_x, lum_BH2_UV, lum_BH2_vis, lum_BH2_inf, lum_BH2_mw = [],[],[],[],[],[]

r_ISCO_M1_cgs      = 6*M1 *G/c**2
r_ISCO_M2_cgs      = 6*M2 *G/c**2
r_sink_cgs         = rs * l0
rr_subsink_M1_cgs  = np.logspace(0,xp.log10(r_sink_cgs/r_ISCO_M1_cgs),10*N) * r_ISCO_M1_cgs #spans ISCO to sink radius
rr_subsink_M2_cgs  = np.logspace(0,xp.log10(r_sink_cgs/r_ISCO_M2_cgs),10*N) * r_ISCO_M2_cgs

Mom_md_p, Mom_md_m = [],[]

def sink_kernel(x,y):
	rr2    = (xx-x)**2 + (yy-y)**2 #2D array
	s2     = rs*rs
	result = xp.exp(-(rr2/s2)**2)
	result[xp.where(rr2 > 16.0 * s2)] = 0.0
	return result

def mass_accretion_rate(x,y,rho_code):
	sink_field = sink_rate * sink_kernel(x,y) * rho_code * m0/l0**2/t0
	return np.trapz(np.trapz(AsNumpy(sink_field), dx=dx*l0), dx=dx*l0)

def cooling_flux_axisymmetric_subsink(boundary_conditions_for_subsink, whichbody):
	tgrow_cgs = boundary_conditions_for_subsink[0]
	if whichbody==1:
		Mbody = M1*1
		rrrr  = rr_subsink_M1_cgs*1
	elif whichbody==2:
		Mbody = M2*1
		rrrr  = rr_subsink_M2_cgs*1
	return (4./3)*sigma*Tempeq_Me2021(Mbody,rrrr,tgrow_cgs,alpha,8./3,'gas')**4/(kappa*Surface_Density_Me2021(Mbody,rrrr,tgrow_cgs,alpha,8./3,'gas')) #one side of disk

numwavs          = int(1e2)
wavelengths_x    = xp.logspace( xp.log10(cm_x[0]/1e3), xp.log10(cm_x[0]    ), numwavs )
wavelengths_UV   = xp.logspace( xp.log10(cm_UV[  0] ), xp.log10(cm_UV[  -1]), numwavs )
wavelengths_vis  = xp.logspace( xp.log10(cm_vis[ 0] ), xp.log10(cm_vis [-1]), numwavs )
wavelengths_inf  = xp.logspace( xp.log10(cm_inf[ 0] ), xp.log10(cm_inf [-1]), numwavs )
wavelengths_mw   = xp.logspace( xp.log10(cm_mw[  0] ), xp.log10(cm_mw[  -1]), numwavs )
wavelengths_bol  = xp.logspace( xp.log10(cm_x[0]/1e3), xp.log10(cm_mw[  -1]), numwavs )

def energy_ratio(wavelength,temperature):
	nu = c/wavelength
	return h*nu/kb/temperature

planck_anti_approx_code = """
#define SPEED_OF_LIGHT 2.99792458e10
#define H_PLANCK 6.6261e-27
#define K_BOLTZMANN 1.38065812e-16
#define M_SUM 100

PRIVATE double energy_ratio(
    double wavelength,
    double temperature)
{
    double nu = SPEED_OF_LIGHT / wavelength;
    return H_PLANCK * nu / K_BOLTZMANN / temperature;
}

PRIVATE double planck_anti_approx(
    double x,
    int m)
{
    double resul = 0.0;
    for (int n = 1; n < m + 1; ++n)
    {
        resul = resul + (pow(x, 3.0) / n + 3.0 * pow(x, 2.0) / pow(n, 2.0) + 6.0 * x / pow(n, 3.0) + 6.0 / pow(n, 4.0)) * exp(-n * x);
    }
    return resul;
}

PUBLIC void planck_anti_array(
    int ni,
    int nj,
    double *temperature, // :: $.shape == (ni, nj)
    double wavelength,
    double *anti) // :: $.shape == (ni, nj)
{
    FOR_EACH_2D(ni, nj)
    {
        double X = energy_ratio(wavelength, temperature[i * nj + j]);
        anti[i * nj + j] = planck_anti_approx(X, M_SUM);
//        printf("Value of anti = %f", anti[i * nj + j]);
    }
}

PUBLIC void planck_anti_array_1D(
    int ni,
    double *temperature, // :: $.shape == (ni,)
    double wavelength,
    double *anti) // :: $.shape == (ni,)
{
    FOR_EACH_1D(ni)
    {
        double X = energy_ratio(wavelength, temperature[i]);
        anti[i] = planck_anti_approx(X, M_SUM);
//        printf("Value of anti = %f", anti[i]);
    }
}
"""

if xp.__name__ == 'cupy':
    library = Library(planck_anti_approx_code, mode="gpu")
elif xp.__name__ == 'numpy':
    library = Library(planck_anti_approx_code, mode="cpu")
else:
    print("Error: xp must be cupy or numpy")

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
    del pres_code, eps_code2, cs_code, sigspeeds, maxspeed

    ### Cooling in the limit dt -> 0
    #rho       = rho_code * m0/l0**2
    #pres      = pres_code * m0/t0**2
    #kTmid     = pres/rho * mp #midplane kT in units of erg
    #Tmid      = kTmid/kb #midplane T in units of Kelvin
    #cooling_one_side     = (4./3) * sigma * Tmid**4 / (rho * kappa) #limit dt->0
    #del rho, pres, Tmid, kTmid

    eps_code_prime        = eps_code * (1. + 3.*A/rho_code**2 * eps_code**3 *dt)**(-1./3)
    ek                    = 0.5 * (vx_code**2 + vy_code**2)
    epsprime_machceiling  = 2.0 * ek / gamma / (gamma - 1.0) / mach_ceiling**2
    del ek

    #Apply Mach ceiling
    bup = xp.where(epsprime_machceiling > eps_code_prime)
    eps_code_prime[bup] = epsprime_machceiling[bup]

    cooling_one_side = -rho_code * (eps_code_prime - eps_code) / dt / 2
    cooling_one_side[xp.where(cooling_one_side<0)] = 0.0
    del eps_code, eps_code_prime, epsprime_machceiling, bup

    #Compute effective temperature
    Teff      = (cooling_one_side / sigma_code / Mdot_boost)**0.25
    Teff[xp.where(Teff==0)] = 1e-16
    del cooling_one_side

    #Update mass accretion rates
    M1dot.append( mass_accretion_rate(x1[-1],y1[-1],rho_code) )
    M2dot.append( mass_accretion_rate(x2[-1],y2[-1],rho_code) )

    #Optionally, compute various mass fluxes and save corotating frame snapshots
    if save_corotation_data==1:
        mom_md_p, mom_md_m = corotate.mdot_minidisk_separator(int(1.0/dx), 1.0, AsNumpy(rho_code), AsNumpy(vx_code), AsNumpy(vy_code), x1[-1], y1[-1], x2[-1], y2[-1], AsNumpy(xx), AsNumpy(yy))
        Mom_md_p.append(mom_md_p)
        Mom_md_m.append(mom_md_m)
        xp.save('Mom_md_p' ,xp.array(Mom_md_p))
        xp.save('Mom_md_m' ,xp.array(Mom_md_m))
        plt.figure(figsize=[10, 8])
        plt.subplot(211)
        plt.plot(np.array(time)/2/np.pi, np.array(Mom_md_p))
        plt.plot(np.array(time)/2/np.pi, np.array(Mom_md_m))
        plt.xlim(3100,3104)

        rho_rotated = corotate.ROTATE(AsNumpy(rho_code.T), x1[-1], y1[-1], 1)[Nh-Mx:Nh+Mx,Nh-My:Nh+My]
        xp.save(fn+'rho_rotated_'+str(n), rho_rotated)
        xmd, ymd = corotate.minidisk_separator(100, 1.0)
        xcirc1, ycirc1 = corotate.circle(100, -0.5, 0.0, 0.5)
        xcirc2, ycirc2 = corotate.circle(100,  0.5, 0.0, 0.5)
        xcirc3, ycirc3 = corotate.circle(100,  0.0, 0.0, 1.0)
        plt.subplot(212)
        plt.imshow(rho_rotated**0.25, origin='lower', cmap='plasma', extent=(-DR*my,DR*my,-DR*mx,DR*mx))
        plt.plot(xmd, ymd, color='white')
        #plt.plot(xcirc1, ycirc1, color='b')
        #plt.plot(xcirc2, ycirc2, color='r')
        #plt.plot(xcirc3, ycirc3, color='g')
        plt.show()
        plt.savefig("rho_"+str(n)+".png")
        plt.close()
        del rho_rotated

    del rho_code, vx_code, vy_code

    #Compute banded luminosity maps
    lum_resolved_cell_vis  = xp.zeros((N,N))
    lum_resolved_cell_inf  = xp.zeros((N,N))

    anti_vis0 = xp.zeros((N,N))
    anti_inf1 = xp.zeros((N,N))
    anti_vis1 = xp.zeros((N,N))
    library.planck_anti_array[Teff.shape](Teff, cm_vis[0], anti_vis0)
    library.planck_anti_array[Teff.shape](Teff, cm_inf[1], anti_inf1)
    library.planck_anti_array[Teff.shape](Teff, cm_vis[1], anti_vis1)

    prefact  = (2/c**2)*(kb*Teff)**4/h**3
    lum_resolved_cell_vis = xp.pi * prefact * (anti_vis1 - anti_vis0)
    lum_resolved_cell_inf = xp.pi * prefact * (anti_inf1 - anti_vis1)
    lum_resolved_cell_bol = xp.ones((N,N)) * xp.pi * prefact * xp.pi**4/15
    del anti_vis0, anti_inf1, anti_vis1, prefact

    #Integrate banded luminosity maps over space, excluding buffer region
    buffmask = xp.ones((N,N))
    buffmask[np.where(rr>DR-buffer_scale)] = 0.0
    lum_resolved_bol .append( xp.sum(lum_resolved_cell_bol*buffmask )*dx*dx*l0*l0 )
    lum_resolved_vis .append( xp.sum(lum_resolved_cell_vis*buffmask )*dx*dx*l0*l0 )
    lum_resolved_inf .append( xp.sum(lum_resolved_cell_inf*buffmask )*dx*dx*l0*l0 )
    del buffmask

    #Collect emission from vicinity of BH1 and BH2 separately
    rphsq = (rp_code/2)**2 #half pericenter distance squared

    mask_BH1 = (x1[-1]-xx)**2 + (y1[-1]-yy)**2 < rphsq
    lum_BH1_bol .append( xp.sum(lum_resolved_cell_bol *mask_BH1)*dx*dx*l0*l0 )
    lum_BH1_vis .append( xp.sum(lum_resolved_cell_vis *mask_BH1)*dx*dx*l0*l0 )
    lum_BH1_inf .append( xp.sum(lum_resolved_cell_inf *mask_BH1)*dx*dx*l0*l0 )
    del mask_BH1

    mask_BH2 = (x2[-1]-xx)**2 + (y2[-1]-yy)**2 < rphsq
    lum_BH2_bol .append( xp.sum(lum_resolved_cell_bol *mask_BH2)*dx*dx*l0*l0 )
    lum_BH2_vis .append( xp.sum(lum_resolved_cell_vis *mask_BH2)*dx*dx*l0*l0 )
    lum_BH2_inf .append( xp.sum(lum_resolved_cell_inf *mask_BH2)*dx*dx*l0*l0 )
    del mask_BH2

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

    #Compute subsink emission
    tgrow_M1_cgs = M1/M1dot[-1]
    tgrow_M2_cgs = M2/M2dot[-1]
    subsink_flux_M1 = cooling_flux_axisymmetric_subsink([tgrow_M1_cgs], 1)
    subsink_flux_M2 = cooling_flux_axisymmetric_subsink([tgrow_M2_cgs], 2)
    subsink_Teff_M1 = (subsink_flux_M1/sigma/Mdot_boost)**0.25
    subsink_Teff_M2 = (subsink_flux_M2/sigma/Mdot_boost)**0.25

    subsink_flux_M1_densitized = subsink_flux_M1 * 2*np.pi*rr_subsink_M1_cgs
    subsink_flux_M2_densitized = subsink_flux_M2 * 2*np.pi*rr_subsink_M2_cgs

    lum_subsink_bol_old .append( np.trapz(AsNumpy(subsink_flux_M1_densitized), x=AsNumpy(rr_subsink_M1_cgs)) + np.trapz(AsNumpy(subsink_flux_M2_densitized), x=AsNumpy(rr_subsink_M2_cgs)) )

    lum_subsink_M1_annulus_vis  = rr_subsink_M1_cgs*0
    lum_subsink_M1_annulus_inf  = rr_subsink_M1_cgs*0
    lum_subsink_M1_annulus_bol  = rr_subsink_M1_cgs*0

    lum_subsink_M2_annulus_vis  = rr_subsink_M2_cgs*0
    lum_subsink_M2_annulus_inf  = rr_subsink_M2_cgs*0
    lum_subsink_M2_annulus_bol  = rr_subsink_M2_cgs*0

    Nsub = len(rr_subsink_M1_cgs)
    anti_vis0 = xp.zeros(Nsub)
    anti_inf1 = xp.zeros(Nsub)
    anti_vis1 = xp.zeros(Nsub)
    library.planck_anti_array_1D[subsink_Teff_M1.shape](subsink_Teff_M1, cm_vis[0], anti_vis0)
    library.planck_anti_array_1D[subsink_Teff_M1.shape](subsink_Teff_M1, cm_inf[1], anti_inf1)
    library.planck_anti_array_1D[subsink_Teff_M1.shape](subsink_Teff_M1, cm_vis[1], anti_vis1)

    prefact  = (2/c**2)*(kb*subsink_Teff_M1)**4/h**3
    lum_subsink_M1_annulus_vis = xp.pi * prefact * (anti_vis1 - anti_vis0)
    lum_subsink_M1_annulus_inf = xp.pi * prefact * (anti_inf1 - anti_vis1)
    lum_subsink_M1_annulus_bol = xp.ones(Nsub) * xp.pi * prefact * xp.pi**4/15

    Nsub = len(rr_subsink_M2_cgs)
    anti_vis0 = xp.zeros(Nsub)
    anti_inf1 = xp.zeros(Nsub)
    anti_vis1 = xp.zeros(Nsub)
    library.planck_anti_array_1D[subsink_Teff_M2.shape](subsink_Teff_M2, cm_vis[0], anti_vis0)
    library.planck_anti_array_1D[subsink_Teff_M2.shape](subsink_Teff_M2, cm_inf[1], anti_inf1)
    library.planck_anti_array_1D[subsink_Teff_M2.shape](subsink_Teff_M2, cm_vis[1], anti_vis1)

    prefact  = (2/c**2)*(kb*subsink_Teff_M2)**4/h**3
    lum_subsink_M2_annulus_vis = xp.pi * prefact * (anti_vis1 - anti_vis0)
    lum_subsink_M2_annulus_inf = xp.pi * prefact * (anti_inf1 - anti_vis1)
    lum_subsink_M2_annulus_bol = xp.ones(Nsub) * xp.pi * prefact * xp.pi**4/15

    lum_subsink_M1_bol .append( np.trapz(AsNumpy(lum_subsink_M1_annulus_bol *2*xp.pi*rr_subsink_M1_cgs), x=AsNumpy(rr_subsink_M1_cgs)) )
    lum_subsink_M1_vis .append( np.trapz(AsNumpy(lum_subsink_M1_annulus_vis *2*xp.pi*rr_subsink_M1_cgs), x=AsNumpy(rr_subsink_M1_cgs)) )
    lum_subsink_M1_inf .append( np.trapz(AsNumpy(lum_subsink_M1_annulus_inf *2*xp.pi*rr_subsink_M1_cgs), x=AsNumpy(rr_subsink_M1_cgs)) )

    lum_subsink_M2_bol .append( np.trapz(AsNumpy(lum_subsink_M2_annulus_bol *2*xp.pi*rr_subsink_M2_cgs), x=AsNumpy(rr_subsink_M2_cgs)) )
    lum_subsink_M2_vis .append( np.trapz(AsNumpy(lum_subsink_M2_annulus_vis *2*xp.pi*rr_subsink_M2_cgs), x=AsNumpy(rr_subsink_M2_cgs)) )
    lum_subsink_M2_inf .append( np.trapz(AsNumpy(lum_subsink_M2_annulus_inf *2*xp.pi*rr_subsink_M2_cgs), x=AsNumpy(rr_subsink_M2_cgs)) )

    xp.save('lum_subsink_M1_vis' ,xp.array(lum_subsink_M1_vis ))
    xp.save('lum_subsink_M1_inf' ,xp.array(lum_subsink_M1_inf ))

    xp.save('lum_subsink_M2_vis' ,xp.array(lum_subsink_M2_vis ))
    xp.save('lum_subsink_M2_inf' ,xp.array(lum_subsink_M2_inf ))

    lum_total_bol = xp.array(lum_resolved_bol ) + xp.array(lum_subsink_M1_bol ) + xp.array(lum_subsink_M2_bol )
    lum_total_vis = xp.array(lum_resolved_vis ) + xp.array(lum_subsink_M1_vis ) + xp.array(lum_subsink_M2_vis )
    lum_total_inf = xp.array(lum_resolved_inf ) + xp.array(lum_subsink_M1_inf ) + xp.array(lum_subsink_M2_inf )

    xp.save('lum_total_vis' ,xp.array(lum_resolved_vis ) + xp.array(lum_subsink_M1_vis ) + xp.array(lum_subsink_M2_vis ))
    xp.save('lum_total_inf' ,xp.array(lum_resolved_inf ) + xp.array(lum_subsink_M1_inf ) + xp.array(lum_subsink_M2_inf ))

    frac_bol = xp.array(lum_resolved_bol) / xp.array(lum_total_bol)
    frac_vis = xp.array(lum_resolved_vis) / xp.array(lum_total_vis)
    frac_inf = xp.array(lum_resolved_inf) / xp.array(lum_total_inf)

    print(frac_vis)
    print(frac_inf)
    xp.save('frac_bol', frac_bol)
    xp.save('frac_vis', frac_vis)
    xp.save('frac_inf', frac_inf)
