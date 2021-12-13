import pdb
import numpy as np
import msgpack
import re
import cupy as cp
from sailfish.kernel.library import Library

exec(open('constants3.py').read())

def reconstitute(filename, fieldnum, numfields):
    chkpt = msgpack.load(open(filename, 'rb'))
    mesh = chkpt['mesh']
    cfl  = chkpt['command_line']['cfl_number']
    tnew = chkpt['time']
    x1new= chkpt['masses'][0]['x']
    y1new= chkpt['masses'][0]['y']
    x2new= chkpt['masses'][1]['x']
    y2new= chkpt['masses'][1]['y']
    primitive = cp.zeros([mesh['ni'], mesh['nj'], numfields])
    for patch in chkpt['primitive_patches']:
        i0 = patch['rect'][0]['start']
        j0 = patch['rect'][1]['start']
        i1 = patch['rect'][0]['end']
        j1 = patch['rect'][1]['end']
        local_prim = cp.array(np.frombuffer(patch['data'])).reshape([i1 - i0, j1 - j0, numfields])
        primitive[i0:i1, j0:j1] = local_prim
    rho, vx, vy, pres = primitive[:,:,0], primitive[:,:,1], primitive[:,:,2], primitive[:,:,3]
    if fieldnum==-1:
        par = re.search('gamma_law_index=(.+?):', chkpt['parameters'])
        if par==None:
            gamma = 1.666666666666666
        else:
            gamma = cp.float(par.group(1))
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
        cs = cp.sqrt(gamma*pres/rho)
        v  = cp.sqrt(vx**2 + vy**2)
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
d         = msgpack.load(open(fn+'chkpt.'+nstr+'.sf','rb'))
d['parameters'] = ':'+d['parameters']+':' #this allows parameter reads at beginning and end of the parameter string
DR        = cp.float(re.search('domain_radius=(.+?):', d['parameters']).group(1))
N         = d['mesh']['ni']
dx        = d['mesh']['dx']
rs        = cp.float(re.search('sink_radius=(.+?):', d['parameters']).group(1))
cfl       = d['command_line']['cfl_number']
alpha     = cp.float(re.search('alpha=(.+?):', d['parameters']).group(1))
gamma     = cp.float(re.search('gamma_law_index=(.+?):', d['parameters']).group(1))
##################################
#Mdot_boost= 2.3e4/10  ############## Mach 21
Mdot_boost= 8.15e5/10  ############## Mach 11
#Mdot_boost= 6.7e6/10  ############## Mach 7
##################################

mach_ceiling = cp.float(re.search('mach_ceiling=(.+?):', d['parameters']).group(1))
buffer_scale = 0.1
sink_rate    = cp.float(re.search('sink_rate=(.+?):', d['parameters']).group(1))
eccentricity = cp.float(re.search(':e=(.+?):', d['parameters']).group(1))
pressure_floor=cp.float(re.search('pressure_floor=(.+?):', d['command_line']['setup']).group(1))

x         = cp.arange((N))*dx - 2*DR/2. + dx/2.
xx,yy     = cp.zeros((N,N)),cp.zeros((N,N))
for i in range(N):
	xx[:,i] = x*1
	yy[i,:] = x*1
rr        = cp.sqrt(xx**2 + yy**2)
r         = cp.arange(int(N/2))*dx

a         = pc*0.0009691860239254802
M         = Msol*8*10**6
M1        = M/2
M2        = M/2

erg_to_eV = 624150686952.7291

m0        = M*1.                 #code mass   unit in cgs
l0        = a*1.                 #code length unit in cgs
t0        = 1./cp.sqrt(G*M/a**3) #code time   unit in cgs

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
rr_subsink_M1_cgs  = np.logspace(0,cp.log10(r_sink_cgs/r_ISCO_M1_cgs),10*N) * r_ISCO_M1_cgs #spans ISCO to sink radius
rr_subsink_M2_cgs  = np.logspace(0,cp.log10(r_sink_cgs/r_ISCO_M2_cgs),10*N) * r_ISCO_M2_cgs

def sink_kernel(x,y):
	rr2    = (xx-x)**2 + (yy-y)**2 #2D array
	s2     = rs*rs
	result = cp.exp(-(rr2/s2)**2)
	result[cp.where(rr2 > 16.0 * s2)] = 0.0
	return result

def mass_accretion_rate(x,y,rho_code):
	sink_field = sink_rate * sink_kernel(x,y) * rho_code * m0/l0**2/t0
	return np.trapz(np.trapz(cp.asnumpy(sink_field), dx=dx*l0), dx=dx*l0)

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
wavelengths_x    = cp.logspace( cp.log10(cm_x[0]/1e3), cp.log10(cm_x[0]    ), numwavs )
wavelengths_UV   = cp.logspace( cp.log10(cm_UV[  0] ), cp.log10(cm_UV[  -1]), numwavs )
wavelengths_vis  = cp.logspace( cp.log10(cm_vis[ 0] ), cp.log10(cm_vis [-1]), numwavs )
wavelengths_inf  = cp.logspace( cp.log10(cm_inf[ 0] ), cp.log10(cm_inf [-1]), numwavs )
wavelengths_mw   = cp.logspace( cp.log10(cm_mw[  0] ), cp.log10(cm_mw[  -1]), numwavs )
wavelengths_bol  = cp.logspace( cp.log10(cm_x[0]/1e3), cp.log10(cm_mw[  -1]), numwavs )

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

library = Library(planck_anti_approx_code, mode="gpu")

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
	pres_code[cp.where(pres_code<0)] = pressure_floor
	eps_code  = pres_code/rho_code/(gamma-1.)
	if cp.amax(abs(eps_code-eps_code2))!=0:
		print("Warning: eps does not agree with pres/rho/(gamma-1).")
	cs_code   = cp.sqrt(gamma*pres_code/rho_code)
	sigspeeds = cp.array([abs(vx_code) + cs_code, abs(vy_code) + cs_code])
	maxspeed  = cp.amax(sigspeeds)
	dt        = cfl*dx/maxspeed
	rho       = rho_code * m0/l0**2
	pres      = pres_code * m0/t0**2
	kTmid     = pres/rho * mp #midplane kT in units of erg
	Tmid      = kTmid/kb #midplane T in units of Kelvin

	#cooling_one_side     = (4./3) * sigma * Tmid**4 / (rho * kappa) #limit dt->0
	eps_code_prime        = eps_code * (1. + 3.*A/rho_code**2 * eps_code**3 *dt)**(-1./3)
	ek                    = 0.5 * (vx_code**2 + vy_code**2)
	epsprime_machceiling  = 2.0 * ek / gamma / (gamma - 1.0) / mach_ceiling**2

	#Apply Mach ceiling
	bup = cp.where(epsprime_machceiling > eps_code_prime)
	eps_code_prime[bup] = epsprime_machceiling[bup]

	cooling_one_side = -rho_code * (eps_code_prime - eps_code) / dt / 2
	cooling_one_side[cp.where(cooling_one_side<0)] = 0.0

	Teff      = (cooling_one_side / sigma_code / Mdot_boost)**0.25
	Teff[cp.where(Teff==0)] = 1e-16
	kTeff     = kb*Teff         #kTeff in units of erg
	kTeffeV   = kTeff*erg_to_eV #kTeff in units of eV

	M1dot.append( mass_accretion_rate(x1[-1],y1[-1],rho_code) )
	M2dot.append( mass_accretion_rate(x2[-1],y2[-1],rho_code) )

	lum_resolved_cell_vis  = cp.zeros((N,N))
	lum_resolved_cell_inf  = cp.zeros((N,N))

	anti_vis0 = cp.zeros((N,N))
	anti_inf1 = cp.zeros((N,N))
	anti_vis1 = cp.zeros((N,N))
	library.planck_anti_array[Teff.shape](Teff, cm_vis[0], anti_vis0)
	library.planck_anti_array[Teff.shape](Teff, cm_inf[1], anti_inf1)
	library.planck_anti_array[Teff.shape](Teff, cm_vis[1], anti_vis1)

	prefact  = (2/c**2)*(kb*Teff)**4/h**3
	lum_resolved_cell_vis = cp.pi * prefact * (anti_vis1 - anti_vis0)
	lum_resolved_cell_inf = cp.pi * prefact * (anti_inf1 - anti_vis1)
	lum_resolved_cell_bol = cp.ones((N,N)) * cp.pi * prefact * cp.pi**4/15

	buffmask = cp.ones((N,N))
	buffmask[np.where(rr>DR-buffer_scale)] = 0.0
	lum_resolved_bol .append( cp.sum(lum_resolved_cell_bol*buffmask )*dx*dx*l0*l0 )
	lum_resolved_vis .append( cp.sum(lum_resolved_cell_vis*buffmask )*dx*dx*l0*l0 )
	lum_resolved_inf .append( cp.sum(lum_resolved_cell_inf*buffmask )*dx*dx*l0*l0 )

	#Collect emission from vicinity of BH1 and BH2 separately
	mask_BH1 = Teff*0
	mask_BH2 = Teff*0
	rphsq = (rp_code/2)**2 #half pericenter distance squared
	distance_to_BH1_squared = (x1[-1]-xx)**2 + (y1[-1]-yy)**2
	distance_to_BH2_squared = (x2[-1]-xx)**2 + (y2[-1]-yy)**2
	mask_BH1 = distance_to_BH1_squared < rphsq
	mask_BH2 = distance_to_BH2_squared < rphsq

	lum_BH1_bol .append( cp.sum(lum_resolved_cell_bol *mask_BH1)*dx*dx*l0*l0 )
	lum_BH1_vis .append( cp.sum(lum_resolved_cell_vis *mask_BH1)*dx*dx*l0*l0 )
	lum_BH1_inf .append( cp.sum(lum_resolved_cell_inf *mask_BH1)*dx*dx*l0*l0 )

	lum_BH2_bol .append( cp.sum(lum_resolved_cell_bol *mask_BH2)*dx*dx*l0*l0 )
	lum_BH2_vis .append( cp.sum(lum_resolved_cell_vis *mask_BH2)*dx*dx*l0*l0 )
	lum_BH2_inf .append( cp.sum(lum_resolved_cell_inf *mask_BH2)*dx*dx*l0*l0 )

	cp.save('lum_resolved_vis' ,cp.array(lum_resolved_vis ))
	cp.save('lum_resolved_inf' ,cp.array(lum_resolved_inf ))
	cp.save('time'             ,cp.array(time             ))
	cp.save('x1'               ,x1                         )
	cp.save('y1'               ,y1                         )
	cp.save('x2'               ,x2                         )
	cp.save('y2'               ,y2                         )
	cp.save('M1dot'            ,cp.array(M1dot            ))
	cp.save('M2dot'            ,cp.array(M2dot            ))

	cp.save('lum_BH1_vis' ,cp.array(lum_BH1_vis ))
	cp.save('lum_BH1_inf' ,cp.array(lum_BH1_inf ))

	cp.save('lum_BH2_vis' ,cp.array(lum_BH2_vis ))
	cp.save('lum_BH2_inf' ,cp.array(lum_BH2_inf ))

	cp.save(fn+                  'Teff_'+str(n),Teff                  )
	cp.save(fn+'lum_resolved_cell_vis_' +str(n),lum_resolved_cell_vis )
	cp.save(fn+'lum_resolved_cell_inf_' +str(n),lum_resolved_cell_inf )

	#Compute subsink emission
	tgrow_M1_cgs = M1/M1dot[-1]
	tgrow_M2_cgs = M2/M2dot[-1]
	subsink_flux_M1 = cooling_flux_axisymmetric_subsink([tgrow_M1_cgs], 1)
	subsink_flux_M2 = cooling_flux_axisymmetric_subsink([tgrow_M2_cgs], 2)
	subsink_Teff_M1 = (subsink_flux_M1/sigma/Mdot_boost)**0.25
	subsink_Teff_M2 = (subsink_flux_M2/sigma/Mdot_boost)**0.25

	subsink_flux_M1_densitized = subsink_flux_M1 * 2*np.pi*rr_subsink_M1_cgs
	subsink_flux_M2_densitized = subsink_flux_M2 * 2*np.pi*rr_subsink_M2_cgs

	lum_subsink_bol_old .append( np.trapz(cp.asnumpy(subsink_flux_M1_densitized), x=cp.asnumpy(rr_subsink_M1_cgs)) + np.trapz(cp.asnumpy(subsink_flux_M2_densitized), x=cp.asnumpy(rr_subsink_M2_cgs)) )

	lum_subsink_M1_annulus_vis  = rr_subsink_M1_cgs*0
	lum_subsink_M1_annulus_inf  = rr_subsink_M1_cgs*0
	lum_subsink_M1_annulus_bol  = rr_subsink_M1_cgs*0

	lum_subsink_M2_annulus_vis  = rr_subsink_M2_cgs*0
	lum_subsink_M2_annulus_inf  = rr_subsink_M2_cgs*0
	lum_subsink_M2_annulus_bol  = rr_subsink_M2_cgs*0

	Nsub = len(rr_subsink_M1_cgs)
	anti_vis0 = cp.zeros(Nsub)
	anti_inf1 = cp.zeros(Nsub)
	anti_vis1 = cp.zeros(Nsub)
	library.planck_anti_array_1D[subsink_Teff_M1.shape](subsink_Teff_M1, cm_vis[0], anti_vis0)
	library.planck_anti_array_1D[subsink_Teff_M1.shape](subsink_Teff_M1, cm_inf[1], anti_inf1)
	library.planck_anti_array_1D[subsink_Teff_M1.shape](subsink_Teff_M1, cm_vis[1], anti_vis1)

	prefact  = (2/c**2)*(kb*subsink_Teff_M1)**4/h**3
	lum_subsink_M1_annulus_vis = cp.pi * prefact * (anti_vis1 - anti_vis0)
	lum_subsink_M1_annulus_inf = cp.pi * prefact * (anti_inf1 - anti_vis1)
	lum_subsink_M1_annulus_bol = cp.ones(Nsub) * cp.pi * prefact * cp.pi**4/15

	Nsub = len(rr_subsink_M2_cgs)
	anti_vis0 = cp.zeros(Nsub)
	anti_inf1 = cp.zeros(Nsub)
	anti_vis1 = cp.zeros(Nsub)
	library.planck_anti_array_1D[subsink_Teff_M2.shape](subsink_Teff_M2, cm_vis[0], anti_vis0)
	library.planck_anti_array_1D[subsink_Teff_M2.shape](subsink_Teff_M2, cm_inf[1], anti_inf1)
	library.planck_anti_array_1D[subsink_Teff_M2.shape](subsink_Teff_M2, cm_vis[1], anti_vis1)

	prefact  = (2/c**2)*(kb*subsink_Teff_M2)**4/h**3
	lum_subsink_M2_annulus_vis = cp.pi * prefact * (anti_vis1 - anti_vis0)
	lum_subsink_M2_annulus_inf = cp.pi * prefact * (anti_inf1 - anti_vis1)
	lum_subsink_M2_annulus_bol = cp.ones(Nsub) * cp.pi * prefact * cp.pi**4/15

	lum_subsink_M1_bol .append( np.trapz(cp.asnumpy(lum_subsink_M1_annulus_bol *2*cp.pi*rr_subsink_M1_cgs), x=cp.asnumpy(rr_subsink_M1_cgs)) )
	lum_subsink_M1_vis .append( np.trapz(cp.asnumpy(lum_subsink_M1_annulus_vis *2*cp.pi*rr_subsink_M1_cgs), x=cp.asnumpy(rr_subsink_M1_cgs)) )
	lum_subsink_M1_inf .append( np.trapz(cp.asnumpy(lum_subsink_M1_annulus_inf *2*cp.pi*rr_subsink_M1_cgs), x=cp.asnumpy(rr_subsink_M1_cgs)) )

	lum_subsink_M2_bol .append( np.trapz(cp.asnumpy(lum_subsink_M2_annulus_bol *2*cp.pi*rr_subsink_M2_cgs), x=cp.asnumpy(rr_subsink_M2_cgs)) )
	lum_subsink_M2_vis .append( np.trapz(cp.asnumpy(lum_subsink_M2_annulus_vis *2*cp.pi*rr_subsink_M2_cgs), x=cp.asnumpy(rr_subsink_M2_cgs)) )
	lum_subsink_M2_inf .append( np.trapz(cp.asnumpy(lum_subsink_M2_annulus_inf *2*cp.pi*rr_subsink_M2_cgs), x=cp.asnumpy(rr_subsink_M2_cgs)) )

	cp.save('lum_subsink_M1_vis' ,cp.array(lum_subsink_M1_vis ))
	cp.save('lum_subsink_M1_inf' ,cp.array(lum_subsink_M1_inf ))

	cp.save('lum_subsink_M2_vis' ,cp.array(lum_subsink_M2_vis ))
	cp.save('lum_subsink_M2_inf' ,cp.array(lum_subsink_M2_inf ))

	lum_total_bol = cp.array(lum_resolved_bol ) + cp.array(lum_subsink_M1_bol ) + cp.array(lum_subsink_M2_bol )
	lum_total_vis = cp.array(lum_resolved_vis ) + cp.array(lum_subsink_M1_vis ) + cp.array(lum_subsink_M2_vis )
	lum_total_inf = cp.array(lum_resolved_inf ) + cp.array(lum_subsink_M1_inf ) + cp.array(lum_subsink_M2_inf )

	cp.save('lum_total_vis' ,cp.array(lum_resolved_vis ) + cp.array(lum_subsink_M1_vis ) + cp.array(lum_subsink_M2_vis ))
	cp.save('lum_total_inf' ,cp.array(lum_resolved_inf ) + cp.array(lum_subsink_M1_inf ) + cp.array(lum_subsink_M2_inf ))

	frac_bol = cp.array(lum_resolved_bol) / cp.array(lum_total_bol)
	frac_vis = cp.array(lum_resolved_vis) / cp.array(lum_total_vis)
	frac_inf = cp.array(lum_resolved_inf) / cp.array(lum_total_inf)

	print(frac_vis)
	print(frac_inf)
	cp.save('frac_bol', frac_bol)
	cp.save('frac_vis', frac_vis)
	cp.save('frac_inf', frac_inf)
