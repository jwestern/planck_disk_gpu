import numpy as np
from scipy.optimize import brentq
import pdb
#cgs unless otherwise specified
G     = 6.6725985e-8
sigma = 5.6705119e-5
kb    = 1.38065812e-16
gamma = 5./3
pc    = 3085678000000000000
mp    = 1.6726e-24
Msol  = 1.989e33
c     = 2.99792458e10
Msolkm= G*Msol/c**2 #~1.48 km
a     = pc*1e-3
M     = Msol*1e5
kappa = 0.4
h     = 6.6261e-27

Au_to_cm   = 1.496e13
cm_to_km   = 1./1e2/1e3  #convert to meters then to km
km_to_cm   = 1./cm_to_km
pc_to_km   = pc*cm_to_km #convert from pc -> cm -> m -> km
km_to_pc   = 1./pc_to_km
Msol_to_km = G/c**2*cm_to_km
km_to_Msol = 1./Msol_to_km

Msol_to_pc = Msol_to_km*km_to_pc
pc_to_Msol = 1./Msol_to_pc

yrs_to_sec = 365.*24*60*60
sec_to_yrs = 1./yrs_to_sec

tsal       = 4.5e7*yrs_to_sec

def Macheq(M,a,tgrow,nubar):
        return ( 32./27 * sigma/kappa * (mp/kb/gamma)**4 * (3*np.pi*tgrow/M)**2 * nubar * a**3  * (G*M/a)**(7./2) )**(1./8)

def Macheq2(M,a,nubar):
	return ( 32./27 * sigma/kappa * (mp/kb/gamma)**4 * (3*np.pi*tsal/M)**2 * nubar * a**3  * (G*M/a)**(7./2) )**(1./10)

def Macheq3_old(M,a,r,tgrow,nubar):
	return ( mp/kb/gamma * G*M/r * ( 27./32 * kappa/sigma * (M/3/np.pi/tgrow)**2 * G*M/r**3 / (nubar*np.sqrt(G*M*a)) )**(-1./4) )**(1./2)

def Macheq3(M,a,r,tgrow,nubar):
	return ( (mp/kb/gamma * G*M/r)**4 * 32./27 * sigma/kappa * (3*np.pi*tgrow/M)**2 * r**3/G/M * nubar*np.sqrt(G*M*a) )**(1./8)

def Macheq3_alpha(M,a,r,tgrow,alpha):
	#return ( (mp/kb/gamma * G*M/r)**4 * 32./27 * sigma/kappa * (3*np.pi*tgrow/M)**2 * r**3/G/M * alpha/np.sqrt(gamma) * np.sqrt(G*M*r) )**(1./10)
	T = Tempeq_alpha(M,a,r,tgrow,alpha)
	return np.sqrt( G*M/r * mp/gamma/kb/T )

def Macheq3_rad_alpha(M,r,tgrow):
        return 2*np.pi/np.sqrt(3) *c/kappa * tgrow/M * r

def Tempeq(M,a,r,tgrow,nubar):
	return ( 27./32 * kappa/sigma * (M/3/np.pi/tgrow)**2 * G*M/r**3 / (nubar*np.sqrt(G*M*a)) )**0.25

def Tempeq_alpha(M,a,r,tgrow,alpha):
	return ( 27./32 * kappa/sigma * (M/3/np.pi/tgrow)**2 * mp/alpha/np.sqrt(gamma)/kb * (G*M/r**3)**(3./2) )**(1./5)

def Tempeq_rad_alpha(M,r,alpha):
	return ( (1./6) * np.sqrt(3./4) * c**2/sigma/kappa * 1./alpha * np.sqrt(G*M/r**3) )**(1./4)

def Tcool(M,a,r,tgrow,nubar): #Needs checking
	#pdb.set_trace()
	Mach = Macheq3(M,a,r,tgrow,nubar)
	return 3./8 * kappa/sigma * (kb/mp)**4 * Mach**6 * (r/G/M)**3 / (3*np.pi*nubar)**2 / (G*M*a) * (M/tgrow)**2 * gamma**3 / (gamma-1)

def Tvisc(M,a,r,nubar):
	return (2./3) * r**2 / (nubar*np.sqrt(G*M*a))

def Tvisc_orbits(M,a,r,nubar):
	return Tvisc(M,a,r,nubar)/365/24/60/60/Orbital_Period_Yrs(M,a)

def Tvisc_alpha(M,a,r,tgrow,alpha):
	Mach = Macheq3_alpha(M,a,r,tgrow,alpha)
	nu   = alpha/np.sqrt(gamma)/Mach**2 * np.sqrt(G*M*r)
	return (2./3) * r**2 / nu

def Tvisc_alpha_orbits(M,a,r,tgrow,alpha):
	return Tvisc_alpha(M,a,r,tgrow,alpha)/365/24/60/60/Orbital_Period_Yrs(M,a)

def Tvisc_alpha_isothermal(M,r,alpha,Mach):
        nu   = alpha/Mach**2 * np.sqrt(G*M*r)
        return (2./3) * r**2 / nu

def Tvisc_alpha_isothermal_orbits(M,a,r,alpha,Mach):
        return Tvisc_alpha_isothermal(M,r,alpha,Mach)/365/24/60/60/Orbital_Period_Yrs(M,a)

def Tsal(M,a,Mach): #Needs checking
	return 9./4 * Mach**8 * G*M/sigma/a**3 * M/2/np.pi * (mp/kb/gamma * G*M/a)**(-4.)

def Decoupling_Radius(M,tgrow):
	return (1./6 * 96./5 * tgrow / c**5. *(G*M)**3)**(1./4)

def Decoupling_Radius_ArbEig(M,tgrow,l): #arbitrary accretion "eigenvalue" l, da/dm = -l * a/m
        return (1./6 * 96./5 * tgrow / c**5. *(G*M)**3 / l)**(1./4)

#Must be wrong because q=1 doesn't recover the previous formula
def Decoupling_Radius_ArbEigq(M,tgrow,l,q): #also arbitrary mass ratio q
        return (1./6 * 96./5 * tgrow / c**5. *(G*M)**3 / l * 4 * q / (1+q)**2)**(1./4)

def Decoupling_Radius_ArbEigqe(M,tgrow,l,q,e): #also arbitrary eccentricity e. Chirp mass get mapped to e_correction * chirp_mass
        e_correction = (1-e**2)**(-7./2) * (1 + 73./24*e**2 + 37/96*e**4)
        return ( tgrow * 2./3 * 96./5 * (G*M)**3/c**5 * q/(1+q)**2 / l / e_correction )**(1./4)

def Orbital_Period_Yrs(M,a):
	return 2*np.pi/np.sqrt(G*M/a**3)/60/60/24/365

def Surface_Density(M,a,nubar,tgrow):
	return M/tgrow /3/np.pi/nubar/np.sqrt(G*M*a)

def Surface_Density_alpha(M,a,r,tgrow,alpha):
	Mach = Macheq3_alpha(M,a,r,tgrow,alpha)
	nu   = alpha/np.sqrt(gamma) / Mach**2 * np.sqrt(G*M*r)
	return M/3/np.pi/tgrow/nu

def Surface_Density_CodeUnits(M,a,nubar,tgrow):
	m0 = M*1
	l0 = a*1
	return Surface_Density(M,a,nubar,tgrow) /(m0/l0**2)

def Surface_Density_alpha_CodeUnits(M,a,r,tgrow,alpha):
	m0 = M*1
	l0 = a*1
	return Surface_Density_alpha(M,a,r,tgrow,alpha) /(m0/l0**2)

def Surface_Pressure(M,a,r,nubar,tgrow):
	return Surface_Density(M,a,nubar,tgrow) * kb/mp * Tempeq(M,a,r,tgrow,nubar)

def Surface_Pressure_alpha(M,a,r,tgrow,alpha):
	return Surface_Density_alpha(M,a,r,tgrow,alpha) * kb/mp * Tempeq_alpha(M,a,r,tgrow,alpha)
	#return 1./Macheq3_alpha(M,a,r,tgrow,alpha)**2 * Surface_Density_alpha(M,a,r,tgrow,alpha) / gamma * G*M/r

def Surface_Pressure_CodeUnits(M,a,r,nubar,tgrow):
	m0 = M*1
	t0 = 1./np.sqrt(G*M/a**3)
	return Surface_Pressure(M,a,r,nubar,tgrow) /(m0/t0**2)

def Surface_Pressure_alpha_CodeUnits(M,a,r,tgrow,alpha):
	m0 = M*1
	t0 = 1./np.sqrt(G*M/a**3)
	return Surface_Pressure_alpha(M,a,r,tgrow,alpha) /(m0/t0**2)

def Disk_Height(M,a,r,nubar,tgrow):
	pres = Surface_Pressure(M,a,r,nubar,tgrow)
	Sigma= Surface_Density(M,a,nubar,tgrow)
	return np.sqrt(pres/Sigma / (G*M/r**3))

def Disk_Height_alpha(M,a,r,tgrow,alpha):
	#pres = Surface_Pressure_alpha(M,a,r,tgrow,alpha)
	#Sigma= Surface_Density_alpha(M,a,r,tgrow,alpha)
	#return np.sqrt(pres/Sigma / (G*M/r**3))
	mach = Macheq3_alpha(M,a,r,tgrow,alpha)
	return np.sqrt(G*M/r)/np.sqrt(gamma)/mach/np.sqrt(G*M/r**3)

def Disk_Height_rad_alpha(M,r,tgrow):
	mach = Macheq3_rad_alpha(M,r,tgrow)
	gamrad = 4./3
	return np.sqrt(G*M/r)/np.sqrt(gamrad)/mach/np.sqrt(G*M/r**3)



def T4_coefficient(M,a):
	m0 = M*1
	l0 = a*1
	t0 = 1./np.sqrt(G*m0/l0**3)

	mp_code    = mp    / m0
	kb_code    = kb    / (m0*l0**2/t0**2)
	sigma_code = sigma / (m0/t0**3)
	return 2.0 * sigma_code * (mp_code/kb_code)**4 * (gamma-1.)**4

def T4_over_kappa_coefficient(M,a):
	m0 = M*1
	l0 = a*1
	t0 = 1./np.sqrt(G*m0/l0**3)

	mp_code    = mp    / m0
	kb_code    = kb    / (m0*l0**2/t0**2)
	sigma_code = sigma / (m0/t0**3)
	kappa_code = kappa / (l0**2/m0)
	return (8./3) * sigma_code * (mp_code/kb_code)**4 * (gamma-1.)**4 / kappa_code



def Toomre_radius(M,a,nubar,tgrow):
	A     = ( 27./32 * kappa/sigma * (M/3/np.pi/tgrow)**2 * G*M / (nubar*np.sqrt(G*M*a)) )**0.25
	Sigma = Surface_Density(M,a,nubar,tgrow)
	return ( np.sqrt(gamma*kb*A*G*M/mp) / np.pi / G / Sigma )**(8./15)

def Toomre_radius_alpha():
	return "TODO"

def Nu_Alpha(M,a,r,tgrow,alpha):
	mach = Macheq3_alpha(M,a,r,tgrow,alpha)
	return alpha/np.sqrt(gamma) / mach**2 * np.sqrt(G*M*r)

def MdotEdd(M): #Assumes electron-scattering opacity kappa (=0.4 above) and efficiency=0.1
	return (4*np.pi*G*c*M/kappa) / (0.1*c**2)

def ZH09_rgasrad(M,alpha,tgrow): #Eq (12) from ZH09. Units of 1e3 Schwarzschild radii
	Mdot = M/tgrow
	return 0.482 * (alpha/0.3)**(2./21) * (Mdot/MdotEdd(M)/0.1)**(16./21) * (M/1e7/Msol)**(2./21)

def ZH09_rToomre(M,alpha,tgrow):
	Mdot = M/tgrow
	return 12.6 * (alpha/0.3)**(8./9) * (Mdot/MdotEdd(M)/0.1)**(-8./27) * (M/1e7/Msol)**(26./27)

def ZH09_rdecouple(M,alpha,tgrow): #Eq. (30b) for middle region
	Mdot = M/tgrow
	return 0.222 * (alpha/0.3)**(-4./13) * (Mdot/MdotEdd(M)/0.1)**(-2./13) * (M/1e7/Msol)**(1./13)

def Rgasrad(M,a,r,tgrow,nubar): #gad/rad transition for constant nu viscosity
	T = Tempeq(M,a,r,tgrow,nubar)
	Prad = (4./3) * (sigma/c) * T**4 * 2 * Disk_Height(M,a,r,nubar,tgrow)
	Pgas = Surface_Pressure(M,a,r,nubar,tgrow)
	igasrad = np.where(Prad<Pgas)[0][0]
	return r[igasrad] / (2*G*M/c**2)

def Rgasrad_alpha(M,a,r,tgrow,alpha): #gad/rad transition for alpha viscosity
	T = Tempeq_alpha(M,a,r,tgrow,alpha)
	Prad = (4./3) * (sigma/c) * T**4 * 2 * Disk_Height_alpha(M,a,r,tgrow,alpha)
	Pgas = Surface_Pressure_alpha(M,a,r,tgrow,alpha)
	igasrad = np.where(Prad<Pgas)[0][0]
	return r[igasrad] / (2*G*M/c**2)

def Tempeq_Goodman2003(M,r,tgrow,alpha,n=4): #n=4 is Goodman2003 value, we use 8./3
	beta = Beta_Goodman2003(M,r,tgrow,alpha,n,gas_or_mixed)
	return (kappa*mp/np.pi**2/alpha/kb/sigma * 9./4/3/3/n)**(1./5) * (M/tgrow)**(2./5) * np.sqrt(G*M/r**3)**(3./5)

def Surface_Density_Goodman2003(M,r,tgrow,alpha,n=4):
	beta = Beta_Goodman2003(M,r,tgrow,alpha,n,gas_or_mixed)
	return n**(1./5)*(9./4)**(-1./5)*(3*np.pi)**(-3./5) * (mp**4*sigma/kb**4)**(1./5) \
	     * alpha**(-4./5) * beta**(4./5) * kappa**(-1./5) * (M/tgrow)**(3./5) * np.sqrt(G*M/r**3)**(2./5)

def Sound_Speed_Goodman2003(M,r,tgrow,alpha,n=4):
	beta = Beta_Goodman2003(M,r,tgrow,alpha,n,gas_or_mixed)
	return np.sqrt( kb*Tempeq_Goodman2003(M,r,tgrow,alpha,n)/mp/beta )

def Disk_Height_Goodman2003(M,r,tgrow,alpha,n=4):
	return Sound_Speed_Goodman2003(M,r,tgrow,alpha,n) / np.sqrt(G*M/r**3)

def Prad_Goodman2003(M,r,tgrow,alpha,n=4):
	return (4./3)*sigma/c * Tempeq_Goodman2003(M,r,tgrow,alpha,n)**4 * 2*Disk_Height_Goodman2003(M,r,tgrow,alpha,n)

def Pgas_Goodman2003(M,r,tgrow,alpha,n=4):
	return Surface_Density_Goodman2003(M,r,tgrow,alpha,n) * kb/mp * Tempeq_Goodman2003(M,r,tgrow,alpha,n)

def Beta_Goodman2003(M,r,tgrow,alpha,n=4):
	#rootfind on Beta_Aux
	roots = r*0
	if np.shape(r)==():
		return brentq(Beta_Aux,1e-16,1-1e-16,args=(M,r,tgrow,alpha,n))
	else:
		for i in range(len(roots)):
			roots[i] = brentq(Beta_Aux,1e-16,1-1e-16,args=(M,r[i],tgrow,alpha,n))
		return roots

def Beta_Aux(beta, M,r,tgrow,alpha,n=4):
	return n**(1./5)*(9./4)**(-1./5)*(3*np.pi)**(-3./5) * ((9./4/3/3/n/np.pi**2)**(1./5))**(-7./2)*3./8 \
	     * alpha**(-1./10)*c*(kb/mp)**(2./5)*sigma**(-1./10) \
	     * kappa**(-9./10) * np.sqrt(G*M/r**3)**(-7./10) * (M/tgrow)**(-4./5) \
	     - beta**(1./2 - 1./10)/(1.-beta)

def Tempeq_Goodman2003_aux(beta,M,r,tgrow,alpha,n=4):
        f = 1.0
        Omega3 = np.sqrt(G*M/r**3)**3
        return (9./4/n*kappa/sigma*mp/kb/alpha*(M/tgrow/3/np.pi)**2*beta/np.sqrt(f)*Omega3)**(1./5)

def Surface_Density_Goodman2003_aux(beta,M,r,tgrow,alpha,n=4):
        f = 1.0
        Omega = np.sqrt(G*M/r**3)
        T = Tempeq_Goodman2003_aux(beta,M,r,tgrow,alpha,n)
        return M/tgrow/3/np.pi/alpha*mp/kb*Omega*beta/np.sqrt(f)/T

def Beta_Goodman2003_aux(beta,M,r,tgrow,alpha,n=4):
        f     = 1.0
        Omega = np.sqrt(G*M/r**3)
        T     = Tempeq_Goodman2003_aux(         beta,M,r,tgrow,alpha,n)
        Sigma = Surface_Density_Goodman2003_aux(beta,M,r,tgrow,alpha,n)
        return np.sqrt(beta) - (1-beta)*3./8*np.sqrt(kb/mp)*c/sigma*Omega*Sigma/T**(7./2)

def Beta_Goodman2003_2(M,r,tgrow,alpha,n=4):
        roots = r*0
        if np.shape(r)==():
                return brentq(Beta_Goodman2003_aux,1e-16,1-1e-16,args=(M,r,tgrow,alpha,n))
        else:
                for i in range(len(roots)):
                        roots[i] = brentq(Beta_Goodman2003_aux,1e-16,1-1e-16,args=(M,r[i],tgrow,alpha,n))
                return roots

def Pratio_Goodman2003(M,r,tgrow,alpha,n=4):
	return Prad_Goodman2003(M,r,tgrow,alpha,n)/Pgas_Goodman2003(M,r,tgrow,alpha,n)

def Mach_Goodman2003(M,r,tgrow,alpha,n=4):
	return np.sqrt(G*M/r) / Sound_Speed_Goodman2003(M,r,tgrow,alpha,n)

def ToomreQ_Goodman2003(M,r,tgrow,alpha,n=4):
	cs    = Sound_Speed_Goodman2003(M,r,tgrow,alpha,n)
	Sigma = Surface_Density_Goodman2003(M,r,tgrow,alpha,n)
	Omega = np.sqrt(G*M/r**3)
	return cs*Omega/np.pi/G/Sigma

def Toomre_aux(r,M,tgrow,alpha,n=4):
	return 1-ToomreQ_Goodman2003(M,r,tgrow,alpha,n)

def Toomre_Goodman2003(M,tgrow,alpha,n=4):
	RS = 2*G*M/c**2
	return brentq(Toomre_aux,RS,1e10*RS,args=(M,tgrow,alpha,n))/RS

def TGW_Haiman2009(M,r):
	RS = 2*G*M/c**2
	return (5./2) * RS/c * (r/RS)**4

def Tvisc_Goodman2003(M,r,tgrow,alpha,n=4):
        Mach = Mach_Goodman2003(M,r,tgrow,alpha,n)
        nu   = alpha/Mach**2 * np.sqrt(G*M*r)
        return (2./3) * r**2 / nu

def TGWvisc_aux(r,M,tgrow,alpha,n=4):
	return Ts_Goodman2003(M,r,tgrow,alpha,n)-TGW_Haiman2009(M,r)

def TGWvisc_Goodman2003(M,r,tgrow,alpha,n=4):
	RS = 2*G*M/c**2
	return brentq(TGWvisc_aux,RS,1e10*RS,args=(M,tgrow,alpha,n))/RS

def Ts_Goodman2003(M,r,tgrow,alpha,n=4):
	mu = M/4 #assumes equal-mass binary
	tv = Tvisc_Goodman2003(M,3*r,tgrow,alpha,n) #use r = 3*r for cavity wall
	qB = 2*M/tgrow * tv/mu #Eq 18 from Haiman+2009
	k  = 3./8 #Eq 19 from Haiman+2009, assumes qB>1
	return qB**(-k) * tv

def Nu_Goodman2003(M,r,tgrow,alpha,n=4):
        cs    = Sound_Speed_Goodman2003(M,r,tgrow,alpha,n)
        h     = Disk_Height_Goodman2003(M,r,tgrow,alpha,n)
        return alpha*cs*h

def fbeta_Me2021(beta,gam):
	return beta + (4-3*beta)**2*(gam-1)/(beta + 12*(gam-1)*(1-beta))

def Tempeq_Me2021_aux(beta,M,r,tgrow,alpha,n=8./3):
	f = fbeta_Me2021(beta,5./3)
	Omega3 = np.sqrt(G*M/r**3)**3
	return (9./4/n*kappa/sigma*mp/kb/alpha*(M/tgrow/3/np.pi)**2*beta/np.sqrt(f)*Omega3)**(1./5)

def Surface_Density_Me2021_aux(beta,M,r,tgrow,alpha,n=8./3):
	f = fbeta_Me2021(beta,5./3)
	Omega = np.sqrt(G*M/r**3)
	T = Tempeq_Me2021_aux(beta,M,r,tgrow,alpha,n)
	return M/tgrow/3/np.pi/alpha*mp/kb*Omega*beta/np.sqrt(f)/T

def Beta_Me2021_aux(beta,M,r,tgrow,alpha,n=8./3):
	f     = fbeta_Me2021(beta,5./3)
	Omega = np.sqrt(G*M/r**3)
	T     = Tempeq_Me2021_aux(         beta,M,r,tgrow,alpha,n)
	Sigma = Surface_Density_Me2021_aux(beta,M,r,tgrow,alpha,n)
	return np.sqrt(beta) - (1-beta)*3./8*np.sqrt(kb/mp)*c/sigma*Omega*Sigma/T**(7./2)

def Beta_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	if gas_or_mixed=='mixed':
		roots = r*0
		if np.shape(r)==():
			return brentq(Beta_Me2021_aux,1e-16,1-1e-16,args=(M,r,tgrow,alpha,n))
		else:
			for i in range(len(roots)):
				roots[i] = brentq(Beta_Me2021_aux,1e-16,1-1e-16,args=(M,r[i],tgrow,alpha,n))
			return roots
	elif gas_or_mixed=='gas':
		return r**0

def Tempeq_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	beta = Beta_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	return Tempeq_Me2021_aux(beta,M,r,tgrow,alpha,n)

def Tempeq_Me2021_rel(M,r,tgrow,alpha,n=8./3):
	Told = Tempeq_Me2021(M,r,tgrow,alpha,n)
	RS   = 2*G*M/c**2
	rel  = (1-np.sqrt(RS/r))**(1./5)
	return Told*rel

def Surface_Density_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	beta = Beta_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	return Surface_Density_Me2021_aux(beta,M,r,tgrow,alpha,n)

def Sound_Speed_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	beta = Beta_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	f    = fbeta_Me2021(beta,5./3)
	T    = Tempeq_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	return np.sqrt(f/beta*kb/mp*T)

def Mach_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	cs = Sound_Speed_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	v  = np.sqrt(G*M/r)
	return v/cs

def Disk_Height_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	beta = Beta_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	f     = fbeta_Me2021(beta,5./3)
	cs    = Sound_Speed_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	Omega = np.sqrt(G*M/r**3)
	return cs/np.sqrt(f)/Omega

def Pressure_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	beta = Beta_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	f     = fbeta_Me2021(beta,5./3)
	cs    = Sound_Speed_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	Sigma = Surface_Density_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	return cs**2*Sigma/f

def Nu_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	cs    = Sound_Speed_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	h     = Disk_Height_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	return alpha*cs*h

def ToomreQ_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	cs    = Sound_Speed_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	Sigma = Surface_Density_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	Omega = np.sqrt(G*M/r**3)
	return cs*Omega/np.pi/G/Sigma

def Toomre_Me2021_aux(r,M,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	return 1-ToomreQ_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)

def Toomre_Me2021(M,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	RS = 2*G*M/c**2
	return brentq(Toomre_Me2021_aux,RS,1e10*RS,args=(M,tgrow,alpha,n,gas_or_mixed))/RS

def Tvisc_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	Mach = Mach_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	nu   = alpha/Mach**2 * np.sqrt(G*M*r)
	return (2./3) * r**2 / nu

def TGWvisc_Me2021_aux(r,M,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	return Ts_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)-TGW_Haiman2009(M,r)

def TGWvisc_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	RS = 2*G*M/c**2
	return brentq(TGWvisc_Me2021_aux,RS,1e10*RS,args=(M,tgrow,alpha,n,gas_or_mixed))/RS

def Ts_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	mu = M/4 #assumes equal-mass binary
	tv = Tvisc_Me2021(M,3*r,tgrow,alpha,n,gas_or_mixed) #use r = 3*r for cavity wall
	qB = 2*M/tgrow * tv/mu #Eq 18 from Haiman+2009
	k  = 3./8 #Eq 19 from Haiman+2009, assumes qB>1
	return qB**(-k) * tv

def Taueff_Me2021(kap,kapes,M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	tauscat = kapes*Surface_Density_Me2021(M,r,tgrow,0.1,8./3,gas_or_mixed)/2
	tauabs  = kap *Surface_Density_Me2021(M,r,tgrow,0.1,8./3,gas_or_mixed)/2
	return np.sqrt(tauabs*(tauabs+tauscat))

def Prad_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	cs    = Sound_Speed_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	Sigma = Surface_Density_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	beta  = Beta_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	fbeta = fbeta_Me2021(beta,5./3)
	return Sigma*cs**2/fbeta

def Prad_Me2021_2(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	T     = Tempeq_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	h     = Disk_Height_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	return 8./3 * sigma/c * T**4 * h

def Pgas_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	T     = Tempeq_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	Sigma = Surface_Density_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	return Sigma*kb/mp*T

def Internal_Energy_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	pgas = Pgas_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	prad = Prad_Me2021_2(M,r,tgrow,alpha,n,gas_or_mixed)
	ptot = pgas + prad
	beta = Beta_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	fbeta= fbeta_Me2021(beta,5./3)
	#return ptot*(beta/((5./3)-1) + (1-beta)/(fbeta-1))
	return ptot/(fbeta-1)

def TCool_Me2021(M,r,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	U = Internal_Energy_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	T = Tempeq_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	Sigma = Surface_Density_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	return 3./8*kappa*Sigma/sigma/T**4*U

def CoolViscTeq_aux_Me2021(r,M,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	tvisc = Tvisc_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	tcool = TCool_Me2021(M,r,tgrow,alpha,n,gas_or_mixed)
	return tvisc-tcool

def CoolViscTeq_Me2021(M,tgrow,alpha,n=8./3,gas_or_mixed='mixed'):
	RS = 2*G*M/c**2
	return brentq(CoolViscTeq_aux_Me2021,RS,1e10*RS,args=(M,tgrow,alpha,n,gas_or_mixed))/RS

