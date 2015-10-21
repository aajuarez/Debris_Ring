######################################################
######### Aaron J. Juarez, Oct 11--20 2015 ###########
######################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import newton, fsolve
font = {'family':'serif', 'size':12}
plt.rc('font', **font)

#### constants and parameters, CGS
c = 2.99792458e10    # Speed of light
h = 6.6260755e-27    # Plank's constant
k = 1.380658e-16     # Boltzmann's constant
G = 6.67259e-8       # Universal G
Jy = 1e-23 # erg s^-1 cm^-2 Hz^-1

## wavelengths: 0.001 - 1000 micron
#wavelen_min = 1e-9      # meters
wavelen_min = 7e-8
wavelen_max = 1e-3
#wavelen_min = 1.0e-7
#wavelen_max = 6.0e-5
wavelen_min, wavelen_max = wavelen_min*1e2, wavelen_max*1e2 # m --> cm

## star parameters
T_star = 8600  # Kelvin, Fomalhaut
R_star = 1.842 * 6.96e10 # R_sun --> cm
M_star = 1.92 * 1.99e33 #M_sub --> g
distance = 24.8 * 9.461e17 # ly --> cm, pm 0.1 ly
distance = 10 * 1.496e13 # AU --> cm
distance = 130 *1.496e13 # AU --> cm

###################################################### functions and global variables
N = 8e4 #resolution
wavelengths = np.linspace(wavelen_min, wavelen_max, N)
frequencies = c/wavelengths
dw = wavelengths[1]-wavelengths[0]
df = []
for i in range(len(frequencies)-1): df.append(abs(frequencies[i]-frequencies[i+1]))
wavelengths=wavelengths[:-1]
frequencies=frequencies[:-1]
angrad = np.arcsin(R_star/distance)

# Calculate Plank spectrum and write output to file
#file = open('spectrum.dat', 'w')
#for i in range(len(frequencies)):
#    radiance = (2 * h * c**2 / wavelengths[i]**5) * (1.0 / (np.exp(h*c/(wavelengths[i]*k*T)) - 1.0))
#    radiance = (2 * h * frequencies[i]**3 / c**2) * (1.0 / (np.exp(h * frequencies[i] /k/T) - 1.0))
#    flux_dens = np.pi * radiance * angrad**2
#    file.write("%g %g %g %g\n" % (wavelengths[i], frequencies[i], radiance, flux_dens))
#file.close()

def radiance_lambda(wavelengths,T):
    return (2 * h * c**2 / wavelengths**5) * (1.0 / (np.exp(h*c/(wavelengths*k*T)) - 1.0))

def radiance_nu(frequencies,T):
    return (2 * h * frequencies**3 / c**2) * (1.0 / (np.exp(h * frequencies /k/T) - 1.0))

def P_in(a, F_nu, Q_abs):
#    P_abs = (0.5*a/distance)**2 * np.pi * B_nu * Q_abs * df
    P_abs = np.pi * a**2 * F_nu * Q_abs * df
    return P_abs

def P_out(a,B_dust,Q_abs):
    return 4 * np.pi**2 * a**2 * B_dust * Q_abs * df

def T_eq(T,a,frequencies,P_in):
    def Power_eq(Tdust,a,P_in,frequencies):
        return P_in.sum() - sum(4*np.pi**2 * a**2 * radiance_nu(frequencies,Tdust) * df)
    T = newton(Power_eq, T, args=(a,P_in,frequencies)) #T_dust, solved by Newton-Raphson method
#    T = fsolve(Power_eq, T, args=(a,frequencies,P_in)) #T_dust, by fsolve; does not work. hmm.
    return T

###################################################### B_nu, B_lambda, F_nu
rad_fr = radiance_nu(frequencies,T_star)
rad_wv = radiance_lambda(wavelengths,T_star)
flux_dens = np.pi * rad_fr * angrad**2 # sin(angrad) = angrad

if distance==10*1.496e13:
    switch='10'
    F_peak_fomalhaut=5e2*1e-3 # mJy --> Jy; Su et al. 2013
else:
    switch='130'
    F_peak_fomalhaut=1e4*1e-3

######################################################
#### Read out data file for a, wavelengths, and Q values
filename='suvSil_21'
a, wv, Q_absarr = [],[],[]
with open(filename, 'r') as f:
    content = f.readlines()
    for i in range(5,len(content)):
        line = content[i].strip().split()
        if len(line)==0: count=0; continue
        else:
            if count==0: a.append(line[0]); count+=1 #grain radius [micron]
            elif count==1: count+=1
            else:  # w(micron)   Q_abs   Q_sca   g=<cos>
                w,Q_abs,Q_sca,avcos = line[0],line[1],line[2],line[3]
                wv.append(w); Q_absarr.append(Q_abs)

a=np.array(a, dtype='float')*1e-4 # micron --> cm
wv=np.array(wv, dtype='float')*1e-4
Q_absarr=np.array(Q_absarr, dtype='float')
indz = np.arange(0,len(wv)+len(wv)/len(a),len(wv)/len(a)); #print indz
col=np.linspace(0,1,len(indz))

###################################################### Part 1: B_nu, F_nu
#'''
fig = plt.figure()
g=fig.add_subplot(111,axisbg='0.87')
g.plot(wavelengths*1e4,rad_fr,'k-')
g.plot(wavelengths*1e4,rad_wv*(c/frequencies**2),'b--')
g.set_xlabel(r'$\lambda\ [\mu \rm m]$',size=16)
g.set_ylabel(r'$B_\nu\ [\rm erg\ s^{-1}\ cm^{-2}\ sr^{-1}\ Hz^{-1}]$',size=16)
g.set_yscale('log');g.set_xscale('log')
plt.tight_layout()
plt.savefig('B_nu_wv.pdf')

fig = plt.figure()
g=fig.add_subplot(111,axisbg='0.87')
g.plot(frequencies,rad_fr,'k-')
g.plot(frequencies,rad_wv*(c/frequencies**2),'b--')
g.set_xlabel(r'$\nu\ [\rm Hz]$',size=16)
g.set_ylabel(r'$B_\nu\ [\rm erg\ s^{-1}\ cm^{-2}\ sr^{-1}\ Hz^{-1}]$',size=16)
g.set_yscale('log');g.set_xscale('log')
plt.tight_layout()
plt.savefig('B_nu_fr.pdf')

fig = plt.figure()
g=fig.add_subplot(111,axisbg='0.87')
g.plot(wavelengths*1e4,flux_dens/Jy,'k-')
g.set_xlabel(r'$\lambda\ [\mu \rm m]$',size=16)
g.set_ylabel(r'$F_\nu\ [\rm Jy]$',size=16)
g.set_yscale('log');g.set_xscale('log')
plt.tight_layout()
plt.savefig('F_nu_wv_'+switch+'.pdf')
#'''

###################################################### Part 2: P_in
#### Interpolate for Q value given input wavelengths
fig = plt.figure(figsize=(8,6)); g=fig.add_subplot(111,axisbg='0.57')
#fig = plt.figure(figsize=(8,6));g1=fig.add_subplot(111,axisbg='0.57')
P_sum = 0; Q_abs_array=[]
for j in range(1,len(indz)):
#    if a[j-1] not in [0.1*1e-4, 1*1e-4, 10*1e-4]: continue
    if a[j-1] not in [0.1*1e-4, 1*1e-4, 10*1e-4]: lw=3; ls=':'; text=''
    else: lw=2.5; ls='-'; text=r'$%.1f$'%(a[j-1]*1e4)
    i1=int(indz[j-1]); i2=int(indz[j])
    Q_i = Q_absarr[i1:i2][::-1]
    wv_i = wv[i1:i2][::-1]
#    print j, a[j-1], len(wv_i)
    Q_abs_interp = np.interp(wavelengths, wv_i, Q_i)
    Q_abs_array.append(Q_abs_interp)
    P_abs = P_in(a[j-1],flux_dens,Q_abs_interp)
#    g.plot(wavelengths*1e4,P_abs,'k-',c=cm.Spectral_r(col[j]),lw=lw,ls=ls)             #plot P_in
#    g.text(np.mean(wavelengths*1e4),P_abs[-1]*50, text, rotation=-30)                  #text
#    g.text(np.mean(wavelengths*1e4),P_abs[-1], r'$%.2E$'%(a[j-1]*1e4), rotation=-30)   #alt text
    g.plot(a[j-1]*1e4,P_abs.sum(),'ko',c=cm.Spectral_r(col[j]))                        #plot P_tot
    g.text(a[j-1]*1e4,P_abs.sum(), text, rotation=-50)                                 #text
#    g1.plot(wavelengths*1e4,Q_abs_interp,'k-',c=cm.Spectral_r(col[j]),lw=1.5)
#    g1.plot(wv_i,Q_i,'k-',c=cm.Spectral_r(col[j]),lw=1.5)
#    P_sum += P_abs**2
#P_sum = np.sqrt(P_sum)

a1mm = 1e3*1e-4 # 1 mm radius dust grain
Q_1mm = 1.0 #perfect absorber
P_1mm = P_in(a1mm, flux_dens, Q_1mm)
#g.plot(wavelengths*1e4,P_1mm,'k-',c='darkred',lw=2.5)                               #P_in
#g.text(np.mean(wavelengths*1e4),P_1mm[-1]*20, r'$%.f$'%(a1mm*1e4), rotation=-20)
asd=g.plot(a1mm*1e4,P_1mm.sum(),'ko',c='darkred',lw=2.5)                           #P_tot
g.text(a1mm*1e4,P_1mm.sum(), r'$%.f$'%(a1mm*1e4), rotation=-50)
asd[0].set_clip_on(False)

#g.set_xlabel(r'$\lambda\ [\mu \rm m]$',size=16)
#g.set_ylabel(r'$P_{\rm in}\ [\rm erg\ s^{-1}\ Hz^{-1}]$',size=16)
#plt.xlim([7e-2,1e3])
g.set_xlabel(r'$a\ [\mu \rm m]$',size=16)
g.set_ylabel(r'$P_{\rm in, tot}\ [\rm erg\ s^{-1}]$',size=16)
g.set_yscale('log')
g.set_xscale('log')
plt.tight_layout()
#plt.savefig('P_in_'+switch+'.pdf')         #P_in
plt.savefig('P_in_'+switch+'tot.pdf')       #P_tot
#plt.show();exit()

#### P_in = P_out --> solve for dust equilibrium temperature
T_d1mm = T_eq(100,a1mm,frequencies,P_1mm); #print T_d1mm
radiance_d1mm = radiance_nu(frequencies,T_d1mm)
Pbal = P_1mm.sum() - P_out(a1mm, radiance_d1mm, 1).sum()

###################################################### Part 3: T_equilibrium
fig = plt.figure(figsize=(8,6)); g=fig.add_subplot(111,axisbg='0.57')
print 'n\t'+'a\t\t'+'Tdust\t\t'+'P_balance'
T_dust=[]
for j in range(1,len(indz)):
#    if a[j-1] not in [0.1*1e-4, 1*1e-4, 10*1e-4]: continue
    if a[j-1] not in [0.1*1e-4, 1*1e-4, 10*1e-4]: text=''
    else: text=r'$%.1f$'%(a[j-1]*1e4)
    Q_abs_interp = Q_abs_array[j-1]
    P_abs = P_in(a[j-1], flux_dens, Q_abs_interp)
    Tdust = T_eq(100,a[j-1],frequencies,P_abs)
    T_dust.append(Tdust)
    rad_dust = radiance_nu(frequencies,Tdust)
    P_emit = P_out(a[j-1],rad_dust,Q_abs_interp)
    Power_bal= P_abs.sum() - P_emit.sum()
    asd=g.plot(Tdust,Power_bal,'ko',c=cm.Spectral_r(j*1./len(indz)))
    g.text(Tdust,Power_bal+0.025, text, rotation=90, va='top')
    print '%i\t%e\t%f\t%f'%(j, a[j-1]*1e4, Tdust, Power_bal)
    asd[0].set_clip_on(False)
asd=g.plot(T_d1mm,Pbal,'ko',c='darkred')
g.text(T_d1mm,Pbal+0.025,r'$%.1f$'%(a1mm*1e4), rotation=90)
print '%i\t%e\t%f\t%f'%(22, a1mm*1e4, T_d1mm, Pbal)
asd[0].set_clip_on(False)
'''
#Dumb Check, P_out = L = 4 pi r^2 sigma T^4
fig = plt.figure(figsize=(8,6)); g=fig.add_subplot(111,axisbg='0.57')
T_dust_ = np.linspace(10,200,40)
for i in range(len(T_dust_)):
    rad_dust = radiance_nu(frequencies,T_dust_[i])
    P_out = 4 * np.pi * a1mm**2 * np.pi * rad_dust * 1 * df
    Power_bal= P_1mm.sum() - P_out.sum()
    g.plot(T_dust_[i],Power_bal,'ko')
#'''
g.set_xlabel(r'$T\ [\rm K]$',size=16)
g.set_ylabel(r'$\rm Energy\ Balance$',size=16)
plt.tight_layout()
plt.savefig('energy_balance_'+switch+'.pdf')
#exit()
###################################################### Part 4: SED --> grain numbers/masses
fig = plt.figure(); g=fig.add_subplot(111,axisbg='0.57')
flux_peak = []
for j in range(1,len(indz)):
#    if a[j-1] not in [0.1*1e-4, 1*1e-4, 10*1e-4]: continue
    if a[j-1] not in [0.1*1e-4, 1*1e-4, 10*1e-4]: lw=3; ls=':'
    else: lw=2.5; ls='-'
    Q_abs_interp = Q_abs_array[j-1]
    P_abs = P_in(a[j-1], flux_dens, Q_abs_interp)
    rad_dust = radiance_nu(frequencies,T_dust[j-1])
    F_nu_dust= np.pi * rad_dust * (a[j-1]/distance)**2 * Q_abs_interp
    flux_peak.append(np.max(F_nu_dust/Jy))
#    g.plot(wavelengths*1e4,F_nu_dust/Jy,'k-',c=cm.Spectral_r(j*1./len(indz)),lw=lw)
    g.plot(frequencies,F_nu_dust/Jy,'k-',c=cm.Spectral_r(j*1./len(indz)),lw=lw,ls=ls)
F_nu_d1mm= np.pi * radiance_d1mm * (a1mm/distance)**2 * Q_1mm
g.plot(frequencies,F_nu_d1mm/Jy,'k-',c='darkred',lw=2.5)
#g.set_xlabel(r'$\lambda\ [\mu \rm m]$',size=16)
g.set_xlabel(r'$\nu\ [\rm Hz]$',size=16)
g.set_ylabel(r'$F_\nu\ [\rm Jy]$',size=16)
g.set_yscale('log');g.set_xscale('log')
if distance==10*1.496e13:
    plt.xlim(3e11,3e14);plt.ylim(1e-44,1e-15)
else:
    plt.xlim(3e11,1e14);plt.ylim(1e-50,1e-18)
plt.tight_layout()
plt.savefig('dust_sed_'+switch+'.pdf')

density = 2. #g cm^-3
N_grains = F_peak_fomalhaut/np.array(flux_peak)
M_grains = N_grains * density * 4/3. * np.pi * a**3

N_g1mm = F_peak_fomalhaut/np.max(F_nu_d1mm/Jy)
M_g1mm = N_g1mm * density * 4/3. * np.pi * a1mm**3
print; print 'n\t'+'a\t\t'+'N_grains\t'+'M_grains'
for i in range(len(a)):
    print '%i\t%e\t%e\t%e'%(i+1, a[i]*1e4, N_grains[i], M_grains[i])
print '%i\t%e\t%e\t%e'%(22, a1mm*1e4, N_g1mm, M_g1mm)

fig = plt.figure(); g=fig.add_subplot(111,axisbg='0.57')
for i in range(len(a)):
    asd=g.plot(a[i]*1e4, N_grains[i], 'ko', c=cm.Spectral_r(i*1./len(indz)))
    asd[0].set_clip_on(False)
asd=g.plot(a1mm*1e4, N_g1mm, 'ko', c='darkred')
asd[0].set_clip_on(False)
g.set_xlabel(r'$a\ [\mu \rm m]$',size=16)
g.set_ylabel(r'$N_{\rm grains}$',size=16)
g.set_yscale('log')
g.set_xscale('log')
plt.tight_layout()
plt.savefig('dust_N_'+switch+'.pdf')

fig = plt.figure(); g=fig.add_subplot(111,axisbg='0.57')
for i in range(len(a)):
    asd=g.plot(a[i]*1e4, M_grains[i], 'ko', c=cm.Spectral_r(i*1./len(indz)))
    asd[0].set_clip_on(False)
asd=g.plot(a1mm*1e4, M_g1mm, 'ko', c='darkred')
asd[0].set_clip_on(False)
g.set_xlabel(r'$a\ [\mu \rm m]$',size=16)
g.set_ylabel(r'$M_{\rm grains}\ [\rm g]$',size=16)
g.set_yscale('log')
g.set_xscale('log')
plt.tight_layout()
plt.savefig('dust_M_'+switch+'.pdf')

###################################################### Part 5
#### Grain removal (via radiation pressure and Poynting-Robertson drag)
print; print 'i time [years]'; timez=[]
for j in range(1,len(indz)):
    Q_abs_interp = Q_abs_array[j-1]
    P_abs = P_in(a[j-1], flux_dens, Q_abs_interp)
    Frad = P_abs.sum()/c
    Brad = Frad * distance**2 / (G * M_star * density * 4/3. *np.pi* a[j-1]**3)
    F_PR = P_abs.sum()/ c**2 * np.sqrt(G * M_star/distance)
#    B_PR = F_PR * distance**2 / (G * M_star * density * 4/3. *np.pi* a[j-1]**3)
    time = (4e2/(M_star/1.99e33)) * (distance/1.496e13)**2 / Brad #years
    timez.append(time)
    print j, time
Frad_d1mm = P_1mm.sum()/c
Brad = Frad_d1mm * distance**2 / (G * M_star * density * 4/3. *np.pi* a1mm**3)
F_PR = P_1mm.sum()/ c**2 * np.sqrt(G * M_star/distance)
time = (4e2/(M_star/1.99e33)) * (distance/1.496e13)**2 / Brad #years; Wyatt pdf
print j+1, time

fig = plt.figure(); g=fig.add_subplot(111,axisbg='0.57')
for i in range(len(a)):
    asd=g.plot(a[i]*1e4, timez[i], 'ko', c=cm.Spectral_r(i*1./len(indz)))
    asd[0].set_clip_on(False)
asd=g.plot(a1mm*1e4, time, 'ko', c='darkred')
asd[0].set_clip_on(False)
g.set_xlabel(r'$a\ [\mu \rm m]$',size=16)
g.set_ylabel(r'$\tau\ [\rm years]$',size=16)
g.set_yscale('log')
g.set_xscale('log')
plt.tight_layout()
plt.savefig('time_rm_'+switch+'.pdf')

plt.show()
