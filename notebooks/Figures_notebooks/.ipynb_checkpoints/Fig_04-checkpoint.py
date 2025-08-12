import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import gc
import xarray as xr
import pandas as pd
import xrft as xrft
import warnings
import sys
import seawater as sw
sys.path.append("/homes/metogra/iufarias/FeedbackSubmeso/useful/")
import romspickle
import xroms 
from pyspec import spectrum
import datetime

from dask.diagnostics import ProgressBar

import scipy.integrate as integ
warnings.filterwarnings("ignore")



plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 22})

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Lucida Grande']

#functions


def plot_chi_error(ki,spec,sn,color='red',label=None,ci=0.95,alpha=0.25):

    Eu,El=spectrum.spec_error(spec,sn=sn, ci=ci)
    return plt.fill_between(ki,El,Eu, color=color, label=None,alpha=alpha)

def block_avg_spectra(kh_in,spec_h_in,multiple=2):

    spec_h_out=np.nanmean(spec_h_in.to_numpy().reshape([spec_h_in.shape[0],spec_h_in.shape[1]//multiple,multiple]),axis=2)
    kh_out=np.nanmean(kh_in.to_numpy().reshape(spec_h_in.shape[1]//multiple,multiple),axis=1)
    
    return kh_out,np.nanmean(spec_h_out,axis=0)
    


path='/data/pacific/lrenault/SASI/CROCO/FULL/'
listm=glob(path+'surf*.nc');listm.sort()
# varf=xr.open_dataset(listm[-9:][xmon])
varf=xr.open_mfdataset(listm[1:-1],data_vars='minimal')
# varf=varf.expand_dims(['s_rho','s_w'])


path='/data/pacific/lrenault/SASI/CROCO/SMTH/'
listt=glob(path+'surf*SASI_6h_his_20*_*.nc');listt.sort()
#vart=xr.open_dataset(listt[xmon])
vart=xr.open_mfdataset(listt[1:],data_vars='minimal')
# vart=vart.expand_dims('s_rho')




dxx=0.5
dyy=dxx




# t_ind=pd.DatetimeIndex(varf.time_counter.compute())
t_ind=varf.time_counter.groupby("time_counter.dayofyear").mean()
t_ind_h=varf.time_counter


mon_ind=t_ind.dt.month
mes=np.array([12,1,2,3,4,5,6,7])
mes_str=['Dec/11','Jan/12','Feb/12','Mar/12','Apr/12','May/12','Jun/12','Jul/12']




n2_full=xr.open_dataarray('/data/pacific/iufarias/APE_global/N2r_full_profile.nc')
n2_smth=xr.open_dataarray('/data/pacific/iufarias/APE_global/N2r_smth_profile.nc')
B_full=xr.open_dataset('/homes/metogra/iufarias/Documents/data/new_spec/B_full_reference_total.nc')['__xarray_dataarray_variable__'] #multiply by N2r before
B_smth=xr.open_dataset('/homes/metogra/iufarias/Documents/data/new_spec/B_smth_reference_total.nc')['__xarray_dataarray_variable__']

## Calculating surface PE

epe_full=(1/(2*n2_full[:,0].mean()))*B_full[:,0]
epe_smth=(1/(2*n2_smth[:,0].mean()))*B_smth[:,0]

epe_full_z=(1/(2*n2_full[:]))*B_full[:]
epe_smth_z=(1/(2*n2_smth[:]))*B_smth[:]


## Calculating surface KE
eke_full=xr.open_dataset('/homes/metogra/iufarias/Documents/data/new_spec/Eke_full_total.nc')['__xarray_dataarray_variable__'][:,0,:]
eke_smth=xr.open_dataset('/homes/metogra/iufarias/Documents/data/new_spec/Eke_smth_total.nc')['__xarray_dataarray_variable__'][:,0,:]


eke_full_z=xr.open_dataset('/homes/metogra/iufarias/Documents/data/new_spec/Eke_full_total.nc')['__xarray_dataarray_variable__'][:,:,:]
eke_smth_z=xr.open_dataset('/homes/metogra/iufarias/Documents/data/new_spec/Eke_smth_total.nc')['__xarray_dataarray_variable__'][:,:,:]


%%time
chunks_new={'time_counter':1}

uf=varf.u#.groupby("time_counter.dayofyear").mean()
uf=uf.chunk(chunks=chunks_new);
# u_f=(uf[:,1:,:]+uf[:,:-1,:])/2
uf=uf.compute();

ut=vart.u#.groupby("time_counter.dayofyear").mean()
ut=ut.chunk(chunks=chunks_new);
# ut=(ut[:,1:,:]+ut[:,:-1,:])/2
ut=ut.compute();

vf=varf.v#.groupby("time_counter.dayofyear").mean()
vf=vf.chunk(chunks=chunks_new);
# vf=(vf[:,:,1:]+vf[:,:,:-1])/2
vf=vf.compute();

vt=vart.v#.groupby("time_counter.dayofyear").mean()
vt=vt.chunk(chunks=chunks_new);
# vt=(vt[:,:,1:]+vt[:,:,:-1])/2
vt=vt.compute();


uf,vf,_=romspickle.uvw2rho_3d(uf,vf,uf)
uf,vf=uf[:,120:-120,120:800],vf[:,120:-120,120:800]

ut,vt,_=romspickle.uvw2rho_3d(ut,vt,ut)
ut,vt=ut[:,120:-120,120:800],vt[:,120:-120,120:800]


zeta_full=  xr.apply_ufunc(np.gradient,vf,kwargs={'axis':1})/dxx - xr.apply_ufunc(np.gradient,uf.rename({'y_u':'y_v','x_u':'x_v'}),kwargs={'axis':2})/dxx
zeta_smth=  xr.apply_ufunc(np.gradient,vt,kwargs={'axis':1})/dxx - xr.apply_ufunc(np.gradient,ut.rename({'y_u':'y_v','x_u':'x_v'}),kwargs={'axis':2})/dxx


sigma_full=  xr.apply_ufunc(np.gradient,uf.rename({'y_u':'y_v','x_u':'x_v'}),kwargs={'axis':2})/dxx + xr.apply_ufunc(np.gradient,vf,kwargs={'axis':1})/dxx
sigma_smth=  xr.apply_ufunc(np.gradient,ut.rename({'y_u':'y_v','x_u':'x_v'}),kwargs={'axis':2})/dxx + xr.apply_ufunc(np.gradient,vt,kwargs={'axis':1})/dxx



nf=2
wdws='hann' #'flattop','hann'
wdw_cor=True
scl='density'


with ProgressBar():
    zeta_full_ispec=xrft.isotropic_power_spectrum(zeta_full,dim=['x_v','y_v'],nfactor=nf,truncate='True',scaling=scl,detrend='linear', 
                                     window=wdws,window_correction=wdw_cor)
    zeta_smth_ispec=xrft.isotropic_power_spectrum(zeta_smth,dim=['x_v','y_v'],nfactor=nf,truncate='True',scaling=scl,detrend='linear', 
                                     window=wdws,window_correction=wdw_cor)
    sigma_full_ispec=xrft.isotropic_power_spectrum(sigma_full,dim=['x_v','y_v'],nfactor=nf,truncate='True',scaling=scl,detrend='linear', 
                                     window=wdws,window_correction=wdw_cor)
    sigma_smth_ispec=xrft.isotropic_power_spectrum(sigma_smth,dim=['x_v','y_v'],nfactor=nf,truncate='True',scaling=scl,detrend='linear', 
                                     window=wdws,window_correction=wdw_cor)


T_ind=np.int((2*np.pi/sw.f(lat=36.6))/(60*60*6))


multip=3


k_block,eke_full_block=block_avg_spectra(kh_in=eke_full.freq_r,
                  spec_h_in=eke_full,multiple=multip)
k_block,eke_smth_block=block_avg_spectra(kh_in=eke_smth.freq_r,
              spec_h_in=eke_smth,multiple=multip)


_,epe_full_block=block_avg_spectra(kh_in=epe_full.freq_r,
                  spec_h_in=epe_full,multiple=multip)
_,epe_smth_block=block_avg_spectra(kh_in=epe_smth.freq_r,
                  spec_h_in=epe_smth,multiple=multip)


_,sigma_full_block=block_avg_spectra(kh_in=sigma_full_ispec.freq_r,
                  spec_h_in=sigma_full_ispec,multiple=multip)
_,sigma_smth_block=block_avg_spectra(kh_in=sigma_smth_ispec.freq_r,
                  spec_h_in=sigma_smth_ispec,multiple=multip)

_,zeta_full_block=block_avg_spectra(kh_in=zeta_full_ispec.freq_r,
                  spec_h_in=zeta_full_ispec,multiple=multip)
_,zeta_smth_block=block_avg_spectra(kh_in=zeta_smth_ispec.freq_r,
                  spec_h_in=zeta_smth_ispec,multiple=multip)



#plot

dof_block=(eke_full.ocean_time.shape[0]/3)*multip

plt.rcParams['text.usetex'] = True
plt.figure(figsize=(20,8),dpi=300)

plt.subplot(1,2,1)



plt.plot(k_block/dxx,eke_full_block*dxx*1e3,label='Surface EKE (FULL)',linewidth=2,color='cornflowerblue')
plot_chi_error(k_block/dxx,spec=eke_full_block*dxx*1e3,sn=dof_block,color='cornflowerblue')

plt.plot(k_block/dxx,eke_smth_block*dxx*1e3,label='Surface EKE (SMTH)',linewidth=2,color='tomato')
plot_chi_error(k_block/dxx,spec=eke_smth_block*dxx*1e3,sn=dof_block,color='tomato')


plt.plot(k_block/dxx,epe_full_block*dxx*1e3,label='Surface EPE (FULL)',linestyle='--',linewidth=2,color='cornflowerblue')
plot_chi_error(k_block/dxx,spec=epe_full_block*dxx*1e3,sn=dof_block,color='cornflowerblue')
       
plt.plot(k_block/dxx,epe_smth_block*dxx*1e3,label='Surface EPE (SMTH)',linestyle='--',linewidth=2,color='tomato')
plot_chi_error(k_block/dxx,spec=epe_smth_block*dxx*1e3,sn=dof_block,color='tomato')

# plt.plot([1/50,1/50],[1000,0],'k--')

# plt.text(2e-2,0.94e1,r'Surface EKE',color='grey',fontsize=15)
# plt.text(2e-2,0.4e3,r'Surface EPE',color='grey',fontsize=15)


plt.xlabel(r'Wavenumber  [cycle km$^{-1}$]')
plt.ylabel(r'Spectral Density [m$^{3}$s$^{-2}$]')

plt.yscale('log')
plt.xscale('log')

# plt.ylim(1e-4,0.8e3)
# plt.ylim(1e-4,200)

plt.xlim(1e-2,1)
# plt.xlim(2e-2,1)
plt.grid()
plt.title('(a)')
plt.legend(fontsize=15)

plt.subplot(1,2,2)


plt.plot(k_block/dxx,sigma_full_block*dxx*1e3,label='Surface $\widehat{\sigma^2}$ (FULL)',linestyle='--',linewidth=2,color='cornflowerblue')
plot_chi_error(k_block/dxx,spec=sigma_full_block*dxx*1e3,sn=dof_block,color='cornflowerblue')


plt.plot(k_block/dxx,sigma_smth_block*dxx*1e3,label='Surface $\widehat{\sigma^2}$ (SMTH)',linestyle='--',linewidth=2,color='tomato')
plot_chi_error(k_block/dxx,spec=sigma_smth_block*dxx*1e3,sn=dof_block,color='tomato')


plt.plot(k_block/dxx,zeta_full_block*dxx*1e3,label='Surface $\widehat{\zeta^2}$ (FULL)',linewidth=2,color='cornflowerblue')
plot_chi_error(k_block/dxx,spec=zeta_full_block*dxx*1e3,sn=dof_block,color='cornflowerblue')


plt.plot(k_block/dxx,zeta_smth_block*dxx*1e3,label='Surface $\widehat{\zeta^2}$ (SMTH)',linewidth=2,color='tomato')
plot_chi_error(k_block/dxx,spec=zeta_smth_block*dxx*1e3,sn=dof_block,color='tomato')


plt.legend(fontsize=15)

plt.xlabel(r'Wavenumber  [cycle km$^{-1}$]')
plt.ylabel(r'Spectral Density [m s$^{-2}$]')

# plt.yscale('symlog',linthresh=1e-6)
plt.yscale('log')
plt.xscale('log')

# plt.ylim(0,2.5)
plt.xlim(1e-2,1)
plt.grid()
plt.title('(b)')
plt.tight_layout()

# plt.savefig('/homes/metogra/iufarias/Documents/figures/2_CROCO/CROCO_surface/total_avg/div_vort_spectral_both.png',dpi=400)
