#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Packages 

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime 
from matplotlib.gridspec import GridSpec
# import seawater as sw
import cmocean as cm
import seaborn as sb
import seaborn as sns
import xrft
import pandas as pd
import cartopy.crs as ccrs
import cartopy
import statistics as stat
import scipy

from scipy.ndimage import generic_filter
from scipy.ndimage import gaussian_filter


# In[2]:


def along_wind_derivative(field,dx,dy,u_wind_ref,v_wind_ref,axis_x=1,axis_y=0): #field is the variable you are looking at, this could be sst,wind, du'/dr etc 
    #ref_winds are background reference
    
    d_field_dx=xr.apply_ufunc(np.gradient,field, kwargs={'axis': axis_x})/dx #np.gradient is like taking a derivative 
    d_field_dy=xr.apply_ufunc(np.gradient,field, kwargs={'axis': axis_y})/dy #the ufunc will keep the shape of the array 
    
    ws=np.sqrt(u_wind_ref**2+v_wind_ref**2) #the wind speed 

    d_field_dr= d_field_dx*(u_wind_ref/ws) + d_field_dy*(v_wind_ref/ws) #d_/dx*my average winds/my average winds + the same in the y direction 
    return d_field_dr #return this new equation 


# In[3]:


def simple_fft_filter(FIELD,DX,DY,coord_x,coord_y,filter_lenght=5): 

    dX=float(np.diff(FIELD[coord_x]).mean()) #obtaining the average (bar) of the derivatives 
    dY=float(np.diff(FIELD[coord_y]).mean()) #obtaining the average of the derivative should be around .25


    WL=filter_lenght #applying the wavelength 

    if FIELD[coord_x].shape[0]==FIELD.shape[1]:
        nX = FIELD.T.shape[0]
        nY = FIELD.T.shape[1]
    else:
        nX = FIELD.shape[0]
        nY = FIELD.shape[1]
    mX = nX//2 #midpoints
    mY = nY//2


    # Wavelength corresponding to a given wavenumber is determined by smallest dimension
    if nX > nY:
        WN = dY*nX / WL
    else:
        WN = dX*nX / WL

    if FIELD[coord_x].shape[0]==FIELD.shape[1]:
        FT=xr.apply_ufunc(np.fft.fft2,FIELD.T)
    else:
        FT=xr.apply_ufunc(np.fft.fft2,FIELD)


    i = 0
    for j in range(1,mY+1):
        if np.sqrt( i**2 + j**2 ) > WN:
            FT[i,j] = 0
            FT[i,nY-j] = 0
    j = 0
    for i in range(1,mX+1):
        if np.sqrt( i**2 + j**2 ) > WN:
            FT[i,j] = 0
            FT[nX-i,j] = 0

    # Then take care of the remaining wavenumbers
    for i in range(1,mX+1):
        for j in range(1,mY+1):
            if np.sqrt( i**2 + j**2 ) > WN:
                FT[i,j] = 0
                FT[nX-i,j] = 0
                FT[i,nY-j] = 0
                FT[nX-i,nY-j] = 0
    if FIELD[coord_x].shape[0]==FIELD.shape[1]:
        T_filt = xr.apply_ufunc(np.fft.ifft2,FT).T
    else:
        T_filt = xr.apply_ufunc(np.fft.ifft2,FT)

    T_filt = T_filt.real
    return T_filt


# In[4]:


##Importing the Data from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
#The data points are 47N, 20S, -78W, and -55E 
var=xr.open_dataset('2005.mars.internal-1701126797.321937-20867-2-ffb8a2f2-54fa-47bc-a74f-58938a8fab83.nc')


# In[5]:


#grouping the data by month and getting the average 
sst=var['sst'][:,26:,18:].groupby('time.month').mean()-273.15
uwind=var['u10'][:,26:,18:].groupby('time.month').mean()
vwind=var['v10'][:,26:,18:].groupby('time.month').mean()
wind=np.sqrt(uwind**2+vwind**2)
pbl=var['blh'][:,26:,18:].groupby('time.month').mean()


# In[6]:


#making copies of the variables in order to filter them
sst_total=sst
u_wind_total=uwind
v_wind_total=vwind
wind_total=wind
pbl_total=pbl 



sst_filt=sst_total.copy()*np.nan
u_wind_filt=u_wind_total.copy()*np.nan
v_wind_filt=v_wind_total.copy()*np.nan
pbl_filt=pbl_total.copy()*np.nan

#applying the filter
wfl=20 #amount of point grids to have 5 degrees 
for i in range(sst_total.shape[0]):
    sst_filt[i]= xr.apply_ufunc(generic_filter,sst_total[i], np.mean, [wfl,wfl]).data
    u_wind_filt[i]= xr.apply_ufunc(generic_filter,u_wind_total[i], np.mean, [wfl,wfl]).data
    v_wind_filt[i]= xr.apply_ufunc(generic_filter,v_wind_total[i], np.mean, [wfl,wfl]).data
    pbl_filt[i]=xr.apply_ufunc(generic_filter,pbl_total[i], np.mean, [wfl,wfl]).data

#Making the anomalies of the SST 
sst_anom=sst_total-sst_filt
sst_anom_neg= sst_anom.where(sst_anom>0)
sst_anom_pos=sst_anom.where(sst_anom<0)

#Making the PBL anomalies 
pbl_anom=pbl_total-pbl_filt


# ## Computing the pertubation winds that are coherent with the background winds ($u_r^{\prime}$ or $\frac{dr^{\prime}}{dt}$)

# In[7]:


u_wind_prime=u_wind_total-u_wind_filt #u-Ubar
v_wind_prime=v_wind_total-v_wind_filt
ur_prime= (u_wind_prime*(u_wind_filt) + v_wind_prime*(v_wind_filt))/(np.sqrt(u_wind_filt**2 + v_wind_filt**2))


# # Computing the Latitude and Longitude

# In[8]:


lon=u_wind_total['longitude'][:]
lat=u_wind_total['latitude'][:]
deg2rad=np.pi/180

#make a grid 
lons,lats=np.meshgrid(lon,lat)

dx=lat.copy()*np.nan;

for i in range(lat.shape[0]): 
    dx[i]=(np.diff(lon.data)*(111000)*np.cos(np.deg2rad(lat.data[i])))[0]
    
    
dy=0.25*111000


# # Computing the along wind derivatives 

# In[9]:


dsst_dr = along_wind_derivative(sst_anom,
                                dx=(dx),dy=(dy), #diff lon mean gives one value 
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

d_ur_dr = along_wind_derivative(ur_prime,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

dpbl_dr = along_wind_derivative(pbl_anom,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

#Raveling the data to graph 
ixx=np.arange(dsst_dr.shape[0])
ravel_sst=np.ravel(dsst_dr[ixx])
ravel_wind=np.ravel(d_ur_dr[ixx])
ravel_pbl=np.ravel(dpbl_dr[ixx])

ravel_sst_copy=np.ravel((dsst_dr)[ixx])
ravel_sst=ravel_sst_copy[~np.isnan(ravel_sst_copy)] #now there is a vector of data without any data in it 
ravel_wind_copy=np.ravel((d_ur_dr)[ixx])
ravel_wind=ravel_wind_copy[~np.isnan(ravel_sst_copy)]
ravel_pbl_copy=np.ravel((dpbl_dr)[ixx])
ravel_pbl=ravel_pbl_copy[~np.isnan(ravel_sst_copy)]

m_s=[]
for icc in range(dsst_dr.shape[0]):
    xravel_sst_copy=np.ravel(dsst_dr[icc,:,:])
    xravel_sst=xravel_sst_copy[~np.isnan(xravel_sst_copy)]
    xravel_wind_copy=np.ravel(d_ur_dr[icc,:,:])
    xravel_wind=xravel_wind_copy[~np.isnan(xravel_sst_copy)]
    m_s.append(np.polyfit(xravel_sst,xravel_wind,deg=1)[0])
m_s=np.array(m_s)


# In[10]:


data_f=pd.DataFrame({'sst':ravel_sst,'wind':ravel_wind,'pbl':ravel_pbl})


# # Parcing

# In[11]:


#Making the masks 


pbl_mask1=(pbl<800) #These are just random values, put in the values you want to test
pbl_mask2=((pbl>=800)&(pbl<=1400))
dummy_pbl=pbl.copy()
dummy_sst=sst.copy()
dummy_wind=wind.copy()

mask_pbl=dummy_pbl.where(pbl_mask1)
mask_sst=dummy_sst.where(pbl_mask1) 
mask_wind=dummy_wind.where(pbl_mask1)

mask2_pbl=dummy_pbl.where(pbl_mask2)
mask2_sst=dummy_sst.where(pbl_mask2)
mask2_wind=dummy_wind.where(pbl_mask2)


# In[12]:


dsst_m_dr= along_wind_derivative(mask_sst, #m stands for mask 
                                dx=(dx),dy=(dy), #diff lon mean gives one value 
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)


dpbl_m_dr = along_wind_derivative(mask_pbl,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

dur_m_dr=along_wind_derivative(mask_wind,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

dsst_m2_dr= along_wind_derivative(mask2_sst,
                                dx=(dx),dy=(dy), #diff lon mean gives one value 
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)


dpbl_m2_dr = along_wind_derivative(mask2_pbl,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)
dur_m2_dr=along_wind_derivative(mask2_wind,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

#Raveling the masked data
ravel_sst_mask=np.ravel(dsst_m_dr[ixx])
ravel_wind_mask=np.ravel(dur_m_dr[ixx])
ravel_pbl_mask=np.ravel(dpbl_m_dr[ixx])

ravel_sst_mask2=np.ravel(dsst_m2_dr[ixx])
ravel_pbl_mask2=np.ravel(dpbl_m2_dr[ixx])
ravel_wind_mask2=np.ravel(dur_m2_dr[ixx])

#have to put into a dataframe in order to use regplot 
data_m=pd.DataFrame({'sst_m':ravel_sst_mask,'pbl_m':ravel_pbl_mask,'u_m':ravel_wind_mask,'sst_m2':ravel_sst_mask2,'pbl_m2':ravel_pbl_mask2,'u_m2':ravel_wind_mask2}) #Data m means masked data 



# ## Getting the warm-cold & Cold-warm cases

# In[13]:


# the cold to warm case depends on the SST so first we have to make a copy of the derivative 
dsst_dr_dummy=dsst_dr.copy()
dsst_dr_pos=dsst_dr_dummy.where(dsst_dr_dummy>0) #positive SST values 
dsst_dr_neg=dsst_dr_dummy.where(dsst_dr_dummy<0) #Negative SST Values 


# In[14]:


##For the function below, uncomment which method you'd like to calcuate### 
#There is one for mean and standard deviation which means one bound is all values below one STD from the mean, one +- STD from the mean, and all values above one STD from the mean 
# Another is based on the 25th and 75th percentiles 
# the last is values I picked based on the histogram 

###PBL masks####
########

pbl_mean=np.mean(pbl) 
pbl_std=np.std(pbl)

ravel_pbl_total=np.ravel(pbl_total[ixx])

pbl_25=np.percentile(ravel_pbl_total,25)
pbl_75=np.percentile(ravel_pbl_total,75)

###################
#for the percentiles 
pbl_lb=pbl_25
pbl_ub=pbl_75

# pbl_lb=(pbl<lb)
# pbl_mb=((pbl>=pbl_lb)&(pbl<=pbl_ub))
# pbl_ub=(pbl>ub)

############
#for the STD 
# pbl_lb=pbl_mean-pbl_std
# pbl_ub=pbl_mean+pbl_std

# pbl_bstd=(pbl<lb)
# pbl_btstd=((pbl>=lb)&(pbl<=ub))
# pbl_abstd=(pbl>ub)

##########
#my choice
# lb=600
# ub=1200

# pbl_lb=((500<pbl)&(pbl<lb)) 
# pbl_mb=((pbl>=lb)&(pbl<=1000))
# pbl_ub=(pbl>ub)


####Wind Masks####
############

wind_mean=np.mean(wind)
wind_std=np.std(wind)

ravel_wind_total=np.ravel(wind[ixx])
wind_lb=np.percentile(ravel_wind_total, 25)
wind_ub=np.percentile(ravel_wind_total, 75)
# plt.hist(ravel_pbl_total,bins=30)

# lb_wind=wind_25
# ub_wind=wind_75

# wind_lb=2.5
# wind_ub=5

# lb_wind=wind_mean-wind_std
# ub_wind=wind_mean+wind_std

# lb=pbl_mean-pbl_std
# ub=pbl_mean+pbl_std

# wind_lb=(wind<lb_wind) 
# wind_mb=((wind>=lb_wind)&(wind<=ub_wind))
# wind_ub=(wind>ub_wind)


# In[15]:


#making new derivatives 
d_ur_pos_dr=d_ur_dr.where(dsst_dr_dummy>0) #wind values corresponding to positive SST values 
d_ur_neg_dr=d_ur_dr.where(dsst_dr_dummy<0) #wind values corresponding to negative SST values 

dpbl_pos_dr=dpbl_dr.where(dsst_dr_dummy>0) #PBL values corresponding to positive SST values 
dpbl_neg_dr=dpbl_dr.where(dsst_dr_dummy<0) #PBL values corresponding to negative SST values 

ravel_sst_pos=np.ravel(dsst_dr_pos[ixx])
ravel_ur_pos=np.ravel(d_ur_pos_dr[ixx])
ravel_ur_neg=np.ravel(d_ur_neg_dr[ixx])

ravel_sst_neg=np.ravel(dsst_dr_neg[ixx])
ravel_pbl_pos=np.ravel(dpbl_pos_dr[ixx])
ravel_pbl_neg=np.ravel(dpbl_neg_dr[ixx])

# ravel_pbl_posnp.ma.masked_where(pbl<500)


data_sst_pos=pd.DataFrame({'pbl_pos':ravel_pbl_pos,'ur_pos':ravel_ur_pos,'sst_pos':ravel_sst_pos})
data_sst_neg=pd.DataFrame({'pbl_neg':ravel_pbl_neg,'ur_neg':ravel_ur_neg,'sst_neg':ravel_sst_neg})

pbl_pos_df=data_sst_pos['pbl_pos'].values
ur_pos_df=data_sst_pos['ur_pos'].values
sst_pos_df=data_sst_pos['sst_pos'].values

pbl_neg_df=data_sst_neg['pbl_neg'].values
ur_neg_df=data_sst_neg['ur_neg'].values
sst_neg_df=data_sst_neg['sst_neg'].values


#making dataframes for the positive and negative values 
data_sst_pos=pd.DataFrame({'pbl_pos':ravel_pbl_pos,'ur_pos':ravel_ur_pos,'sst_pos':ravel_sst_pos})
data_sst_neg=pd.DataFrame({'pbl_neg':ravel_pbl_neg,'ur_neg':ravel_ur_neg,'sst_neg':ravel_sst_neg})

pbl_pos_df=data_sst_pos['pbl_pos'].values
ur_pos_df=data_sst_pos['ur_pos'].values
sst_pos_df=data_sst_pos['sst_pos'].values

pbl_neg_df=data_sst_neg['pbl_neg'].values
ur_neg_df=data_sst_neg['ur_neg'].values
sst_neg_df=data_sst_neg['sst_neg'].values


# In[16]:


# Separating into the bounds (there is a lower (lb), middle (mb), and upper bound (ub)) these are based on the masks set above 
#PBL Masks 

##PBL Positive 
pbl_pos_pbl_lb=np.ma.masked_where(ravel_pbl_total>pbl_lb, ravel_pbl_pos) #positive SST values with PBL thresholds 
pbl_pos_pbl_mb=np.ma.masked_where(np.logical_and(ravel_pbl_total<pbl_lb ,ravel_pbl_total>pbl_ub), ravel_pbl_pos) #hid everything greater than upper bound
pbl_pos_pbl_ub=np.ma.masked_where(ravel_pbl_total<pbl_ub, ravel_pbl_pos) #positive SST values with PBL thresholds 

pbl_pos_pbl_lb =pbl_pos_pbl_lb[~pbl_pos_pbl_lb.mask]
pbl_pos_pbl_mb =pbl_pos_pbl_mb[~pbl_pos_pbl_mb.mask]
pbl_pos_pbl_mb=np.ravel(pbl_pos_pbl_mb)
pbl_pos_pbl_ub =pbl_pos_pbl_ub[~pbl_pos_pbl_ub.mask]

#SST Positive
sst_pos_pbl_lb=np.ma.masked_where(ravel_pbl_total>pbl_lb, ravel_sst_pos) #positive SST values with PBL thersholds 
sst_pos_pbl_mb=np.ma.masked_where(np.logical_and(ravel_pbl_total<pbl_lb ,ravel_pbl_total>pbl_ub), ravel_sst_pos)
sst_pos_pbl_ub=np.ma.masked_where(ravel_pbl_total<pbl_ub, ravel_sst_pos)

sst_pos_pbl_lb =sst_pos_pbl_lb[~sst_pos_pbl_lb.mask]
sst_pos_pbl_mb= sst_pos_pbl_mb[~sst_pos_pbl_mb.mask]
sst_pos_pbl_mb=np.ravel(sst_pos_pbl_mb)
sst_pos_pbl_ub =sst_pos_pbl_ub[~sst_pos_pbl_ub.mask]

#Wind Positive 
ur_pos_pbl_lb=np.ma.masked_where(ravel_pbl_total>pbl_lb,ravel_ur_pos) #negative SST values with PBL thresholds 
ur_pos_pbl_mb=np.ma.masked_where(np.logical_and(ravel_pbl_total<pbl_lb ,ravel_pbl_total>pbl_ub), ravel_ur_pos)
ur_pos_pbl_ub=np.ma.masked_where(ravel_pbl_total<pbl_ub,ravel_ur_pos) #negative SST values with PBL thresholds 

ur_pos_pbl_lb =ur_pos_pbl_lb[~ur_pos_pbl_lb.mask]
ur_pos_pbl_mb= ur_pos_pbl_mb[~ur_pos_pbl_mb.mask]
ur_pos_pbl_mb=np.ravel(ur_pos_pbl_mb)
ur_pos_pbl_ub =ur_pos_pbl_ub[~ur_pos_pbl_ub.mask]



##PBl Negative 
pbl_neg_pbl_lb=np.ma.masked_where(ravel_pbl_total>pbl_lb, ravel_pbl_neg) #negative SST values with PBL thresholds 
pbl_neg_pbl_mb=np.ma.masked_where(np.logical_and(ravel_pbl_total<pbl_lb ,ravel_pbl_total>pbl_ub), ravel_pbl_neg)
pbl_neg_pbl_ub=np.ma.masked_where(ravel_pbl_total<pbl_ub, ravel_pbl_neg) #negative SST values with PBL thresholds 

pbl_neg_pbl_lb =pbl_neg_pbl_lb[~pbl_neg_pbl_lb.mask]
pbl_neg_pbl_mb =pbl_neg_pbl_mb[~pbl_neg_pbl_mb.mask]
pbl_neg_pbl_mb=np.ravel(pbl_neg_pbl_mb)
pbl_neg_pbl_ub =pbl_neg_pbl_ub[~pbl_neg_pbl_ub.mask]


##SST Negative
sst_neg_pbl_lb=np.ma.masked_where(ravel_pbl_total>pbl_lb, ravel_sst_neg) #positive SST values with PBL thersholds 
sst_neg_pbl_mb=np.ma.masked_where(np.logical_and(ravel_pbl_total<pbl_lb ,ravel_pbl_total>pbl_ub), ravel_sst_neg)
sst_neg_pbl_ub=np.ma.masked_where(ravel_pbl_total<pbl_ub, ravel_sst_neg)

sst_neg_pbl_lb =sst_neg_pbl_lb[~sst_neg_pbl_lb.mask]
sst_neg_pbl_mb= sst_neg_pbl_mb[~sst_neg_pbl_mb.mask]
sst_neg_pbl_mb=np.ravel(sst_neg_pbl_mb)
sst_neg_pbl_ub =sst_neg_pbl_ub[~sst_neg_pbl_ub.mask]

##Ur Negative 
ur_neg_pbl_lb=np.ma.masked_where(ravel_pbl_total>pbl_lb,ravel_ur_neg)#negative SST values with PBL thresholds 
ur_neg_pbl_mb=np.ma.masked_where(np.logical_and(ravel_pbl_total<pbl_lb ,ravel_pbl_total>pbl_ub), ravel_ur_neg)
ur_neg_pbl_ub=np.ma.masked_where(ravel_pbl_total<pbl_ub,ravel_ur_neg)#negative SST values with PBL thresholds 


ur_neg_pbl_lb =ur_neg_pbl_lb[~ur_neg_pbl_lb.mask]
ur_neg_pbl_mb= ur_neg_pbl_mb[~ur_neg_pbl_mb.mask]
ur_neg_pbl_mb=np.ravel(ur_neg_pbl_mb)
ur_neg_pbl_ub =ur_neg_pbl_ub[~ur_neg_pbl_ub.mask]



# In[17]:


# Wind Masks 

#PBL
pbl_pos_wind_lb=np.ma.masked_where(ravel_wind_total>wind_lb,ravel_pbl_pos) #positive SST values with wind thresholds 
pbl_pos_wind_mb=np.ma.masked_where(np.logical_and(ravel_wind_total<wind_lb ,ravel_wind_total>wind_ub), ravel_pbl_pos)
pbl_pos_wind_ub=np.ma.masked_where(ravel_wind_total<wind_ub,ravel_pbl_pos)#positive SST values with wind thresholds 

pbl_neg_wind_lb=np.ma.masked_where(ravel_wind_total>wind_lb,ravel_pbl_neg) #negative SST values with wind thresholds 
pbl_neg_wind_mb=np.ma.masked_where(np.logical_and(ravel_wind_total<wind_lb ,ravel_wind_total>wind_ub), ravel_pbl_neg)
pbl_neg_wind_ub=np.ma.masked_where(ravel_wind_total<wind_ub,ravel_pbl_neg)#negative SST values with wind thresholds 

pbl_pos_wind_lb =pbl_pos_wind_lb[~pbl_pos_wind_lb.mask]
pbl_pos_wind_mb =pbl_pos_wind_mb[~pbl_pos_wind_mb.mask]
pbl_pos_wind_mb=np.ravel(pbl_pos_wind_mb)
pbl_pos_wind_ub =pbl_pos_wind_ub[~pbl_pos_wind_ub.mask]

pbl_neg_wind_lb =pbl_neg_wind_lb[~pbl_neg_wind_lb.mask]
pbl_neg_wind_mb =pbl_neg_wind_mb[~pbl_neg_wind_mb.mask]
pbl_neg_wind_mb=np.ravel(pbl_neg_wind_mb)
pbl_neg_wind_ub =pbl_neg_wind_ub[~pbl_neg_wind_ub.mask]

##SST
sst_pos_wind_lb=np.ma.masked_where(ravel_wind_total>wind_lb, ravel_sst_pos) #positive SST values with PBL thersholds 
sst_pos_wind_mb=np.ma.masked_where(np.logical_and(ravel_wind_total<wind_lb ,ravel_wind_total>wind_ub), ravel_sst_pos)
sst_pos_wind_ub=np.ma.masked_where(ravel_wind_total<wind_ub, ravel_sst_pos)

sst_neg_wind_lb=np.ma.masked_where(ravel_wind_total>wind_lb, ravel_sst_neg) #positive SST values with PBL thersholds 
sst_neg_wind_mb=np.ma.masked_where(np.logical_and(ravel_wind_total<wind_lb ,ravel_wind_total>wind_ub), ravel_sst_neg)
sst_neg_wind_ub=np.ma.masked_where(ravel_wind_total<wind_ub, ravel_sst_neg)

sst_pos_wind_lb =sst_pos_wind_lb[~sst_pos_wind_lb.mask]
sst_pos_wind_mb =sst_pos_wind_mb[~sst_pos_wind_mb.mask]
sst_pos_wind_mb=np.ravel(sst_pos_wind_mb)
sst_pos_wind_ub =sst_pos_wind_ub[~sst_pos_wind_ub.mask]

sst_neg_wind_lb =sst_neg_wind_lb[~sst_neg_wind_lb.mask]
sst_neg_wind_mb =sst_neg_wind_mb[~sst_neg_wind_mb.mask]
sst_neg_wind_mb=np.ravel(sst_neg_wind_mb)
sst_neg_wind_ub =sst_neg_wind_ub[~sst_neg_wind_ub.mask]


##WIND
ur_pos_wind_lb=np.ma.masked_where(ravel_wind_total>wind_lb,ravel_ur_pos) #positive SST values with wind thresholds 
ur_pos_wind_mb=np.ma.masked_where(np.logical_and(ravel_wind_total<wind_lb ,ravel_wind_total>wind_ub), ravel_ur_pos)
ur_pos_wind_ub=np.ma.masked_where(ravel_wind_total<wind_ub,ravel_ur_pos) #positive SST values with wind thresholds 

ur_neg_wind_lb=np.ma.masked_where(ravel_wind_total>wind_lb,ravel_ur_neg) #negative SST values with wind thresholds 
ur_neg_wind_mb=np.ma.masked_where(np.logical_and(ravel_wind_total<wind_lb ,ravel_wind_total>wind_ub), ravel_ur_neg)
ur_neg_wind_ub=np.ma.masked_where(ravel_wind_total<wind_ub,ravel_ur_neg) #negative SST values with wind thresholds 

ur_pos_wind_lb =ur_pos_wind_lb[~ur_pos_wind_lb.mask]
ur_pos_wind_mb =ur_pos_wind_mb[~ur_pos_wind_mb.mask]
ur_pos_wind_mb=np.ravel(ur_pos_wind_mb)
ur_pos_wind_ub =ur_pos_wind_ub[~ur_pos_wind_ub.mask]

ur_neg_wind_lb =ur_neg_wind_lb[~ur_neg_wind_lb.mask]
ur_neg_wind_mb =ur_neg_wind_mb[~ur_neg_wind_mb.mask]
ur_neg_wind_mb=np.ravel(ur_neg_wind_mb)
ur_neg_wind_ub =ur_neg_wind_ub[~ur_neg_wind_ub.mask]







# In[18]:


##Making more dataframes for the applied masks 
#the middle bounds are not the same size array so they must be put in a separate dataframe


pbl_tresh_mb_df=pd.DataFrame({'pbl_pos_mb':pbl_pos_pbl_mb, 'pbl_neg_mb': pbl_neg_pbl_mb,  #PBL masks middle bound 
                              'ur_pos_mb': ur_pos_pbl_mb, 'ur_neg_mb': ur_neg_pbl_mb,
                             'sst_pos_mb': sst_pos_pbl_mb,'sst_neg_mb': sst_neg_pbl_mb})


wind_tresh_mb_df=pd.DataFrame({'pbl_pos_mb':pbl_pos_wind_mb, 'pbl_neg_mb': pbl_neg_wind_mb, #wind masks middle bounds
                              'ur_pos_mb': ur_pos_wind_mb, 'ur_neg_mb': ur_neg_wind_mb,
                             'sst_pos_mb': sst_pos_wind_mb,'sst_neg_mb': sst_neg_wind_mb})

pbl_tresh_df=pd.DataFrame({'pbl_pos_lb':pbl_pos_pbl_lb,'pbl_neg_lb':pbl_neg_pbl_lb, #pbl masks in the upper and lower bounds 
                           'pbl_pos_ub':pbl_pos_pbl_ub,'pbl_neg_ub':pbl_neg_pbl_ub,'ur_pos_lb':ur_pos_pbl_lb,'ur_neg_lb':ur_neg_pbl_lb, 'ur_pos_ub':ur_pos_pbl_ub,'ur_neg_ub':ur_neg_pbl_ub,
                          'sst_pos_lb':sst_pos_pbl_lb,'sst_neg_lb':sst_neg_pbl_lb,'sst_pos_ub':sst_pos_pbl_ub,'sst_neg_ub':sst_neg_pbl_ub})


wind_tresh_df=pd.DataFrame({'pbl_pos_lb':pbl_pos_wind_lb,'pbl_neg_lb':pbl_neg_wind_lb,'pbl_pos_ub':pbl_pos_wind_ub,'pbl_neg_ub':pbl_neg_wind_ub, #wind masks in the upper and lower bounds 
                           'ur_pos_lb':ur_pos_wind_lb,'ur_neg_lb':ur_neg_wind_lb,'ur_pos_ub':ur_pos_wind_ub,'ur_neg_ub':ur_neg_wind_ub,
                          'sst_pos_lb':sst_pos_wind_lb,'sst_neg_lb':sst_neg_wind_lb,'sst_pos_ub':sst_pos_wind_ub,'sst_neg_ub':sst_neg_wind_ub})




# # Calculating the slopes 

# In[19]:


##SST and PBL with wind masks 
# variables are labeled by what is applied_ maskedapplied_variablestested_sign
# so the first one is the polyfit on wind masks for pbl and SST over positive SST values 

#positive values 
polyfit_wind_pblsst_pos_ub=np.polyfit(pbl_pos_wind_ub[~np.isnan(pbl_pos_wind_ub) & ~np.isnan(sst_pos_wind_ub)],sst_pos_wind_ub[~np.isnan(sst_pos_wind_ub) & ~np.isnan(pbl_pos_wind_ub)], deg=1)
polyfit_wind_pblsst_pos_mb=np.polyfit(pbl_pos_wind_mb[~np.isnan(pbl_pos_wind_mb) & ~np.isnan(sst_pos_wind_mb)],sst_pos_wind_mb[~np.isnan(sst_pos_wind_mb) & ~np.isnan(pbl_pos_wind_mb)], deg=1)
polyfit_wind_pblsst_pos_lb=np.polyfit(pbl_pos_wind_lb[~np.isnan(pbl_pos_wind_lb) & ~np.isnan(sst_pos_wind_lb)],sst_pos_wind_lb[~np.isnan(sst_pos_wind_lb) & ~np.isnan(pbl_pos_wind_lb)], deg=1)
polyfit_wind_pblsst_pos=np.polyfit(pbl_pos_df[~np.isnan(sst_pos_df) & ~np.isnan(pbl_pos_df)],sst_pos_df[~np.isnan(sst_pos_df) & ~np.isnan(pbl_pos_df)], deg=1)

#Negative Values 
polyfit_wind_pblsst_neg_ub=np.polyfit(pbl_neg_wind_ub[~np.isnan(pbl_neg_wind_ub) & ~np.isnan(sst_neg_wind_ub)],sst_neg_wind_ub[~np.isnan(sst_neg_wind_ub) & ~np.isnan(pbl_neg_wind_ub)], deg=1)
polyfit_wind_pblsst_neg_mb=np.polyfit(pbl_neg_wind_mb[~np.isnan(pbl_neg_wind_mb) & ~np.isnan(sst_neg_wind_mb)],sst_neg_wind_mb[~np.isnan(sst_neg_wind_mb) & ~np.isnan(pbl_neg_wind_mb)], deg=1)
polyfit_wind_pblsst_neg_lb=np.polyfit(pbl_neg_wind_lb[~np.isnan(pbl_neg_wind_lb) & ~np.isnan(sst_neg_wind_lb)],sst_neg_wind_lb[~np.isnan(sst_neg_wind_lb) & ~np.isnan(pbl_neg_wind_lb)], deg=1)
polyfit_wind_pblsst_neg=np.polyfit(pbl_neg_df[~np.isnan(sst_neg_df) & ~np.isnan(pbl_neg_df)],sst_neg_df[~np.isnan(sst_neg_df) & ~np.isnan(pbl_neg_df)], deg=1)


# In[20]:


# SST and Ur with wind masks 

#positive Values 
polyfit_wind_sstur_pos_ub=np.polyfit(sst_pos_wind_ub[~np.isnan(sst_pos_wind_ub) & ~np.isnan(ur_pos_wind_ub)],ur_pos_wind_ub[~np.isnan(ur_pos_wind_ub) & ~np.isnan(sst_pos_wind_ub)], deg=1)
polyfit_wind_sstur_pos_mb=np.polyfit(sst_pos_wind_mb[~np.isnan(sst_pos_wind_mb) & ~np.isnan(ur_pos_wind_mb)],ur_pos_wind_mb[~np.isnan(ur_pos_wind_mb) & ~np.isnan(sst_pos_wind_mb)], deg=1)
polyfit_wind_sstur_pos_lb=np.polyfit(sst_pos_wind_lb[~np.isnan(sst_pos_wind_lb) & ~np.isnan(ur_pos_wind_lb)],ur_pos_wind_lb[~np.isnan(ur_pos_wind_lb) & ~np.isnan(sst_pos_wind_lb)], deg=1)
polyfit_wind_sstur_pos=np.polyfit(sst_pos_df[~np.isnan(ur_pos_df) & ~np.isnan(sst_pos_df)],ur_pos_df[~np.isnan(ur_pos_df) & ~np.isnan(sst_pos_df)], deg=1)

#Negative Values 
polyfit_wind_sstur_neg_ub=np.polyfit(sst_neg_wind_ub[~np.isnan(sst_neg_wind_ub) & ~np.isnan(ur_neg_wind_ub)],ur_neg_wind_ub[~np.isnan(ur_neg_wind_ub) & ~np.isnan(sst_neg_wind_ub)], deg=1)
polyfit_wind_sstur_neg_mb=np.polyfit(sst_neg_wind_mb[~np.isnan(sst_neg_wind_mb) & ~np.isnan(ur_neg_wind_mb)],ur_neg_wind_mb[~np.isnan(ur_neg_wind_mb) & ~np.isnan(sst_neg_wind_mb)], deg=1)
polyfit_wind_sstur_neg_lb=np.polyfit(sst_neg_wind_lb[~np.isnan(sst_neg_wind_lb) & ~np.isnan(ur_neg_wind_lb)],ur_neg_wind_lb[~np.isnan(ur_neg_wind_lb) & ~np.isnan(sst_neg_wind_lb)], deg=1)
polyfit_wind_sstur_neg=np.polyfit(sst_neg_df[~np.isnan(ur_neg_df) & ~np.isnan(sst_neg_df)],ur_neg_df[~np.isnan(ur_neg_df) & ~np.isnan(sst_neg_df)], deg=1)


# In[21]:


# PBL Gradients 

#Positive Values for PBL and Wind 
# variables are labeled by what is applied_ variablestested_maskedapplied_sign
# for example, the first one is wind and PBL values with a PBL mask over positive SST values 

polyfit_windpbl_pbl_pos_ub=np.polyfit(pbl_pos_pbl_ub[~np.isnan(pbl_pos_pbl_ub) & ~np.isnan(ur_pos_pbl_ub)],ur_pos_pbl_ub[~np.isnan(ur_pos_pbl_ub) & ~np.isnan(pbl_pos_pbl_ub)], deg=1)
polyfit_windpbl_pbl_pos_mb=np.polyfit(pbl_pos_pbl_mb[~np.isnan(pbl_pos_pbl_mb) & ~np.isnan(ur_pos_pbl_mb)],ur_pos_pbl_mb[~np.isnan(ur_pos_pbl_mb) & ~np.isnan(pbl_pos_pbl_mb)], deg=1)
polyfit_windpbl_pbl_pos_lb=np.polyfit(pbl_pos_pbl_lb[~np.isnan(pbl_pos_pbl_lb) & ~np.isnan(ur_pos_pbl_lb)],ur_pos_pbl_lb[~np.isnan(ur_pos_pbl_lb) & ~np.isnan(pbl_pos_pbl_lb)], deg=1)
polyfit_windpbl_pbl_pos=np.polyfit(pbl_pos_df[~np.isnan(ur_pos_df) & ~np.isnan(pbl_pos_df)],ur_pos_df[~np.isnan(ur_pos_df) & ~np.isnan(pbl_pos_df)], deg=1)

#Negative Values for PBL and Wind 
polyfit_windpbl_pbl_neg_ub=np.polyfit(pbl_neg_pbl_ub[~np.isnan(pbl_neg_pbl_ub) & ~np.isnan(ur_neg_pbl_ub)],ur_neg_pbl_ub[~np.isnan(ur_neg_pbl_ub) & ~np.isnan(pbl_neg_pbl_ub)], deg=1)
polyfit_windpbl_pbl_neg_mb=np.polyfit(pbl_neg_pbl_mb[~np.isnan(pbl_neg_pbl_mb) & ~np.isnan(ur_neg_pbl_mb)],ur_neg_pbl_mb[~np.isnan(ur_neg_pbl_mb) & ~np.isnan(pbl_neg_pbl_mb)], deg=1)
polyfit_windpbl_pbl_neg_lb=np.polyfit(pbl_neg_pbl_lb[~np.isnan(pbl_neg_pbl_lb) & ~np.isnan(ur_neg_pbl_lb)],ur_neg_pbl_lb[~np.isnan(ur_neg_pbl_lb) & ~np.isnan(pbl_neg_pbl_lb)], deg=1)
polyfit_windpbl_pbl_neg=np.polyfit(pbl_neg_df[~np.isnan(ur_neg_df) & ~np.isnan(pbl_neg_df)],ur_neg_df[~np.isnan(ur_neg_df) & ~np.isnan(pbl_neg_df)], deg=1)

#Positive Values for PBL and SST 
polyfit_pblsst_pbl_pos_ub=np.polyfit(pbl_pos_pbl_ub[~np.isnan(pbl_pos_pbl_ub) & ~np.isnan(sst_pos_pbl_ub)],sst_pos_pbl_ub[~np.isnan(sst_pos_pbl_ub) & ~np.isnan(pbl_pos_pbl_ub)], deg=1)
polyfit_pblsst_pbl_pos_mb=np.polyfit(pbl_pos_pbl_mb[~np.isnan(pbl_pos_pbl_mb) & ~np.isnan(sst_pos_pbl_mb)],sst_pos_pbl_mb[~np.isnan(sst_pos_pbl_mb) & ~np.isnan(pbl_pos_pbl_mb)], deg=1)
polyfit_pblsst_pbl_pos_lb=np.polyfit(pbl_pos_pbl_lb[~np.isnan(pbl_pos_pbl_lb) & ~np.isnan(sst_pos_pbl_lb)],sst_pos_pbl_lb[~np.isnan(sst_pos_pbl_lb) & ~np.isnan(pbl_pos_pbl_lb)], deg=1)
polyfit_pblsst_pbl_pos=np.polyfit(pbl_pos_df[~np.isnan(pbl_pos_df) & ~np.isnan(sst_pos_df)],sst_pos_df[~np.isnan(pbl_pos_df) & ~np.isnan(sst_pos_df)], deg=1)


#Negative Values for PBL and SST
polyfit_pblsst_pbl_neg_ub=np.polyfit(pbl_neg_pbl_ub[~np.isnan(pbl_neg_pbl_ub) & ~np.isnan(sst_neg_pbl_ub)],sst_neg_pbl_ub[~np.isnan(sst_neg_pbl_ub) & ~np.isnan(pbl_neg_pbl_ub)], deg=1)
polyfit_pblsst_pbl_neg_mb=np.polyfit(pbl_neg_pbl_mb[~np.isnan(pbl_neg_pbl_mb) & ~np.isnan(sst_neg_pbl_mb)],sst_neg_pbl_mb[~np.isnan(sst_neg_pbl_mb) & ~np.isnan(pbl_neg_pbl_mb)], deg=1)
polyfit_pblsst_pbl_neg_lb=np.polyfit(pbl_neg_pbl_lb[~np.isnan(pbl_neg_pbl_lb) & ~np.isnan(sst_neg_pbl_lb)],sst_neg_pbl_lb[~np.isnan(sst_neg_pbl_lb) & ~np.isnan(pbl_neg_pbl_lb)], deg=1)
polyfit_pblsst_pbl_neg=np.polyfit(pbl_neg_df[~np.isnan(sst_neg_df) & ~np.isnan(pbl_neg_df)],sst_neg_df[~np.isnan(sst_neg_df) & ~np.isnan(pbl_neg_df)], deg=1)


#Positive Values for UR and SST 
polyfit_ursst_pbl_pos_ub=np.polyfit(ur_pos_pbl_ub[~np.isnan(ur_pos_pbl_ub) & ~np.isnan(sst_pos_pbl_ub)],sst_pos_pbl_ub[~np.isnan(sst_pos_pbl_ub) & ~np.isnan(ur_pos_pbl_ub)], deg=1)
polyfit_ursst_pbl_pos_mb=np.polyfit(ur_pos_pbl_mb[~np.isnan(ur_pos_pbl_mb) & ~np.isnan(sst_pos_pbl_mb)],sst_pos_pbl_mb[~np.isnan(sst_pos_pbl_mb) & ~np.isnan(ur_pos_pbl_mb)], deg=1)
polyfit_ursst_pbl_pos_lb=np.polyfit(ur_pos_pbl_lb[~np.isnan(ur_pos_pbl_lb) & ~np.isnan(sst_pos_pbl_lb)],sst_pos_pbl_lb[~np.isnan(sst_pos_pbl_lb) & ~np.isnan(ur_pos_pbl_lb)], deg=1)
polyfit_ursst_pbl_pos=np.polyfit(ur_pos_df[~np.isnan(sst_pos_df) & ~np.isnan(ur_pos_df)],sst_pos_df[~np.isnan(sst_pos_df) & ~np.isnan(ur_pos_df)], deg=1)

#Negative Values for UR and SST 
polyfit_ursst_pbl_neg_ub=np.polyfit(ur_neg_pbl_ub[~np.isnan(ur_neg_pbl_ub) & ~np.isnan(sst_neg_pbl_ub)],sst_neg_pbl_ub[~np.isnan(sst_neg_pbl_ub) & ~np.isnan(ur_neg_pbl_ub)], deg=1)
polyfit_ursst_pbl_neg_mb=np.polyfit(ur_neg_pbl_mb[~np.isnan(ur_neg_pbl_mb) & ~np.isnan(sst_neg_pbl_mb)],sst_neg_pbl_mb[~np.isnan(sst_neg_pbl_mb) & ~np.isnan(ur_neg_pbl_mb)], deg=1)
polyfit_ursst_pbl_neg_lb=np.polyfit(ur_neg_pbl_lb[~np.isnan(ur_neg_pbl_lb) & ~np.isnan(sst_neg_pbl_lb)],sst_neg_pbl_lb[~np.isnan(sst_neg_pbl_lb) & ~np.isnan(ur_neg_pbl_lb)], deg=1)
polyfit_ursst_pbl_neg=np.polyfit(ur_neg_df[~np.isnan(sst_neg_df) & ~np.isnan(ur_neg_df)],sst_neg_df[~np.isnan(sst_neg_df) & ~np.isnan(ur_neg_df)], deg=1)


# In[22]:


#Wind and PBL wind Gradients 

#positive Values
polyfit_wind_pblur_pos_ub=np.polyfit(pbl_pos_wind_ub[~np.isnan(pbl_pos_wind_ub) & ~np.isnan(ur_pos_wind_ub)],ur_pos_wind_ub[~np.isnan(ur_pos_wind_ub) & ~np.isnan(pbl_pos_wind_ub)], deg=1)
polyfit_wind_pblur_pos_mb=np.polyfit(pbl_pos_wind_mb[~np.isnan(pbl_pos_wind_mb) & ~np.isnan(ur_pos_wind_mb)],ur_pos_wind_mb[~np.isnan(ur_pos_wind_mb) & ~np.isnan(pbl_pos_wind_mb)], deg=1)
polyfit_wind_pblur_pos_lb=np.polyfit(pbl_pos_wind_lb[~np.isnan(pbl_pos_wind_lb) & ~np.isnan(ur_pos_wind_lb)],ur_pos_wind_lb[~np.isnan(ur_pos_wind_lb) & ~np.isnan(pbl_pos_wind_lb)], deg=1)
polyfit_wind_pblur_pos=np.polyfit(pbl_pos_df[~np.isnan(ur_pos_df) & ~np.isnan(pbl_pos_df)],ur_pos_df[~np.isnan(ur_pos_df) & ~np.isnan(pbl_pos_df)], deg=1)

#Negative Values 
polyfit_wind_pblur_neg_ub=np.polyfit(pbl_neg_wind_ub[~np.isnan(pbl_neg_wind_ub) & ~np.isnan(ur_neg_wind_ub)],ur_neg_wind_ub[~np.isnan(ur_neg_wind_ub) & ~np.isnan(pbl_neg_wind_ub)], deg=1)
polyfit_wind_pblur_neg_mb=np.polyfit(pbl_neg_wind_mb[~np.isnan(pbl_neg_wind_mb) & ~np.isnan(ur_neg_wind_mb)],ur_neg_wind_mb[~np.isnan(ur_neg_wind_mb) & ~np.isnan(pbl_neg_wind_mb)], deg=1)
polyfit_wind_pblur_neg_lb=np.polyfit(pbl_neg_wind_lb[~np.isnan(pbl_neg_wind_lb) & ~np.isnan(ur_neg_wind_lb)],ur_neg_wind_lb[~np.isnan(ur_neg_wind_lb) & ~np.isnan(pbl_neg_wind_lb)], deg=1)
polyfit_wind_pblur_neg=np.polyfit(pbl_neg_df[~np.isnan(ur_neg_df) & ~np.isnan(pbl_neg_df)],ur_neg_df[~np.isnan(ur_neg_df) & ~np.isnan(pbl_neg_df)], deg=1)




# In[ ]:





# In[ ]:





# In[ ]:





# # Graphing for the positive and negative values

# In[33]:


#SST and PBL 
x_range_pbl_wind_pos=np.arange(0,6e-5,0.5e-6)


pbl_wind_pos_graph=plt.figure(figsize=(10,10))
sns.regplot(data=wind_tresh_df,x="sst_pos_lb",y="pbl_pos_lb",x_bins=x_range_pbl_wind_pos,color='purple', label=r'Ur$<${0:1.3f} m/s m={1:.4f}'.format(wind_lb,polyfit_wind_pblsst_pos_lb[0]))
sns.regplot(data=wind_tresh_mb_df,x="sst_pos_mb",y="pbl_pos_mb",x_bins=x_range_pbl_wind_pos,color='black',label=r'{0:1.3f}$<$Ur$<${1:1.3f} m/s m={2:.4f}'.format(wind_lb, wind_ub, polyfit_wind_pblsst_pos_mb[0]))
sns.regplot(data=wind_tresh_df,x="sst_pos_ub",y="pbl_pos_ub",x_bins=x_range_pbl_wind_pos,color='orange',label=r'{0:1.3f}$<$Ur m/s m={1:.4f}'.format(wind_ub,polyfit_wind_pblsst_pos_ub[0]))
sns.regplot(data=data_sst_pos,x='sst_pos',y='pbl_pos',x_bins=x_range_pbl_wind_pos,color='green',label='Ur total m={0:.4f} m/s'.format(polyfit_wind_pblsst_pos[0]))
plt.scatter(data_sst_pos['sst_pos'],data_sst_pos['pbl_pos'],alpha=0.25)
plt.xlim(0,3e-5)
# plt.ylim(-0.00005,0.0003)
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime +} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}PBL^{\prime +}$')
plt.title('Positive SST Values with wind gradients: \n SST and PBL')

plt.legend(loc=0,fontsize='x-small')
plt.tight_layout()


x_range_pbl_wind_neg=np.arange(-7.5e-5,0,.5e-6)
    
pbl_wind_neg_graph=plt.figure(figsize=(10,10))
sns.regplot(data=wind_tresh_df,x="sst_neg_lb",y="pbl_neg_lb",x_bins=x_range_pbl_wind_neg,color='purple',label=r'Ur$<${0:1.3f} m/s m={1:.4f}'.format(wind_lb, polyfit_wind_pblsst_neg_lb[0]))
sns.regplot(data=wind_tresh_mb_df,x="sst_neg_mb",y="pbl_neg_mb",x_bins=x_range_pbl_wind_neg,color='black',label=r'{0:1.3f}$<$Ur$<${1:1.3f} m/s m={2:.4f}'.format(wind_lb,wind_ub,polyfit_wind_pblsst_neg_mb[0]))
sns.regplot(data=wind_tresh_df,x="sst_neg_ub",y="pbl_neg_ub",x_bins=x_range_pbl_wind_neg,color='orange',label=r'{0:1.3f}$<$ Ur m/s m={1:.4f}'.format(wind_ub,polyfit_wind_pblsst_neg_ub[0]))
sns.regplot(data=data_sst_neg,x='sst_neg',y='pbl_neg',x_bins=x_range_pbl_wind_neg,color='green',label='Ur total m={0:0.4f}'.format(polyfit_wind_pblsst_neg[0]))
plt.scatter(data_sst_neg['sst_neg'],data_sst_neg['pbl_neg'],alpha=0.25)
plt.xlim(-5e-5,0.01e-4)
plt.ylim(-0.003,0.002)
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime -} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}PBL^{\prime -}$')
plt.title('Negative SST Values with wind gradients: \n SST and PBL ')
# plt.title('Negative SST Values: \ nSST and PBL \n lower bound m={0:2.f} \n upper bound m={1:0.2f}'.format(np.polyfit(np.isfinite(pbl_neg_pbl_lb
# plt.title('Negative SST Values with PBL Thresholds: \n lower bound={0:.2f} \n upper bound m={1:0.2f}'.format((np.polyfit(np.isfinite(pbl_neg_pbl_lb),np.isfinite(sst_neg_pbl_lb),deg=1[0])),(np.polyfit(np.isfinite(pbl_neg_pbl_ub),np.isfinite(sst_neg_pbl_ub),deg=1[0])))

plt.legend(loc=0,fontsize='x-small')
plt.tight_layout()


# In[34]:


##Ur and PBL 
x_range_ur_wind_pos=np.arange(0,6e-5,1e-7)


pbl_wind_pos_graph=plt.figure(figsize=(10,10)) 

sns.regplot(data=wind_tresh_df,x="ur_pos_lb",y="pbl_pos_lb",x_bins=x_range_ur_wind_pos,color='purple',label=r'Ur$<${0:1.3f} m/s m={1:.4f}'.format(wind_lb,polyfit_wind_pblur_pos_lb[0]))
sns.regplot(data=wind_tresh_mb_df,x="ur_pos_mb",y="pbl_pos_mb",x_bins=x_range_ur_wind_pos,color='black',label=r'{0:.4f}$<$Ur$<${1:1.3f} m/s m={2:.4f}'.format(wind_lb,wind_ub, polyfit_wind_pblur_pos_mb[0]))
sns.regplot(data=wind_tresh_df,x="ur_pos_ub",y="pbl_pos_ub",x_bins=x_range_ur_wind_pos,color='orange',label=r'{0:1.3f}$<$Ur m/s m={1:.4f}'.format(wind_ub, polyfit_wind_pblur_pos_ub[0]))
sns.regplot(data=data_sst_pos,x='ur_pos',y='pbl_pos',x_bins=x_range_ur_wind_pos,color='green',label='Ur total Values m={0:.4f}'.format(polyfit_wind_pblur_pos[0]))
plt.scatter(data_sst_pos['ur_pos'],data_sst_pos['pbl_pos'],alpha=0.25)
plt.xlim(0,2e-5)
# plt.ylim(-0.00005,0.0003)
plt.xlabel(r'$ \frac{\partial}{\partial r}Ur ^ {\prime +} $(°C/m)') 
plt.ylabel(r' $\frac{\partial}{\partial r}PBL^{\prime +}$')
plt.title('Positive SST Values with wind gradients: \n Ur and PBL')

plt.legend(loc=0,fontsize='x-small')
plt.tight_layout()


x_range_ur_wind_neg=np.arange(-7.5e-5,0,1e-7)
    
pbl_wind_neg_graph=plt.figure(figsize=(10,10))
# sns.regplot(data=data_sst_neg, x="pbl_neg", y="ur_neg",x_bins=x_range_ur_pbl_neg,color='orange',label='Negative SST Values')
sns.regplot(data=wind_tresh_df,x="ur_neg_lb",y="pbl_neg_lb",x_bins=x_range_ur_wind_neg,color='purple',label=r'Ur$<${0:1.3f} m/s m={1:.4f}'.format(wind_lb, polyfit_wind_pblur_neg_lb[0]))
sns.regplot(data=wind_tresh_mb_df,x="ur_neg_mb",y="pbl_neg_mb",x_bins=x_range_ur_wind_neg,color='black',label=r'{0:.4f}$<$Ur$<${1:1.3f} m={2:.4f}'.format(wind_lb,wind_ub, polyfit_wind_pblur_neg_mb[0]))
sns.regplot(data=wind_tresh_df,x="ur_neg_ub",y="pbl_neg_ub",x_bins=x_range_ur_wind_neg,color='orange',label=r'{0:.4f}$<$Ur m/s m={1:.4f}'.format(wind_ub,polyfit_wind_pblur_neg_ub[0]))
sns.regplot(data=data_sst_neg,x='ur_neg',y='pbl_neg',x_bins=x_range_ur_wind_neg,color='green',label='Ur total Values m={0:0.4f}'.format(polyfit_wind_pblur_neg[0]))
plt.scatter(data_sst_neg['ur_neg'],data_sst_neg['pbl_neg'],alpha=0.25)
plt.xlim(-1.5e-5,0.01e-4)
plt.ylim(-0.003,0.002)
plt.xlabel(r'$ \frac{\partial}{\partial r}Ur ^ {\prime -} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}PBL^{\prime -}$')
plt.title('Negative SST Values with wind gradients: \n Ur and PBL ')
# plt.title('Negative SST Values: \ nUr and PBL \n lower bound m={0:2.f} \n upper bound m={1:0.2f}'.format(np.polyfit(np.isfinite(pbl_neg_pbl_lb
# plt.title('Negative SST Values with PBL Thresholds: \n lower bound={0:.2f} \n upper bound m={1:0.2f}'.format((np.polyfit(np.isfinite(pbl_neg_pbl_lb),np.isfinite(sst_neg_pbl_lb),deg=1[0])),(np.polyfit(np.isfinite(pbl_neg_pbl_ub),np.isfinite(sst_neg_pbl_ub),deg=1[0])))

plt.legend(loc=0,fontsize='x-small')
plt.tight_layout()




# In[25]:


##Ur and SST
x_range_ur_wind_pos=np.arange(0,6e-5,0.5e-6)

# m={0:.4f}'.format(polyfit_windneg_lb[0])

pbl_wind_pos_graph=plt.figure(figsize=(10,10)) 
sns.regplot(data=wind_tresh_df,x="sst_pos_lb",y="ur_pos_lb",x_bins=x_range_ur_wind_pos,color='purple',label=r'Ur$<${0:1.3f} m/s m={1:.4f}'.format(wind_lb,polyfit_wind_sstur_pos_lb[0]))
sns.regplot(data=wind_tresh_mb_df,x="sst_pos_mb",y="ur_pos_mb",x_bins=x_range_ur_wind_pos,color='black',label=r'{0:.4f}$<$Ur$<${1:1.3f} m/s m={2:.4f}'.format(wind_lb,wind_ub, polyfit_wind_sstur_pos_mb[0]))
sns.regplot(data=wind_tresh_df,x="sst_pos_ub",y="ur_pos_ub",x_bins=x_range_ur_wind_pos,color='orange',label=r'{0:.4f} m/s$<$Ur m={1:.4f}'.format(wind_ub, polyfit_wind_sstur_pos_ub[0]))
sns.regplot(data=data_sst_pos,x='sst_pos',y='ur_pos',x_bins=x_range_ur_wind_pos,color='green',label='Ur total m={0:.4f}'.format(polyfit_wind_sstur_pos[0]))
plt.scatter(data_sst_pos['sst_pos'],data_sst_pos['ur_pos'],alpha=0.25)
plt.xlim(0,7.5e-5)
# plt.ylim(-0.00005,0.0003)
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime +} $(°C/m)') 
plt.ylabel(r' $\frac{\partial}{\partial r}Ur^{\prime +}$')
plt.title('Positive SST Values with wind gradients: \n SST and Ur')

plt.legend(loc=0,fontsize='x-small')
plt.tight_layout()


x_range_ur_wind_neg=np.arange(-7.5e-5,0,0.5e-6)
    
pbl_wind_neg_graph=plt.figure(figsize=(10,10))
# sns.regplot(data=data_sst_neg, x="pbl_neg", y="ur_neg",x_bins=x_range_ur_pbl_neg,color='orange',label='Negative SST Values')
sns.regplot(data=wind_tresh_df,x="sst_neg_lb",y="ur_neg_lb",x_bins=x_range_ur_wind_neg,color='purple',label=r'Ur$<${0:1.3f} m={0:.4f}'.format(wind_lb,polyfit_wind_sstur_neg_lb[0]))
sns.regplot(data=wind_tresh_mb_df,x="sst_neg_mb",y="ur_neg_mb",x_bins=x_range_ur_wind_pos,color='black',label=r'{0:.4f}$<$Ur$<${1:1.3f} m/s m={2:.4f}'.format(wind_lb,wind_ub,polyfit_wind_sstur_neg_mb[0]))
sns.regplot(data=wind_tresh_df,x="sst_neg_ub",y="ur_neg_ub",x_bins=x_range_ur_wind_neg,color='orange',label=r'{0:.4f} m/s$<$Ur m={1:.4f}'.format(wind_ub, polyfit_wind_sstur_neg_ub[0]))
sns.regplot(data=data_sst_neg,x='sst_neg',y='ur_neg',x_bins=x_range_ur_wind_neg,color='green',label='Ur total m={0:0.4f}'.format(polyfit_wind_sstur_neg[0]))
plt.scatter(data_sst_neg['sst_neg'],data_sst_neg['ur_neg'],alpha=0.25)
plt.xlim(-7.5e-5,0.01e-4)
# plt.ylim(-0.003,0.002)
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime -} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}Ur^{\prime -}$')
plt.title('Negative SST Values with wind gradients: \n SST and Ur ')
# plt.title('Negative SST Values: \ nUr and PBL \n lower bound m={0:2.f} \n upper bound m={1:0.2f}'.format(np.polyfit(np.isfinite(pbl_neg_pbl_lb
# plt.title('Negative SST Values with PBL Thresholds: \n lower bound={0:.2f} \n upper bound m={1:0.2f}'.format((np.polyfit(np.isfinite(pbl_neg_pbl_lb),np.isfinite(sst_neg_pbl_lb),deg=1[0])),(np.polyfit(np.isfinite(pbl_neg_pbl_ub),np.isfinite(sst_neg_pbl_ub),deg=1[0])))

plt.legend(loc=0,fontsize='x-small')
plt.tight_layout()


# In[26]:


#SST and PBL 

x_range_ur_pbl_neg=np.arange(-7.5e-5,0,0.5e-6)

    
pbl_wind_neg_graph=plt.figure(figsize=(10,10))
plt.scatter(data_sst_neg['sst_neg'],data_sst_neg['pbl_neg'],alpha=0.25)
sns.regplot(data=pbl_tresh_df,x="sst_neg_lb",y="pbl_neg_lb",x_bins=x_range_ur_pbl_neg,color='purple',label=r'PBL$<${0:1.3f}m m={1:.4f}'.format(pbl_lb, polyfit_pblsst_pbl_neg_lb[0]))
sns.regplot(data=pbl_tresh_mb_df,x="sst_neg_mb",y="pbl_neg_mb",x_bins=x_range_ur_wind_neg,color='black',label=r'{0:.4f}$<$PBL$<${1:1.3f}m m={2:.4f}'.format(pbl_lb,pbl_ub,polyfit_pblsst_pbl_neg_mb[0]))
sns.regplot(data=pbl_tresh_df,x="sst_neg_ub",y="pbl_neg_ub",x_bins=x_range_ur_pbl_neg,color='orange',label=r'{0:.4f}m$<$PBL m={1:.4f}'.format(pbl_ub, polyfit_pblsst_pbl_neg_ub[0]))
sns.regplot(data=data_sst_neg,x='sst_neg',y='pbl_neg',x_bins=x_range_ur_pbl_neg,color='green',label='PBL total m={0:0.4f}'.format(polyfit_pblsst_pbl_neg[0]))
plt.xlim(-5e-5,0.01e-4)
# plt.ylim(-0.0005,0.0003)
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime -} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}PBL^{\prime -}$')
plt.title('b) Negative SST Values with PBL Thresholds: \n SST and PBL')
# plt.title('Negative SST Values: \ nSST and PBL \n lower bound m={0:2.f} \n upper bound m={1:0.2f}'.format(np.polyfit(np.isfinite(pbl_neg_pbl_lb
# plt.title('Negative SST Values with PBL Thresholds: \n lower bound={0:.2f} \n upper bound m={1:0.2f}'.format((np.polyfit(np.isfinite(pbl_neg_pbl_lb),np.isfinite(sst_neg_pbl_lb),deg=1[0])),(np.polyfit(np.isfinite(pbl_neg_pbl_ub),np.isfinite(sst_neg_pbl_ub),deg=1[0])))

plt.legend(loc=0,fontsize='x-small')
plt.tight_layout()

x_range_ur_pbl_pos=np.arange(0,6e-5,0.5e-6)

polyfit_pblpos_lb=np.polyfit(np.isfinite(pbl_pos_pbl_lb),np.isfinite(sst_pos_pbl_lb), deg=1)
polyfit_pblpos_ub=np.polyfit(np.isfinite(pbl_pos_pbl_ub),np.isfinite(sst_pos_pbl_ub), deg=1)
    
pbl_wind_pos_graph=plt.figure(figsize=(10,10))

plt.scatter(data_sst_pos['sst_pos'],data_sst_pos['pbl_pos'],alpha=0.25)
sns.regplot(data=pbl_tresh_df,x="sst_pos_lb",y="pbl_pos_lb",x_bins=x_range_ur_pbl_pos,color='purple',label=r'PBL$<${0:1.3f}m m={1:.4f}'.format(pbl_lb,polyfit_pblsst_pbl_pos_lb[0]))
sns.regplot(data=pbl_tresh_mb_df,x="sst_pos_mb",y="pbl_pos_mb",x_bins=x_range_ur_wind_pos,color='black',label=r'{0:.4f}$<$PBL$<${1:1.3f}m m={2:.4f}'.format(pbl_lb,pbl_ub,polyfit_pblsst_pbl_pos_mb[0]))
sns.regplot(data=pbl_tresh_df,x="sst_pos_ub",y="pbl_pos_ub",x_bins=x_range_ur_pbl_pos,color='orange',label=r'{0:.4f}m$<$PBL m={1:.4f}'.format(pbl_ub, polyfit_pblsst_pbl_pos_ub[0]))
sns.regplot(data=data_sst_pos,x='sst_pos',y='pbl_pos',x_bins=x_range_ur_pbl_pos,color='green',label='PBL total m={0:0.4f}'.format(polyfit_pblsst_pbl_pos[0]))
plt.xlim(0,5e-5)
# plt.ylim(-0.00005,0.0003)
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime +} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}PBL^{\prime +}$')
# plt.title('Positive SST Values: \n SST and PBL \n m={0:.2f}'.format(np.polyfit(np.isfinite(pbl_pos_pbl_lb),np.isfinite(sst_pos_pbl_lb),deg=1)[0]))
plt.title('c) Positive SST Values with PBL Thresholds: \n SST and PBL')
plt.legend(loc=0,fontsize='x-small')
plt.tight_layout()


# In[27]:


#PBL and WIND 
x_range_ur_pbl_pos=np.arange(0,1.5e-5,1e-7)

    
pbl_wind_neg_graph=plt.figure(figsize=(10,10))
sns.regplot(data=pbl_tresh_df,x="ur_pos_lb",y="pbl_pos_lb",x_bins=x_range_ur_pbl_pos,color='purple',label=r'PBL$<${0:1.3f}m m={1:.4f}'.format(pbl_lb,polyfit_windpbl_pbl_pos_lb[0]))
sns.regplot(data=pbl_tresh_mb_df,x="ur_pos_mb",y="pbl_pos_mb",x_bins=x_range_ur_pbl_pos,color='black',label=r'{0:.4f}$<$PBL$<${1:1.3f}m m={2:.4f}'.format(pbl_lb,pbl_ub,polyfit_windpbl_pbl_pos_mb[0]))
sns.regplot(data=pbl_tresh_df,x="ur_pos_ub",y="pbl_pos_ub",x_bins=x_range_ur_pbl_pos,color='orange',label=r'{0:.4f}m$<$PBL m={1:.4f} '.format(pbl_ub,polyfit_windpbl_pbl_pos_ub[0]))
sns.regplot(data=data_sst_pos,x='ur_pos',y='pbl_pos',x_bins=x_range_ur_pbl_pos,color='green',label='PBL total m={0:.4f}'.format(polyfit_windpbl_pbl_pos[0]))
plt.scatter(data_sst_pos['ur_pos'],data_sst_pos['pbl_pos'],alpha=0.25)
plt.xlim(0,1e-5)
plt.ylim(-0.003,0.003)
plt.xlabel(r'$ \frac{\partial}{\partial r}Ur ^ {\prime +} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}PBL^{\prime +}$')
# plt.title('Positive SST Values: \n SST and PBL \n m={0:.2f}'.format(np.polyfit(np.isfinite(pbl_pos_pbl_lb),np.isfinite(sst_pos_pbl_lb),deg=1)[0]))
plt.title('c) Positive SST Values with PBL thresholds: \n Ur and PBL')
plt.legend(loc=0,fontsize='x-small')
plt.tight_layout()

x_range_ur_pbl_neg=np.arange(-1e-5,1e-5,1e-7)

pbl_wind_neg_graph=plt.figure(figsize=(10,10))
sns.regplot(data=pbl_tresh_df,x="ur_neg_lb",y="pbl_neg_lb",x_bins=x_range_ur_pbl_neg,color='purple',label=r'PBL$<${0:1.3f}m m={1:.4f}'.format(pbl_lb, polyfit_windpbl_pbl_neg_lb[0]))
sns.regplot(data=pbl_tresh_mb_df,x="ur_neg_mb",y="pbl_neg_mb",x_bins=x_range_ur_pbl_neg,color='black',label=r'{0:.4f}$<$PBL$<${1:1.3f}m  m={2:.4f}'.format(pbl_lb,pbl_ub,polyfit_windpbl_pbl_neg_mb[0]))
sns.regplot(data=pbl_tresh_df,x="ur_neg_ub",y="pbl_neg_ub",x_bins=x_range_ur_pbl_neg,color='orange',label=r'{0:.4f}m$<$PBL m={1:.4f}'.format(pbl_lb,polyfit_windpbl_pbl_neg_ub[0]))
sns.regplot(data=data_sst_neg,x='ur_neg',y='pbl_neg',x_bins=x_range_ur_pbl_neg,color='green',label='PBL total m={0:.4f}'.format(polyfit_windpbl_pbl_neg[0]))
plt.scatter(data_sst_neg['ur_neg'],data_sst_neg['pbl_neg'],alpha=0.25)
plt.xlim(-1e-5,0)
plt.ylim(-0.003,0.002)
plt.xlabel(r'$ \frac{\partial}{\partial r}Ur ^ {\prime +} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}PBL^{\prime +}$')
# plt.title('Positive SST Values: \n SST and PBL \n m={0:.2f}'.format(np.polyfit(np.isfinite(pbl_pos_pbl_lb),np.isfinite(sst_pos_pbl_lb),deg=1)[0]))
plt.title('b) Negative SST Values with PBL thresholds: \n Ur and PBL')
plt.legend(loc=0,fontsize='x-small')

plt.tight_layout()


# In[28]:


##SST and UR 
x_range_ur_pbl_pos=np.arange(-0,7.5e-5,0.5e-6)

pbl_wind_neg_graph=plt.figure(figsize=(10,10))
sns.regplot(data=pbl_tresh_df,x="sst_pos_lb",y="ur_pos_lb",x_bins=x_range_ur_pbl_pos,color='purple',label=r'PBL$<${0:1.3f}m m={1:.4f}'.format(pbl_lb,polyfit_ursst_pbl_pos_lb[0]))
sns.regplot(data=pbl_tresh_mb_df,x="sst_pos_mb",y="ur_pos_mb",x_bins=x_range_ur_pbl_pos,color='black',label=r'{0:.4f}$<$PBL$<${1:1.3f}m m={2:.4f}'.format(pbl_lb,pbl_ub,polyfit_ursst_pbl_pos_mb[0]))
sns.regplot(data=pbl_tresh_df,x="sst_pos_ub",y="ur_pos_ub",x_bins=x_range_ur_pbl_pos,color='orange',label=r'{0:.4f}m$<$PBL m={1:.4f}'.format(pbl_ub,polyfit_ursst_pbl_pos_ub[0]))
sns.regplot(data=data_sst_pos,x='sst_pos',y='ur_pos',x_bins=x_range_ur_pbl_pos,color='green',label='PBL total m={0:.4f}'.format(polyfit_ursst_pbl_pos[0]))
plt.scatter(data_sst_pos['sst_pos'],data_sst_pos['ur_pos'],alpha=0.25)
plt.xlim(-0,5e-5) 
plt.ylim(-3e-5,4e-5)
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime +} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}Ur^{\prime +}$')
# plt.title('Positive SST Values: \n SST and PBL \n m={0:.2f}'.format(np.polyfit(np.isfinite(pbl_pos_pbl_lb),np.isfinite(sst_pos_pbl_lb),deg=1)[0]))
plt.title('Positive SST Values with PBL thresholds: \n SST and Ur')
plt.legend(loc=0,fontsize='x-small')
plt.tight_layout()

x_range_ur_pbl_neg=np.arange(-5e-5,0.1e-5,0.5e-6)

    
pbl_wind_neg_graph=plt.figure(figsize=(10,10))
sns.regplot(data=pbl_tresh_df,x="sst_neg_lb",y="ur_neg_lb",x_bins=x_range_ur_pbl_neg,color='purple',label=r'PBL$<${0:1.3f}m m={1:.4f}'.format(pbl_lb,polyfit_ursst_pbl_neg_lb[0]))
sns.regplot(data=pbl_tresh_mb_df,x="sst_neg_mb",y="ur_neg_mb",x_bins=x_range_ur_pbl_neg,color='black',label=r'{0:.4f}$<$PBL$<${1:1.3f}m m={2:.4f}'.format(pbl_lb,pbl_ub,polyfit_ursst_pbl_neg_mb[0]))
sns.regplot(data=pbl_tresh_df,x="sst_neg_ub",y="ur_neg_ub",x_bins=x_range_ur_pbl_neg,color='orange',label=r'{0:.4f}m$<$PBL m={1:.4f}'.format(pbl_ub,polyfit_ursst_pbl_neg_ub[0]))
sns.regplot(data=data_sst_neg,x="sst_neg",y="ur_neg",x_bins=x_range_ur_pbl_neg,color='green',label='PBL total m={0:.4f}'.format(polyfit_ursst_pbl_neg[0]))
plt.scatter(data_sst_neg['sst_neg'],data_sst_neg['ur_neg'],alpha=0.25)
plt.xlim(-5e-5,0.01e-4)
# plt.ylim(-3e-5,5e-5)
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime +} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}^Ur{\prime +}$')
# plt.title('Positive SST Values: \n SST and PBL \n m={0:.2f}'.format(np.polyfit(np.isfinite(pbl_pos_pbl_lb),np.isfinite(sst_pos_pbl_lb),deg=1)[0]))
plt.title('Negative SST Values with PBL thresholds: \n SST and Ur')
plt.legend(loc=0,fontsize='x-small')

plt.tight_layout()


# # Poster Figures- Similar method but with different values some code may overlap but it's best to keep it all together

# In[29]:


#This is just with PBL thresholds 
pbl_mean=np.mean(pbl)
pbl_std=np.std(pbl)
# pbl_std

ravel_pbl_total=np.ravel(pbl_total[ixx])
pbl_25=np.percentile(ravel_pbl_total, 25)
pbl_75=np.percentile(ravel_pbl_total, 75)
# plt.hist(ravel_pbl_total,bins=30)

lb=pbl_25
ub=pbl_75

# lb=pbl_mean-pbl_std
# ub=pbl_mean+pbl_std

# lb=600
# ub=1200

pbl_bstd=(pbl<lb)
pbl_btstd=((pbl>=lb)&(pbl<=ub))
pbl_abstd=(pbl>ub)

dummy_pbl=pbl.copy()
dummy_sst=sst.copy()
dummy_sst_pos_1= dsst_dr_pos.copy()
dummy_sst_neg_1=dsst_dr_neg.copy()

belowstd_pbl=dummy_pbl.where(pbl_bstd)
belowstd_sst=dummy_sst.where(pbl_bstd) 
belowstd_sst_pos=dummy_sst_pos_1.where(pbl_bstd)
belowstd_sst_neg=dummy_sst_neg_1.where(pbl_bstd)
belowstd_wind=dummy_wind.where(pbl_bstd)

btwstd_pbl=dummy_pbl.where(pbl_btstd)
btwstd_sst=dummy_sst.where(pbl_btstd)
btwstd_sst_pos=dummy_sst_pos_1.where(pbl_btstd)
btwstd_sst_neg=dummy_sst_neg_1.where(pbl_btstd)
btwstd_wind=dummy_wind.where(pbl_btstd)

abovestd_pbl=dummy_pbl.where(pbl_abstd)
abovestd_sst=dummy_sst.where(pbl_abstd)
abovestd_sst_pos=dummy_sst_pos_1.where(pbl_abstd)
abovestd_sst_neg=dummy_sst_neg_1.where(pbl_abstd)
abovestd_wind=dummy_wind.where(pbl_abstd)



#####Below STD#####
dsst_bstd_dr= along_wind_derivative(belowstd_sst,
                                dx=(dx),dy=(dy), #diff lon mean gives one value 
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)


dpbl_bstd_dr= along_wind_derivative(belowstd_pbl,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

dur_bstd_dr=along_wind_derivative(belowstd_wind,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)



ravel_sst_blstd=np.ravel(dsst_bstd_dr[ixx])
ravel_wind_blstd=np.ravel(dur_bstd_dr[ixx])
ravel_pbl_blstd=np.ravel(dpbl_bstd_dr[ixx])



##Between STD#####

dsst_btstd_dr= along_wind_derivative(btwstd_sst,
                                dx=(dx),dy=(dy), #diff lon mean gives one value 
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)


dpbl_btstd_dr = along_wind_derivative(btwstd_pbl,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

dur_btstd_dr=along_wind_derivative(btwstd_wind,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

ravel_sst_btstd=np.ravel(dsst_btstd_dr[ixx])
ravel_wind_btstd=np.ravel(dur_btstd_dr[ixx])
ravel_pbl_btstd=np.ravel(dpbl_btstd_dr[ixx])




####Above STD######

dsst_abstd_dr= along_wind_derivative(abovestd_sst,
                                dx=(dx),dy=(dy), #diff lon mean gives one value 
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)


dpbl_abstd_dr = along_wind_derivative(abovestd_pbl,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

dur_abstd_dr=along_wind_derivative(abovestd_wind,
                                dx=(dx),dy=(dy),
                                u_wind_ref=u_wind_filt,
                                v_wind_ref=v_wind_filt,
                                axis_x=2,axis_y=1)

ravel_sst_abstd=np.ravel(dsst_abstd_dr[ixx])
ravel_wind_abstd=np.ravel(dur_abstd_dr[ixx])
ravel_pbl_abstd=np.ravel(dpbl_abstd_dr[ixx])


data_std=pd.DataFrame({'sst_blstd':ravel_sst_blstd,'pbl_blstd':ravel_pbl_blstd,'u_blstd':ravel_wind_blstd,
                       'sst_btstd':ravel_sst_btstd, 'pbl_btstd':ravel_pbl_btstd,'u_btstd':ravel_wind_btstd,
                       'sst_abstd':ravel_sst_abstd,'pbl_abstd':ravel_pbl_abstd,'u_abstd':ravel_wind_abstd})


p=data_std['sst_blstd'].isna().sum()
l=len(data_std['sst_blstd'])

sst_blstd=data_std['sst_blstd'].dropna()
pbl_blstd=data_std['pbl_blstd'].dropna()
u_blstd=data_std['u_blstd'].dropna()


sst_btstd=data_std['sst_btstd'].dropna()
pbl_btstd=data_std['pbl_btstd'].dropna()
u_btstd=data_std['u_btstd'].dropna()

sst_abstd=data_std['sst_abstd'].dropna()
pbl_abstd=data_std['pbl_abstd'].dropna()
u_abstd=data_std['u_abstd'].dropna()


# In[30]:


lb_sst_u=np.polyfit(sst_blstd,u_blstd,deg=1)
mb_sst_u=np.polyfit(sst_btstd,u_btstd,deg=1)
ub_sst_u=np.polyfit(sst_abstd,u_abstd,deg=1)
total_sst_u=np.polyfit(data_f['sst'],data_f['wind'],deg=1)
r_total=scipy.stats.linregress(data_f['sst'], y=data_f['wind'], alternative='two-sided') 


plt.figure(figsize=(10,10), dpi=300)
mask_SST_wind=plt.figure(figsize=(10,10))
# x_range_2_5= np.arange(-5e-5,5e-5,2e-6)
x_range_2= np.arange(-6e-4,6e-4,3e-6)
plt.scatter(data_f['sst'],data_f['wind'],alpha=0.1)
# sns.regplot(data=data_std, x="sst_blstd", y="u_blstd",x_bins=x_range_2,color='purple',label=r'PBL$<${0:1.3f}m m={1:1.3f}(m/°Cs)'.format(lb,lb_sst_u[0]))#change the titles to be values of the mask 
# sns.regplot(data=data_std, x="sst_btstd", y="u_btstd",x_bins=x_range_2,color='black',label=r'{0:1.3f} m$<$PBL$<$ {1:1.3f} m m={2:1.3f}(m/°Cs)'.format(lb,ub,mb_sst_u[0]))
# sns.regplot(data=data_std, x="sst_abstd", y="u_abstd",x_bins=x_range_2,color='orange',label=r'{0:1.3f}m$<$PBL m={1:1.3f}(m/°Cs)'.format(ub,ub_sst_u[0]))
# sns.regplot(data=data_f, x="sst", y="wind",x_bins=x_range_2,color='darkorange',label='Total PBL m={0:1.3f}(m/°Cs)'.format(total_sst_u[0]))
sns.regplot(data=data_f, x="sst", y="wind",x_bins=x_range_2,color='darkorange')
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime} $(°C/m)')
plt.ylabel(r'$ \frac{\partial}{\partial r}Ur ^ {\prime} $(1/s)')
# plt.legend(loc='upper left', fontsize='x-small')
plt.xlim(-5.5e-5,5.5e-5) 
# plt.ylim(-1.2e-5,1.2e-5)
plt.ylim(-2e-5,2e-5)
# plt.title('SST and Ur \n m={0:.2f}'.format(np.polyfit(ravel_sst,ravel_wind,deg=1)[0]))
# plt.title('a) SST and Ur with PBL Thresholds \n')
plt.title('a) SST and Ur \n m={0:1.3f}(m/°Cs) $r^2$={1:1.3f}'.format(total_sst_u[0],r_total[2])) 
plt.tight_layout()


# In[31]:


lb_sst_u=np.polyfit(sst_blstd,u_blstd,deg=1)
mb_sst_u=np.polyfit(sst_btstd,u_btstd,deg=1)
ub_sst_u=np.polyfit(sst_abstd,u_abstd,deg=1)
total_sst_u=np.polyfit(data_f['sst'],data_f['wind'],deg=1)


# plt.figure(figsize=(10,10))
mask_SST_wind=plt.figure(figsize=(10,10),dpi=300)
# x_range_2_5= np.arange(-5e-5,5e-5,2e-6)
x_range_2= np.arange(-6e-4,6e-4,3e-6)
plt.scatter(data_f['sst'],data_f['wind'],alpha=0.1)
sns.regplot(data=data_std, x="sst_blstd", y="u_blstd",x_bins=x_range_2,color='blueviolet',label=r'PBL$<${0:1.3f}m m={1:1.3f}(m/°Cs)'.format(lb,lb_sst_u[0]))#change the titles to be values of the mask 
sns.regplot(data=data_std, x="sst_btstd", y="u_btstd",x_bins=x_range_2,color='crimson',label=r'{0:1.3f} m$<$PBL$<$ {1:1.3f} m m={2:1.3f}(m/°Cs)'.format(lb,ub,mb_sst_u[0]))
sns.regplot(data=data_std, x="sst_abstd", y="u_abstd",x_bins=x_range_2,color='orange',label=r'{0:1.3f}m$<$PBL m={1:1.3f}(m/°Cs)'.format(ub,ub_sst_u[0]))
sns.regplot(data=data_f, x="sst", y="wind",x_bins=x_range_2,color='forestgreen',label='Total PBL m={0:1.3f}(m/°Cs)'.format(total_sst_u[0]))
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime} $(°C/m)')
plt.ylabel(r'$ \frac{\partial}{\partial r}Ur ^ {\prime} $(1/s)')
plt.legend(loc='upper left', fontsize='x-small')
plt.xlim(-5.5e-5,5.5e-5) 
# plt.ylim(-1.2e-5,1.2e-5)
plt.ylim(-2e-5,2e-5)
# plt.title('SST and Ur \n m={0:.2f}'.format(np.polyfit(ravel_sst,ravel_wind,deg=1)[0]))
plt.title('d) SST and Ur with PBL Thresholds \n')
plt.tight_layout()


# In[32]:


x_range_2= np.arange(-6e-4,6e-4,1e-6)
plt.figure(figsize=(10,10),dpi=300) #figsize #increase x-range 1e-6 see above
# sns.regplot(ravel_sst,ravel_wind,x_bins=np.arange(-2,2.1,0.1),ci=95)
lb_slope=np.polyfit(sst_blstd,pbl_blstd,deg=1)
mb_slope=np.polyfit(sst_btstd,pbl_btstd,deg=1)
ub_slope=np.polyfit(sst_abstd,pbl_abstd,deg=1)
total_slope=np.polyfit(data_f['sst'],data_f['pbl'],deg=1)
r_total=scipy.stats.linregress(data_f['sst'], y=data_f['pbl'], alternative='two-sided')

plt.scatter(data_f['sst'],data_f['pbl'],alpha=0.1)
# sns.regplot(data=data_std, x="sst_blstd", y="pbl_blstd",x_bins=x_range_2,color='purple',label=r'PBL$<${0:1.3f}m m={1:1.3f}(m/°C)'.format(lb,lb_slope[0]))#had to make a dataframe
# sns.regplot(data=data_std, x="sst_btstd", y="pbl_btstd",x_bins=x_range_2,color='black',label=r'{0:1.3f} m$<$PBL$<$ {1:1.3f}m m={2:1.3f}(m/°C)'.format(lb,ub,mb_slope[0]))
# sns.regplot(data=data_std, x="sst_abstd", y="pbl_abstd",x_bins=x_range_2,color='orange',label=r'{0:1.3f}$<$PBLm m={1:1.3f}(m/°C)'.format(ub,ub_slope[0]))
sns.regplot(data=data_f, x="sst", y="pbl",x_bins=x_range_2,color='darkorange',label='PBL Total m={0:1.3f} (m/°C)'.format(total_slope[0]))
plt.xlabel(r'$ \frac{\partial}{\partial r}SST ^ {\prime} $(°C/m)') #Check which SST would look better location wise
plt.ylabel(r' $\frac{\partial}{\partial r}PBL^{\prime}$')
# plt.legend(loc='upper left', fontsize='x-small')
plt.xlim(-5e-5,5e-5)
plt.ylim(-0.004,0.004)
# plt.title('SST and PBL \n m={0:.2f}'.format(np.polyfit(ravel_sst,ravel_pbl,deg=1)[0]))
# plt.title('c) SST and PBL \n m={0:1.3f} (m/°C)'.format(total_slope[0]))
plt.title('c) SST and PBL \n m={0:1.3f}(m/°C) $r^2$={1:1.3f}'.format(total_slope[0],r_total[2])) 

# SST and PBL \n lower bond  m={0:.2f} \n middle bond m={1:0.2f} \n upper bond m={2:0.2f}'.format([0],np.polyfit(sst_btstd,pbl_btstd,deg=1)[0],np.polyfit(sst_abstd,pbl_abstd,deg=1)[0]))
# plt.title(r'$ PBL and SST with masks$', fontsize='small') #pasting the slope 
plt.tight_layout()


# # Saving as a .py file

# In[ ]:




