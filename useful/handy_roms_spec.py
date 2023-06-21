"""
@author: dantecn, iufarias
"""

import numpy as np
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from glob import glob
from netCDF4 import Dataset
from datetime import datetime
import multiprocessing as mp
import scipy.integrate as integ



### FUNCTION UVW2RHO_3D ##################################################
def uvw2rho_3d(ufield,vfield,wfield):

    """
    ################################################################
    #
    #   compute the values of u,v at t,s points ...
    #
    #   adptation of the w matrix
    #
    #   Dante C Napolitano, LaDO IOUSP
    #   dante.napolitano@usp.br (Jan 2018)
    ################################################################
    """

    ur_field = 0.5 * (ufield[:,:,:-1] + ufield[:,:,1:])
    ur_field = ur_field[:,1:-1,:]
    vr_field = 0.5 * (vfield[:,:-1,:] + vfield[:,1:,:])
    vr_field = vr_field[:,:,1:-1]

    wr_field = wfield[:,1:-1,1:-1]

    return ur_field,vr_field,wr_field



### FUNCTION UVW2RHO_4D ##################################################
def uvw2rho_4d(ufield,vfield,wfield):

    """
    ################################################################
    #
    #   compute the values of u,v at t,s points ...
    #
    #   adptation of the w matrix
    #
    #   Dante C Napolitano, LaDO IOUSP
    #   dante.napolitano@usp.br (Jan 2018)
    ################################################################
    """

    ur_field = 0.5 * (ufield[:,:,:,:-1] + ufield[:,:,:,1:])
    ur_field = ur_field[:,:,1:-1,:]
    vr_field = 0.5 * (vfield[:,:,:-1,:] + vfield[:,:,1:,:])
    vr_field = vr_field[:,:,:,1:-1]

    wr_field = wfield[:,:,1:-1,1:-1]

    return ur_field,vr_field,wr_field


### FUNCTION TS2RHO_3D ##################################################
def ts2rho_3d(tfield,sfield):

    """
    ################################################################
    #
    #   adptation of the t,s matrices
    #
    #   Dante C Napolitano, LaDO IOUSP
    #   dante.napolitano@usp.br (Jan 2018)
    ################################################################
    """

    tr_field = tfield[:,1:-1,1:-1]
    sr_field = sfield[:,1:-1,1:-1]

    return tr_field,sr_field

### FUNCTION UNIQUE GRID ##################################################
def unique_grid(prop):

    """
    #################################################################
    #
    #   performs an adaption on "rho_grid" so all properties may share
    #   the same lon, lat coordinates
    #
    #   Dante C Napolitano, LaDO IOUSP
    #   dante.napolitano@usp.br (Jan 2018)
    #################################################################
    """
    return prop[1:-1,1:-1]

### FUNCTION ZCOORDS #######################################################
def zcoords(prop,z,znew):

    DF = pd.DataFrame([],index=znew)
    i,j,k = prop.shape
    ct=0
    for ri,zi in zip(prop.reshape(i,j*k).T,z.reshape(i,j*k).T):
        df = pd.DataFrame(ri[::-1],index=zi[::-1],columns=[ct])
        df = df.reindex(znew[znew<df.index.values[-1]],method='backfill',limit=1)
        df = df.interpolate(limit_direction='backward')
        DF = DF.join(df)
        ct+=1

    return np.reshape(DF.values,newshape=(len(znew),j,k))



'Spectral Ogive Graph'
def spec_ogive(spec,kr):
    
    Ogive_Kr=np.array(kr)
    # Middle Value Wavenumber position for plot
    Ogive_Krx=(Ogive_Kr[:-1]+Ogive_Kr[1:])/2
    # dK=np.diff(Ogive_Kr)[0]
    
    #Flipped Integration in wavenumber and then flipped again
    Ogive_inv=integ.cumtrapz(y=spec[::-1],x=Ogive_Kr[::-1]);
    Ogive_Spec=-Ogive_inv[::-1];


    return Ogive_Krx,Ogive_Spec



def ogive(deltaf,G):
    '''
    by nldias (Nelson Dias 2008)

    ogive(deltaf,G): use very simple integration to calculate the ogive 
    from a spectrum G with data sampled at frequency deltaf.
    2017-01-10T09:40:31 going back to a single frequency for Os and Gs
    '''
    
    #WORKS JUST AS MUCH AS THE FIRST ONE
    # --------------------------------------------------------------------
    # 2016-10-08T09:47:12 re-created with numpy
    # --------------------------------------------------------------------
    M1 = len(G)
    Og = np.zeros(M1,float)
    Og[0:M1] = np.cumsum(np.flipud(G[0:M1]))
    Og *= deltaf
    Og = np.flipud(Og)
    return Og



def grid_dist_same(lon,lat):
    import seawater as sw
    if len(lon.shape)==2:
        grid_x=np.zeros(lon.shape)+np.nan
    else:
        grid_x=np.zeros([lat.shape[0],lon.shape[0]])
    
    grid_d=sw.dist(lon=lon,lat=lat)[0]
    grid_x[0:-1,:]=grid_d
    grid_x[-1,:]=grid_d[-1,:]
    
    return grid_x

def area_filter(lon,lat,filter_scale,dim_x,dim_y):
    #paramerers for convolution
    if dim_x=='x_rho' and dim_y=='y_rho':
    
        grid=grid_dist_same(lon,lat)
        gs=xr.DataArray(grid,dims=(dim_y,dim_x))

        radius = int( (filter_scale / gs.min().compute() / 2 ).round()) 
        # get the radius in grid-cells that covers the convolution kernel also for the smallest grid-spacing
        window_size = 2 * radius + 1


        gsr = gs.rolling(x_rho=window_size,center=True).construct("lon_window").rolling(y_rho=window_size, center=True).construct("lat_window")

        gsr_lat = gsr.cumsum("lat_window")
        gsr_lat -= gsr_lat.isel(lat_window=radius)
        gsr_lon = gsr.cumsum("lon_window")
        gsr_lon -= gsr_lon.isel(lon_window=radius)
        circ = ((gsr_lat ** 2 + gsr_lon ** 2) ** 0.5 < filter_scale / 2)
        Asum = (circ * (gsr ** 2)).sum(dim = ["lat_window","lon_window"])
    return gs,radius,circ,Asum

    
def apply_area_filter(var,gs,radius,circ,Asum):
    
    window_size = 2 * radius + 1
    
    var_sm = var.copy()
    var_sm += np.nan;



    varA  = gs ** 2 * var # multiplication with area
    varAr = varA.rolling(x_rho=window_size,center=True).construct("lon_window").rolling(y_rho=window_size, center=True).construct("lat_window")
    var_sm_tmp = ((varAr * circ).sum(dim = ["lat_window","lon_window"]) / Asum)
    # set the pixels at the boundary to NaN, where the convolution kernel extends over the boundary
    var_sm_tmp2 = np.zeros((var_sm_tmp.shape)) + np.nan
    var_sm_tmp2[radius:-radius,radius:-radius] = var_sm_tmp[radius:-radius,radius:-radius]
    var_sm[:] = var_sm_tmp2
    return var_sm
    


def ultimate_filter2d(lon,lat,var,filter_scale,dim_x,dim_y):
    #filter scale in km
    if dim_x=='x_rho' and dim_y=='y_rho':
    
        grid=grid_dist_same(lon,lat)
        gs=xr.DataArray(grid,dims=(dim_y,dim_x))

        var_sm = var.copy()
        var_sm += np.nan;

        radius = int( (filter_scale / gs.min().compute() / 2 ).round()) 
        # get the radius in grid-cells that covers the convolution kernel also for the smallest grid-spacing
        window_size = 2 * radius + 1


        gsr = gs.rolling(x_rho=window_size,center=True).construct("lon_window").rolling(y_rho=window_size, center=True).construct("lat_window")

        gsr_lat = gsr.cumsum("lat_window")
        gsr_lat -= gsr_lat.isel(lat_window=radius)
        gsr_lon = gsr.cumsum("lon_window")
        gsr_lon -= gsr_lon.isel(lon_window=radius)
        circ = ((gsr_lat ** 2 + gsr_lon ** 2) ** 0.5 < filter_scale / 2)
        Asum = (circ * (gsr ** 2)).sum(dim = ["lat_window","lon_window"])



        varA  = gs ** 2 * var # multiplication with area
        varAr = varA.rolling(x_rho=window_size,center=True).construct("lon_window").rolling(y_rho=window_size, center=True).construct("lat_window")
        var_sm_tmp = ((varAr * circ).sum(dim = ["lat_window","lon_window"]) / Asum)
        # set the pixels at the boundary to NaN, where the convolution kernel extends over the boundary
        var_sm_tmp2 = np.zeros((var_sm_tmp.shape)) + np.nan
        var_sm_tmp2[radius:-radius,radius:-radius] = var_sm_tmp[radius:-radius,radius:-radius]
        var_sm[:] = var_sm_tmp2
    return var_sm

def spec_ogive(spec,kr):
    
    Ogive_Kr=np.array(kr)
    # Middle Value Wavenumber position for plot
    Ogive_Krx=(Ogive_Kr[:-1]+Ogive_Kr[1:])/2
    # dK=np.diff(Ogive_Kr)[0]
    
    #Flipped Integration in wavenumber and then flipped again
    Ogive_inv=integ.cumtrapz(y=spec[::-1],x=Ogive_Kr[::-1]);
    Ogive_Spec=-Ogive_inv[::-1];


    return Ogive_Krx,Ogive_Spec

def ogive(deltaf,G):
   '''
   ogive(deltaf,G): use very simple integration to calculate the ogive 
   from a spectrum G with data sampled at frequency deltaf.
   2017-01-10T09:40:31 going back to a single frequency for Os and Gs
   '''
# --------------------------------------------------------------------
# 2016-10-08T09:47:12 re-created with numpy
# --------------------------------------------------------------------
   M1 = len(G)
   Og = np.zeros(M1,float)
   Og[0:M1] = np.cumsum(np.flipud(G[0:M1]))
   Og *= deltaf
   Og = np.flipud(Og)
   return Og


