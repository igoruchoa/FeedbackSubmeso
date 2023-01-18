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
    # --------------------------------------------------------------------
    # 2016-10-08T09:47:12 re-created with numpy
    # --------------------------------------------------------------------
    M1 = len(G)
    Og = np.zeros(M1,float)
    Og[0:M1] = np.cumsum(np.flipud(G[0:M1]))
    Og *= deltaf
    Og = np.flipud(Og)
    return Og
