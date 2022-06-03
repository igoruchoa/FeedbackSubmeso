#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:30:38 2018

@author: dantecn
"""

# save pickle from netcdf files
import numpy as np

#from OceanLab import *
import numpy as np
import pandas as pd
import os
#import romslab

from collections import OrderedDict
# from romslab import zlevs
from glob import glob
#from OceanLab import utils
from netCDF4 import Dataset
# from netcdftime import utime
from datetime import datetime

import multiprocessing as mp

#from zlevs.py
def csf(sc, theta_s,theta_b):
#######################################################################
#
#  function h = csf(sc, theta_s,theta_b);
#
#  Further Information:
#  http://www.brest.ird.fr/Roms_tools/
#
#  This file is part of ROMSTOOLS
#
#  ROMSTOOLS is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published
#  by the Free Software Foundation; either version 2 of the License,
#  or (at your option) any later version.
#
#  ROMSTOOLS is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#  MA  02111-1307  USA
#
######################################################################

    if (theta_s > 0 ):
        csrf=(1-np.cosh(sc*theta_s))/(np.cosh(theta_s)-1)
    else:
        csrf=-sc**2

    if (theta_b > 0):
        h = (np.exp(theta_b*csrf)-1)/(1-np.exp(-theta_b))
    else:
        h  = csrf.copy();

    return h




def zlevs(h,zeta,theta_s,theta_b,hc,N,type='r',vtransform=2):
    """
    ################################################################
    #
    #  function z = zlevs(h,zeta,theta_s,theta_b,hc,N,type);
    #
    #  this function compute the depth of rho or w points for ROMS
    #
    #  On Input:
    #
    #    type    'r': rho point 'w': w point
    #
    #  On Output:
    #
    #    z       Depths (m) of RHO- or W-points (3D matrix).
    #
    #  Further Information:
    #  http://www.brest.ird.fr/Roms_tools/
    #
    #  This file is part of ROMSTOOLS
    #
    #  ROMSTOOLS is free software; you can redistribute it and/or modify
    #  it under the terms of the GNU General Public License as published
    #  by the Free Software Foundation; either version 2 of the License,
    #  or (at your option) any later version.
    #
    #  ROMSTOOLS is distributed in the hope that it will be useful, but
    #  WITHOUT ANY WARRANTY; without even the implied warranty of
    #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #  GNU General Public License for more details.
    #
    #  You should have received a copy of the GNU General Public License
    #  along with this program; if not, write to the Free Software
    #  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
    #  MA  02111-1307  USA
    #
    #  Copyright (c) 2002-2006 by Pierrick Penven
    #  e-mail:Pierrick.Penven@ird.fr
    #  Translated to Python by Rafael Soutelino, rsoutelino@gmail.com
    #  Last Modification: Aug, 2010
    ################################################################
    """

    M, L = h.shape

    # Set S-Curves in domain [-1 < sc < 0] at vertical W- and RHO-points.

    sc_r=np.zeros(N)
    Cs_r=np.zeros(N)
    sc_w=np.zeros(N+1)
    Cs_w=np.zeros(N+1)

    if (vtransform == 2):
        ds=1./N;
        if type=='w':

            sc_w[0] = -1.0
            sc_w[-1] =  0
            Cs_w[0] = -1.0
            Cs_w[-1] =  0

            sc_w[1:N] = ds*(np.arange(1,N)-N)
            Cs_w = csf(sc_w, theta_s,theta_b)
            N+=1
    #    Still in matlab code
    #    disp(['===================================='])
    #    for k=N:-1:1
    #        disp(['Niveau S=',num2str(k),' Cs=',num2str( Cs_w(k), '%8.7f')])
    #    end
    #    disp(['===================================='])

        else:

            sc = ds*(np.arange(1,N+1)-N-0.5)
            Cs_r = csf(sc, theta_s,theta_b)
            sc_r = sc.copy()
    #    Still in matlab code
    #    disp(['===================================='])
    #    for k=N:-1:1
    #        disp(['Niveau S=',num2str(k),' Cs=',num2str( Cs_r(k), '%8.7f')])
    #    end
    #    disp(['===================================='])

    else:
        cff1 = 1.0 / np.sinh( theta_s )
        cff2 = 0.5 / np.tanh( 0.5*theta_s )
        ##
        if type=='w':
            sc = ( np.arange(0,N+1) - N ) / N
            N  = N + 1
        else:
            sc = ( np.arange(1,N+1) - N - 0.5 ) / N
        ##

        Cs = (1 - theta_b) * cff1 * np.sinh( theta_s * sc ) + \
            theta_b * ( cff2 * np.tanh(theta_s *(sc + 0.5) ) - 0.5 )

    #   Still in matlab code
    #    disp(['===================================='])
    #    for k=N:-1:1
    #        disp(['Niveau S=',num2str(k),' Cs=',num2str( Cs(k), '%8.7f')])
    #    end
    #    disp(['===================================='])

    # ------------------------------------------------------------------ #

    # Create S-coordinate system: based on model topography h(i,j),
    # fast-time-averaged free-surface field and vertical coordinate
    # transformation metrics compute evolving depths of of the three-
    # dimensional model grid. Also adjust zeta for dry cells.

    Dcrit = 0.2 # min water depth in dry cells
    h[h==0] = 1e-14
    zeta[zeta<(Dcrit-h)] = Dcrit - h[zeta<(Dcrit-h)]
    hinv = 1./h
    z = np.zeros([N,M,L])

    if (vtransform == 2):

        if type=='w':
            cff1 = Cs_w.copy()
            cff2 = sc_w.copy() + 1
            sc = sc_w.copy()
        else:
            cff1 = Cs_r.copy()
            cff2 = sc_r.copy() + 1
            sc = sc_r.copy()

        h2 = h + hc
        cff = hc*sc
        h2inv = 1./h2

        for k in np.arange(0,N):
            z0 = cff[k] + cff1[k]*h
            z[k,:,:] = (z0*h)/h2 + zeta*(1.+ z0*h2inv)

    else:

        cff1 = Cs.copy()
        cff2 = sc + 1
        cff = hc*(sc-Cs)

        for k in range(0, N):
            z0         = cff[k] + cff1[k]*h
            z[k,:,:] = z0 + zeta*( 1 + z0*hinv )

    return z


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
'''
# zlevs parameters
# execfile('/home/dantecn/Documents/ROMS/zlevs.py')

############################################################################

#ftime = utime('hours since 2000-01-01 00:00:00')

# main directory
roots = '/home/dantecn/Documents/ROMS/rodada_Dinamo/pickledata'
#rootl = '/home/dantecn/Documents/ROMS/rodada_Dinamo/ncdata'
rootl = '/media/dantecn/campo02/ROMS_Dinamo/6km'

# years of netcdf data
#ye = np.arange(2003,2006)
#ye = [2006]

prop = OrderedDict({'uvw':'VEL_UVW','ts':'TEMP_SAL'})

#p = 1;
p = 0;
year = 2005;
os.system(u'mkdir %s/%s'%(roots,year))
os.system(u'mkdir %s/%s/%s'%(roots,year,prop.values()[p]))

# list all files from a year
pathread = rootl+'/roms_avg2_SODA_Y%s*.nc'%(year)
lista = glob(pathread)
lista.sort()

def romspickle(filename):

    theta_s = 6.; theta_b = 0.

	# READ daily files
    nc = Dataset(filename)

	# lon,lat grid postions
    [lon,lat] = map(unique_grid, [nc['lon_rho'][:],nc['lat_rho'][:]])
    lon,lat = lon[0,:],lat[:,0]

    pathsave = roots+'/%s/%s/'%(year,prop.values()[p])

    month=int(filename.split('_')[-1][6:8])

    # number of days in month
    timespan = len(nc['time'][:])

    for t in range(timespan):

        # to save files daily
        dias=pd.date_range(start='%s-%s-01'%(year,month),
                           end='%s-%s-%s'%(year,month,timespan),Freq='1d')

        dia = dias[t].strftime('%Y%m%d')

        # 3 subregions, everyday
        loma,lomi,lama,lami = -37.8,-40.3,-19.5,-22.

        cutlo = (lon>=lomi) & (lon<=loma)
        cutla = (lat>=lami) & (lat<=lama)

        lons=lon[cutlo]
        lats=lat[cutla]

        # applying grid and subset to depth and mask matrices

        [mask,h] = map(unique_grid, [nc['mask_rho'][:],nc['h'][:]])

        h = h[cutla,:]; h = h[:,cutlo]

        # set mask on land
        mask = mask[cutla,:]; mask = mask[:,cutlo]
        mask = (mask == 0)

        # sea surface height
        zeta = unique_grid(nc['zeta'][t,:,:])
        zeta = zeta[cutla,:];   zeta = zeta[:,cutlo]

        # depths at rho_grid
        z = -1.*zlevs(h,zeta,theta_s,theta_b,nc['hc'][:],30,type='r',vtransform=2)

        cutz = z<=1509 # 1500 > z > 0.

        #################################
        ##### velocity u and v data #####
        #################################
        if  prop.keys()[p] == 'uvw':

            # velocities
            u_raw = nc['u'][t,:]
            v_raw = nc['v'][t,:]
            w_raw = nc['w'][t,:]

            u,v,w = uvw2rho_3d(u_raw,v_raw,w_raw)

            u = u[:,cutla,:];   u = u[:,:,cutlo]
            v = v[:,cutla,:];   v = v[:,:,cutlo]
            w = w[:,cutla,:];   w = w[:,:,cutlo]

            # masked array
            u[:,mask] = np.nan

            # masked array
            v[:,mask] = np.nan

            # Interpolate U and V to regular z-grid
            ztot = np.arange(0,np.int(np.nanmax(z[cutz])),10)

            # interpolate series index

            unew = zcoords(u,z,ztot)
            vnew = zcoords(v,z,ztot)
            wnew = zcoords(w,z,ztot)

            UVDIC = dict({'u':unew,'v':vnew,'w':wnew,'z':ztot,
                      'lat':lats,'lon':lons,'time':dias[t]})

            print('xinhovel %d'%t)
            del unew,vnew,wnew,ztot,u,v,w,lons,lats,mask,h,zeta,u_raw,v_raw,w_raw

        ####################################
        ##### temperature and salinity #####
        ####################################
        elif prop.keys()[p] == 'ts':

            salt = nc['salt'][t,:]
            temp = nc['temp'][t,:]

            temp,salt = ts2rho_3d(temp,salt)

            temp = temp[:,cutla,:];   temp = temp[:,:,cutlo]
            salt = salt[:,cutla,:];   salt = salt[:,:,cutlo]

            # masked array
            salt[:,mask] = np.nan

            # masked array
            temp[:,mask] = np.nan

            # Interpolate U and V to regular z-grid
            ztot = np.arange(0,np.int(np.nanmax(z)),10)

            # interpolate series index

            tnew = zcoords(temp,z,ztot)
            snew = zcoords(salt,z,ztot)

            TSDIC = dict({'temp':tnew,'salt':snew,'z':ztot,
                          'lat':lats,'lon':lons,'time':dias[t]})

            print('xinhots %d'%t)
            del tnew,snew,ztot,temp,salt,lons,lats,mask,h,zeta

        ############################################################
        if  prop.keys()[p] == 'uvw':
            utils.save_pickle(UVDIC,pathsave+'UVW_%s'%(dia))
        elif prop.keys()[p] == 'ts':
            utils.save_pickle(TSDIC,pathsave+'TS_%s'%(dia))
        ############################################################

    return

# RUN FNCTIONS
pool = mp.Pool(mp.cpu_count())

tst=pool.map(romspickle,lista)

print('saved year %s - %s'%(year,prop.keys()[p]))
'''
# end of loop
# hãhãhãhãhãhã
#execfile('/home/dantecn/Documents/Utils/inutils/playxinho.py')
#


def zlev(ncw):

    h = ncw['h'][:]
    hc = ncw['hc'][:]
    theta_s,theta_b,N = 6,0,30
    type='r'
    vtransform=2
    zeta = ncw['zeta'][:]

    comp = np.arange(0,30)

    for d in np.arange(0,zeta.shape[0]):
        z = zlevs(h,zeta[d,:,:],theta_s,theta_b,hc,N,type,vtransform);
        i,j,k = z.shape
        zn = z.reshape(i,j*k)

        for zi in zn.T:
            I = np.argsort(zi)
            ok = (I == comp);
            if any(~ok):
                print('DEU PAU LIXO')

        print('dia %s funcionou!'%d)
    print('mes funcionou!')
    return z
