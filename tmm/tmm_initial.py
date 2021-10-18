
# -*- coding: utf-8 -*-
"""
Examples of plots and calculations using the tmm package.
"""

from __future__ import division, print_function, absolute_import

from tmm.tmm_core import (coh_tmm, unpolarized_RT, ellips,
                       position_resolved, find_in_structure_with_inf)

from numpy import pi, linspace, inf, array
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

def sample2():
    """
    Here's the transmitted intensity versus wavelength through a single-layer
    film which has some complicated wavelength-dependent index of refraction.
    (I made these numbers up, but in real life they could be read out of a
    graph / table published in the literature.) Air is on both sides of the
    film, and the light is normally incident.
    """
    #index of refraction of my material: wavelength in nm versus index.
    SiO2_nk_data = np.loadtxt('SiO2_index.csv',encoding='UTF-8-sig',delimiter=',')
    SiO2_nk_data[:,0] = SiO2_nk_data[:,0]*1000
    TiO2_nk_data = np.loadtxt('TiO2_index.csv',encoding='UTF-8-sig',delimiter=',')
    TiO2_nk_data[:,0] = TiO2_nk_data[:,0]*1000
    
    # TiO2_nk_data = array([[200, 2.1+0.1j],
    #                           [300, 2.4+0.3j],
    #                           [400, 2.3+0.4j],
    #                           [500, 2.2+0.4j],
    #                           [750, 2.2+0.5j]])
    TiO2_nk_fn = interp1d(TiO2_nk_data[:,0].real,
                              TiO2_nk_data[:,1], kind='quadratic',fill_value="extrapolate")
    # SiO2_nk_data = array([[200, 1.1+0.1j],
    #                           [300, 1.2+0.3j],
    #                           [400, 1.7+0.4j],
    #                           [500, 1.6+0.4j],
    #                           [750, 2.2+0.5j]])
    SiO2_nk_fn = interp1d(SiO2_nk_data[:,0].real,
                              SiO2_nk_data[:,1], kind='quadratic',fill_value="extrapolate")
    
    
    
    
    
    d_list = [inf, 100,200,100,200,100,200, 200,200,200,200,inf] #in nm test_0
    d_list = [inf, 100,200,100,200,100,200, 300,200,200,200,inf] #in nm test_1
    d_list = [inf, 134,138,141,123,136,116, 124,121,130,116,
              134,138,141,123,136,116, 124,121,130,116,
              134,138,141,123,136,116, 124,121,130,116,
              134,138,141,123,136,116, 124,121,130,116,inf] #in nm test_1
    lambda_list = linspace(400, 800, 10000) #in nm
    T_list = []
    for lambda_vac in lambda_list:
        n_list = [1, SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                   SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                   SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                   SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),
                  SiO2_nk_fn(lambda_vac), TiO2_nk_fn(lambda_vac),1]
        T_list.append(coh_tmm('s', n_list, d_list, 0, lambda_vac)['T'])
    plt.figure()
    plt.plot(lambda_list, T_list,label='tmm')
    fdtd = np.loadtxt('../FDTDdata/fin_1.txt',encoding='UTF-8-sig',delimiter=',')
    fdtd[:,0]=fdtd[:,0]*1e9
    # add = -np.array(range(100))*0.08+18
    plt.plot(fdtd[:,0]+18,fdtd[:,1],label='fdtd')
    plt.legend()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Fraction of power transmitted')
    plt.title('Transmission at normal incidence')
    plt.show()
    
sample2()  
















    
    