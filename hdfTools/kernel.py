import os
import pandas as pd, numpy as np, h5py
from math import pi
from .helpers import *

"""
This module contains functions to calculate BRDF scattering kernels. Equations can be found
in the following papers:
  
Colgan, M. S., Baldeck, C. A., Feret, J. B., & Asner, G. P. (2012). 
Mapping savanna tree species at ecosystem scales using support vector machine classification 
and BRDF correction on airborne hyperspectral and LiDAR data. 
Remote Sensing, 4(11), 3462-3480.            
    
Schlapfer, D., Richter, R., & Feingersh, T. (2015). 
Operational BRDF effects correction for wide-field-of-view optical scanners (BREFCOR). 
IEEE Transactions on Geoscience and Remote Sensing, 53(4), 1855-1864.

Wanner, W., Li, X., & Strahler, A. H. (1995). 
On the derivation of kernels for kernel-driven models of bidirectional reflectance.
Journal of Geophysical Research: Atmospheres, 100(D10), 21077-21089.
"""

def geom_sol_view(metadata):
    """Return viewing and solar geometry.
    
    Retrieve and calculate the solar and viewing geometry needed for generating
    the BRDF scattering kernels.
    
    Parameters
    ----------
    metadata : HDF metadata object

    Returns
    -------
    solZn : m x n numpy array of solar zenith angles in radians
    viewZn : m x n numpy array of viewing zeniths angles in radians
    relAz : m x n numpy array of raaltive azimuth angles in radians

    """
    solZn = np.radians(metadata['Logs']['Solar_Zenith_Angle'].value)
    solAz = np.radians(metadata['Logs']['Solar_Azimuth_Angle'].value)
       
    viewZn = np.radians(metadata['to-sensor_Zenith_Angle'][:,:])
    viewAz = np.radians(metadata['to-sensor_Azimuth_Angle'][:,:])
    
    solZn = solZn * np.ones(viewZn.shape)
    relAz  =  viewAz - solAz 

    nodataMask = metadata['to-sensor_Zenith_Angle'][:,:] == -9999
    viewZn[nodataMask] = -9999
    viewAz[nodataMask] = -9999
    relAz[nodataMask] = -9999
    
    return solZn,viewZn,relAz


def brdf_kernel(solZn,viewZn,relAz, ross = 'thick', li = 'dense'):
    '''Calculate the geometric and volume scattering coeffiecients. 
    
    Parameters
    ----------
    solZn : Solar zenith angle in radians
    viewZn :  Viewing zenith angle in radians (signed?)
    relAz : Relative angle between sun and view azimuth 
    ross : Volume scattering kernel type
    li : Geometric scattering kernel type
    
    Returns
    -------
    Volume and geomtric scattering kernels as m x n numpy array
    '''
    
    #Ross kernels 
    ############

    # Eq 2. Schlapfer et al. IEEE-TGARS 2015
    phase = np.arccos(np.cos(solZn)*np.cos(viewZn) + np.sin(solZn)*np.sin(viewZn)*  np.cos(relAz))
    
    if ross == 'thick':
        # Eq 13. Wanner et al. JGRA 1995
        kVol = ((pi/2 - phase)*np.cos(phase) + np.sin(phase))/ (np.cos(viewZn)*np.cos(solZn)) - pi/4
    elif ross == 'thin':
        # Eq 13. Wanner et al. JGRA 1995
        kVol = ((pi/2 - phase)*np.cos(phase) + np.sin(phase))/ (np.cos(viewZn)*np.cos(solZn)) - pi/2

    #Li kernels
    ############
    # Constants used from Colgan  et al. RS 2012 
    
    # Eq. 37,52. Wanner et al. JGRA 1995
    solZn_ = np.arctan(10* np.tan(solZn))
    viewZn_ = np.arctan(10* np.tan(viewZn))
    # Eq 50. Wanner et al. JGRA 1995
    D = np.sqrt((np.tan(solZn_)**2) + (np.tan(viewZn_)**2) - 2*np.tan(solZn_)*np.tan(viewZn_)*np.cos(relAz))    
    # Eq 49. Wanner et al. JGRA 1995
    t_num = 2. * np.sqrt(D**2 + (np.tan(solZn_)*np.tan(viewZn_)*np.sin(relAz))**2) 
    t_denom = (1/np.cos(solZn_))  + (1/np.cos(viewZn_))
    t = np.arccos(np.clip(t_num/t_denom,-1,1))
    # Eq 33,48. Wanner et al. JGRA 1995
    O = (1/pi) * (t - np.sin(t)*np.cos(t)) * t_denom
    # Eq 51. Wanner et al. JGRA 1995
    cosPhase_ =  np.cos(solZn_)*np.cos(viewZn_) + np.sin(solZn_)*np.sin(viewZn_)*np.cos(relAz)

    if li == 'sparse':
        # Eq 32. Wanner et al. JGRA 1995
        kGeo = O - (1/np.cos(solZn_)) - (1/np.cos(viewZn_)) + .5*(1+ cosPhase_) * (1/np.cos(viewZn_))
    elif li == 'dense':
        # Eq 47. Wanner et al. JGRA 1995
        kGeo = (((1+cosPhase_) * (1/np.cos(viewZn_)))/ (t_denom - O)) - 2
    
    kGeo[viewZn == -9999] = -9999
    kVol[viewZn == -9999] = -9999

    return kVol,kGeo


def write_kernel(srcFile):
    """
    Write volume and geometric scattering kernel to HDF.
        
    Parameters
    ----------
    srcFile : Pathname of HDF file
    
    Returns
    -------
    newName : New file name    
    
    TODO : add ability to calculate scattering class specific kernels using landcover mask
    """

    objectHDF = h5py.File(srcFile,'r+')
    metadata = objectHDF[objectHDF.keys()[0]]['Reflectance']['Metadata']
    
    # Get solar and viewing geometry
    solZn, viewZn, relAz = geom_sol_view(metadata)
    # Calculate kernels
    kVol,kGeo = brdf_kernel(solZn,viewZn,relAz,ross = 'thick',li='dense')

    # Calculate kernels at nadir viewing zenith and azimuth
    relAzNadir = np.zeros(viewZn.shape)-np.radians(metadata['Logs']['Solar_Azimuth_Angle'].value)
    kVolNadir,kGeoNadir = brdf_kernel(solZn,np.zeros(viewZn.shape),relAzNadir,ross = 'thick',li='dense')

    # Write kernels to hdf file
    ancData = objectHDF[objectHDF.keys()[0]]["Reflectance"]["Metadata"]['Ancillary_Imagery']
    ancData.create_dataset('kVol', data = kVol)
    ancData.create_dataset('kGeo', data = kGeo)
    ancData.create_dataset('kVol_nadi', data = kVolNadir)
    ancData.create_dataset('kGeo_nadir', data = kGeoNadir)

    objectHDF.close()
    
    # Rename file
    newName = '%s/%s_knl.h5' % (os.getcwd(),os.path.splitext(os.path.basename(srcFile))[0])
    
    os.rename(srcFile, newName)
    
    return newName


