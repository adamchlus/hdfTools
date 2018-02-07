import os,shutil,glob,subprocess,time
import numpy as np, h5py
from .helpers import *


"""
This modules contains functions for resampling spectra given source and
destinations wavelengths and full width half maximum.
"""

def gaussian(x,mu,fwhm):
    """Return a gaussian distribution.
    
    Parameters
    ----------
    x : Numpy array of values along which to generate gaussian.    
    mu : Mean of the gaussian function.
    fwhm : Full width half maximum.

    Returns
    -------
    Numpy array of gaussian along input range.
    """
    
    c = fwhm/(2* np.sqrt(2*np.log(2)))
    return np.exp(-1*((x-mu)**2/(2*c**2)))
    
def resample_coeff(srcWaves,srcFWHMs,dstWaves,dstFWHMs, spacing = 1):
    """ Return a set of coeffiencients for spectrum resampling
    
    Given a set of source and destination wavelengths and FWHMs this
    function caculates the relative contribution or each input wavelength
    to the output wavelength. It assumes that both input and output 
    response functions follow a gaussian distribution.
    
    Parameters
    ----------
    srcWaves : List of source wavelength centers.    
    srcFWHMs : List of source full width half maxes.
    dstWaves : List of destination wavelength centers.    
    dstFWHMs : List of destination full width half maxes.
    spacing : resolution at which to model the spectral resposnse functions

    Returns
    -------
    m x n matrix of coeffiecients, where m is the number of source wavelengths
    and n is the number of destination wavelengths.    
    """

    # For each destination band calculate the relative contribution
    # of each wavelength to the band response at 1nm resolution
    dstMatrix = []
    oneNM = np.arange(280,2600)
    for dstWave,dstFWHM in zip(dstWaves,dstFWHMs):
        a = gaussian(oneNM -.5,dstWave,dstFWHM)
        b = gaussian(oneNM +.5,dstWave,dstFWHM) 
        areas = (a +b)/2
        dstMatrix.append(np.divide(areas,np.sum(areas)))
    dstMatrix = np.array(dstMatrix)

    # For each source wavelength generate the gaussion response
    # function at 1nm resolution
    srcMatrix = []
    for srcWave,srcFWHM in zip(srcWaves,srcFWHMs):
        srcMatrix.append(gaussian(oneNM ,srcWave,srcFWHM))
    srcMatrix = np.array(srcMatrix)
   
    # Calculate the relative contribution of each source response function
    # TODO : rename variables
    ratio =  srcMatrix/srcMatrix.sum(axis=0)
    ratio[np.isnan(ratio)] = 0
    ratio2 = np.einsum('ab,cb->acb',ratio,dstMatrix)
    
    # Calculate the relative contribution of each input wavelength
    # to each destination wavelength
    coeffs = np.trapz(ratio2)

    return coeffs

def apply_resample(srcFile,dstFile,dstWaves,dstFWHMs):
    """Apply spectral resampling coeffiecients to an image.
    
    Parameters
    ----------
    srcFile : Pathname of input HDF file to to be resampled.
    dstFile : Pathname for resampled HDF file.
    dstWaves : List of output wavelength centers.
    dstFWHMs : List of output full width half maxes.
    
    Returns
    -------
    None
    
    """
    # Make a temp copy of the input data file 
    tmpFile = "%s/temp.h5" %  os.path.split(dstFile)[0]
        
    # Get image metadata dictionary
    srcMetaDict =get_metadata(srcFile)
    
    # Open HDF files
    srcHDF = h5py.File(srcFile,'r')
    tmpHDF = h5py.File(tmpFile,'r+')
    
    # Get source and destination fwhms and wavelengths
    srcFWHMs = srcMetaDict['fwhm']
    srcWaves = srcMetaDict['wavelengths']
    
    coeffs = resample_coeff(srcWaves,srcFWHMs,dstWaves,dstFWHMs)
    
    # Load source data and get chunks dimensions
    srcData = srcHDF[srcHDF.keys()[0]]['Reflectance']['Reflectance_Data']
    yChunk,xChunk,bChunk = srcData.chunks

    # Delete old  reflectance dataset   
    baseKey = tmpHDF.keys()[0]
    del tmpHDF['/%s/Reflectance/Reflectance_Data' % baseKey]
    
    # Create new reflectance dataset in destination file
    dstData = tmpHDF.create_dataset('/%s/Reflectance/Reflectance_Data' % baseKey, 
                                    (srcMetaDict['rows'],  srcMetaDict['columns'],len(dstWaves)), 
                                    chunks= (yChunk,xChunk,bChunk))

    # Resample data chunkwise
    for y in range(srcMetaDict['rows']/yChunk+1):
        perc = round(float(y)/(srcMetaDict['rows']/yChunk+1),2)
        print '%s complete    %s ' % (perc,time.ctime()) 
        yStart = y*yChunk
        yEnd = (y+1)*yChunk      
        if yEnd >= srcMetaDict['rows']:
            yEnd = srcMetaDict['rows'] 
        
        for x in range(srcMetaDict['columns']/xChunk+1):
            xStart = x*xChunk
            xEnd = (x+1)*xChunk
            if xEnd >= srcMetaDict['columns']:
                xEnd = srcMetaDict['columns'] 
                
            srcDataArr =srcData[yStart:yEnd,xStart:xEnd,:]
            dstData[yStart:yEnd,xStart:xEnd,:] = np.einsum('rcb,bn->rcn',srcDataArr,coeffs).astype(np.int16)    

    # Update wavelengths and FWHMs in resampled HDF file
    del tmpHDF['/%s/Reflectance/Metadata/Spectral_Data/FWHM' % baseKey]
    tmpHDF.create_dataset('/%s/Reflectance/Metadata/Spectral_Data/FWHM' % baseKey,
                          data = dstFWHMs)
    del tmpHDF['/%s/Reflectance/Metadata/Spectral_Data/Wavelength' % baseKey]
    tmpHDF.create_dataset('/%s/Reflectance/Metadata/Spectral_Data/Wavelength' % baseKey,
                          data = dstWaves)

    srcHDF.close()
    tmpHDF.close()
    
    #repack HDF and delete temporary file
    print "Beginning repack-- %s" % time.ctime()
    process = subprocess.Popen("h5repack -f GZIP=6 %s %s" % (tmpFile,dstFile),shell= True)
    process.wait()
    print "Repack complete-- %s" % time.ctime()








