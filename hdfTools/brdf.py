import pandas as pd, numpy as np,os,h5py,shutil
from os.path import expanduser
home = expanduser('~')
import numpy.ma as ma
from .helpers import *

"""
This module contains functions to apply a modified version of the the BRDF correction
described in the following papers:

Colgan, M. S., Baldeck, C. A., Feret, J. B., & Asner, G. P. (2012). 
Mapping savanna tree species at ecosystem scales using support vector machine classification 
and BRDF correction on airborne hyperspectral and LiDAR data. 
Remote Sensing, 4(11), 3462-3480.        

Collings, S., Caccetta, P., Campbell, N., & Wu, X. (2010). 
Techniques for BRDF correction of hyperspectral mosaics. 
IEEE Transactions on Geoscience and Remote Sensing, 48(10), 3733-3746.
    
Schlapfer, D., Richter, R., & Feingersh, T. (2015). 
Operational BRDF effects correction for wide-field-of-view optical scanners (BREFCOR). 
IEEE Transactions on Geoscience and Remote Sensing, 53(4), 1855-1864.

Wanner, W., Li, X., & Strahler, A. H. (1995). 
On the derivation of kernels for kernel-driven models of bidirectional reflectance. 
Journal of Geophysical Research: Atmospheres, 100(D10), 21077-21089.

Weyermann, J., Kneubuhler, M., Schlapfer, D., & Schaepman, M. E. (2015). 
Minimizing Reflectance Anisotropy Effects in Airborne Spectroscopy Data Using Ross-Li Model Inversion 
With Continuous Field Land Cover Stratification. 
IEEE Transactions on Geoscience and Remote Sensing, 53(11), 5814-5823.

BRDF correction consists of the following steps:
    
    1. Stratified random sampling of the image(s) based on predefined scattering classes
    2. Regression modeling per scattering class, per wavelength
            reflectance = fIso + fVol*kVol +  fGeo*kGeo
            (eq 2. Weyermann et al. IEEE-TGARS 2015)
    3. Adjust reflectance using a multiplicative correction per class, per wavelength. 
            (eq 5. Weyermann et al. IEEE-TGARS 2015)
"""

def sample_hdf(srcFile, sampPerc = 0.1):
    """
    Randomly sample image and output samples to file.

    """
    # Open HDF file
    objectHDF = h5py.File(srcFile,'r')
    metadata = objectHDF[objectHDF.keys()[0]]['Reflectance']['Metadata']
    metaDict =get_metadata(srcFile)

    # Load reflectance data object and other needed layers
    data = objectHDF[objectHDF.keys()[0]]["Reflectance"]["Reflectance_Data"]
    kVol = metadata['Ancillary_Imagery']['kVol'][:,:]
    kGeo = metadata['Ancillary_Imagery']['kGeo'][:,:]
    mask = metadata['Ancillary_Imagery']['Landcover'][:,:]
    mask = np.ma.masked_array(mask, mask = mask == -9999.)
    
    scattClasses = [1]
    
    #sample array to sample from
    sampleArray = np.zeros(mask.shape)
    
    # Cycle through scatter class ranges and generate sampling map
    for scatter in scattClasses:
        idx = np.array(np.where((mask.mask == False) & (mask == scatter))).T
        idxRand= idx[np.random.choice(range(len(idx)),int(len(idx)*sampPerc), replace = False)].T
        sampleArray[idxRand[0],idxRand[1]] = scatter

    yChunk,xChunk,bChunk = data.chunks

    # Sample data chunkwise
    for y in range(metaDict['rows']/yChunk+1):
        samples= []
        perc = round(float(y)/(metaDict['rows']/yChunk+1),2)
        print '%s complete    %s ' % (perc,time.ctime()) 
        yStart = y*yChunk
        yEnd = (y+1)*yChunk      
        if yEnd >= metaDict['rows']:
            yEnd = metaDict['rows'] 
        
        for x in range(metaDict['columns']/xChunk+1):
            xStart = x*xChunk
            xEnd = (x+1)*xChunk
            if xEnd >= metaDict['columns']:
                xEnd = metaDict['columns'] 
            
            # Load reflectance data and sampling map
            dataArr =data[yStart:yEnd,xStart:xEnd,:]
            sampleChunk = sampleArray[yStart:yEnd,xStart:xEnd]
            
            # Sample data array and append to sample list
            for scatter in scattClasses:
                scatterRand = sampleChunk == scatter
                if scatterRand.sum() >0:
                    kVolRand = np.expand_dims(kVol[yStart:yEnd,xStart:xEnd][scatterRand],axis=1)
                    kGeoRand = np.expand_dims(kGeo[yStart:yEnd,xStart:xEnd][scatterRand],axis=1)                
                    randData = dataArr[scatterRand]
                    samples.append(np.concatenate([randData,kVolRand,kGeoRand,np.ones(kGeoRand.shape)*scatter],axis=1))
        # Write sampled data to file
        if len(samples)> 0:
            prefix = os.path.splitext(os.path.basename(srcFile))[0]
            bin_writer(np.concatenate(samples),metaDict,prefix)

    del data,kVol,kGeo
    objectHDF.close()


def bin_writer(samples,metaDict, prefix = ''):
    """
    Write samples to binary file.
    
    """
   
    files = metaDict['wavelengths'].tolist() + ['kVol','kGeo','class']

    # Write data to individual files
    for index,filename in enumerate(files):
        pathname = '%s/%s_%s.bin' % (os.getcwd(),prefix,filename)
        if os.path.isfile(pathname):
            wrMode = "a+"
        else:
            wrMode = "w+"
        
        with open(pathname, wrMode) as f:
            # Jump to end of the file
            f.seek(0,2)
            samples[:,index].tofile(f)
            f.close()


def apply_brdf(srcFile,brdfCoeffDF):    

    # Open HDF file
    srcHDF = h5py.File(srcFile,'r+')
    metadata = srcHDF[srcHDF.keys()[0]]['Reflectance']['Metadata']
    data = srcHDF[srcHDF.keys()[0]]["Reflectance"]["Reflectance_Data"]
    metaDict =get_metadata(srcFile)

    # Load kernels
    kVol = metadata['Ancillary_Imagery']['kVol'][:,:]
    kGeo = metadata['Ancillary_Imagery']['kGeo'][:,:]
    kVolNadir = metadata['Ancillary_Imagery']['kVol_nadi'][:,:]
    kGeoNadir = metadata['Ancillary_Imagery']['kGeo_nadir'][:,:]

    yChunk,xChunk,bChunk = data.chunks

    # Apply correction chunkwise
    for y in range(metaDict['rows']/yChunk+1):
        perc = round(float(y)/(metaDict['rows']/yChunk+1),2)
        print "%s:  %s complete" % (os.path.basename(srcFile),perc) 
        yStart = y*yChunk
        yEnd = (y+1)*yChunk      
        if yEnd >= metaDict['rows']:
            yEnd = metaDict['rows'] 
        
        for x in range(metaDict['columns']/xChunk+1):
            xStart = x*xChunk
            xEnd = (x+1)*xChunk
            if xEnd >= metaDict['columns']:
                xEnd = metaDict['columns'] 
                
            # Slice and mask chunk    
            dataArr =data[yStart:yEnd,xStart:xEnd,:]
            dataArr = ma.masked_array(dataArr, mask = dataArr == -9999).astype(float)
            
            # Get scattering kernel for chunks
            kVolNadir_chunk = kVolNadir[y*yChunk:(y+1)*yChunk,x*xChunk:(x+1)*xChunk]
            kGeoNadir_chunk = kGeoNadir[y*yChunk:(y+1)*yChunk,x*xChunk:(x+1)*xChunk]
            kVol_chunk = kVol[y*yChunk:(y+1)*yChunk,x*xChunk:(x+1)*xChunk]
            kGeo_chunk = kGeo[y*yChunk:(y+1)*yChunk,x*xChunk:(x+1)*xChunk]

            # Apply brdf correction 
            # eq 5. Weyermann et al. IEEE-TGARS 2015)
            brdf = np.einsum('i,jk-> jki', brdfCoeffDF.kVol,kVol_chunk) + np.einsum('i,jk-> jki', brdfCoeffDF.kGeo,kGeo_chunk)  + brdfCoeffDF.kIso.values
            brdfNadir = np.einsum('i,jk-> jki', brdfCoeffDF.kVol,kVolNadir_chunk) + np.einsum('i,jk-> jki', brdfCoeffDF.kGeo,kGeoNadir_chunk)  + brdfCoeffDF.kIso.values
            correctionFactor = brdfNadir/brdf
            dataArr*= correctionFactor
            dataArr.data[dataArr.mask] = -9999
            
            # Write corrected chunk to file
            data[y*yChunk:(y+1)*yChunk,x*xChunk:(x+1)*xChunk,:] = dataArr.astype(np.int16)

    # Write BRDF coefficients to hdfile
    ancData = srcHDF[srcHDF.keys()[0]]["Reflectance"]["Metadata"]['Ancillary_Imagery']
    ancData.create_dataset("BRDF_coefficients", data = brdfCoeffDF.as_matrix())
    ancData['BRDF_coefficients'].attrs['column_names'] = brdfCoeffDF.columns.values.tolist()

    del correctionFactor,dataArr,kVol,kGeo,kVolNadir,kGeoNadir
    srcHDF.close()
    
    # Rename file
    brdfName = '%s/%s_brdf.h5' % (os.getcwd(),os.path.splitext(os.path.basename(srcFile))[0])
    os.rename(srcFile, brdfName)
    
    return brdfName
    