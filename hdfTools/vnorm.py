import os,shutil
import pandas as pd, numpy as np, h5py
from .helpers import *

home = os.path.expanduser("~")


"""
This module contains functions to vector normalize image spectroscopy
data. 
"""


def vnorm(srcFile):
    """
    Vector normalize imaging spectroscopy data in HDF format.
    
    First the norm of each pixel is calculated using bands not affected by
    water absorption bands. The spectrum of each pixel is then divided by its
    respective norm.
    
    Parameters
    ----------
    srcFile : Pathname of image to be vector normalized  

    Returns
    -------
    None
    
    Vector normalized image saved to disk.
    """

    print srcFile
    vnormFile =  '%s/%s_vnorm.h5' % (os.getcwd(),os.path.splitext(os.path.basename(srcFile))[0])
    shutil.copyfile(srcFile, vnormFile)
    
    #get image metadata dictionary
    metaDict =get_metadata(vnormFile)
    #open HDF file
    vnormHDF = h5py.File(vnormFile,'r+')
    metadata = vnormHDF[vnormHDF.keys()[0]]['Reflectance']['Metadata']
    data = vnormHDF[vnormHDF.keys()[0]]["Reflectance"]["Reflectance_Data"]
    
    yChunk,xChunk,bChunk = data.chunks

    #Vector normalize data chunkwise
    for y in range(metaDict['rows']/yChunk+1):
        perc = round(float(y)/(metaDict['rows']/yChunk+1),2)
        print "%s:  %s complete" % (os.path.basename(vnormFile),perc) 
        yStart = y*yChunk
        yEnd = (y+1)*yChunk      
        if yEnd >= metaDict['rows']:
            yEnd = metaDict['rows'] 
        
        for x in range(metaDict['columns']/xChunk+1):
            xStart = x*xChunk
            xEnd = (x+1)*xChunk
            if xEnd >= metaDict['columns']:
                xEnd = metaDict['columns'] 
            dataArr =data[yStart:yEnd,xStart:xEnd,:]

            # Calculate norm and divide out of spectrum
            norm = np.expand_dims(np.linalg.norm(dataArr[:,:,metaDict['good_bands']],axis=2),axis=2)
            vnorm = 100000*dataArr/norm
            vnorm[dataArr == -9999] = -9999
            data[yStart:yEnd,xStart:xEnd,:] = vnorm.astype(np.int16)
    
    vnormHDF.close()
    


    
    
    
