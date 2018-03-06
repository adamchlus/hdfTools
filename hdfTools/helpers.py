import numpy as np, h5py, pandas as pd
from math import *
import time,os,gdal
from os.path import expanduser

home = expanduser("~")



def get_window(easting,northing,window,hdfFile):
    '''Retreives imagery from a window around a input set of coordinates.
    
    Returns the upper left hand coordinates of the image,the
    subset as a numpy array, and metadata object from HDF for geotiff export
    
    WARNING: Inputing too large of a window can cause a crash.
    '''
    # Use exceptions to close the hdf file in the event of an error
    try:
        
        if not os.path.isfile(hdfFile):
            print "HDF file not found"
            return 

        # Load HDF file
        metaDict =get_metadata(hdfFile)
        objectHDF = h5py.File(hdfFile,'r')
        data = objectHDF[objectHDF.keys()[0]]["Reflectance"]["Reflectance_Data"] 
 
        # Convert input coordinates into array indices
        xArray = int(easting - metaDict['ulX'])
        yArray = int(metaDict['ulY'] - northing)

        # Generate extents of the window L(eft), R(ight),T(op), B(ottom), adjust if out of image bounds
        winLx = max(xArray - window,0)
        winRx = min(xArray + window+1,metaDict['columns'])
        winTy = max(yArray - window,0)
        winBy = min(yArray + window+1, metaDict['rows'])
    
        # If the box does not intersect the image
        if any([coord <= 0 for coord in [winRx,winBy]]) or winTy> metaDict['rows'] or winLx >metaDict['columns']:
            print "No intersection"
            subset = "No intersection"
            return subset,metaDict
        else:
            # Slice out subset as float
            subset = data[winTy:winBy,winLx:winRx,:]
            subset = subset.astype(float)
        
            # Check if greater than 50% of the image is noData if not skip
            if np.sum(subset[:,:,1] == -9999)/float(subset.shape[0]*subset.shape[1]) > .50:
                subset = "No intersection"
                
        objectHDF.close()       
        
        # Update metadata
        metaDict['ulX'] = metaDict['ulX'] +winLx
        metaDict['ulY'] = metaDict['ulY']-winTy
        metaDict['winRx'] = winRx
        metaDict['winLx'] = winLx
        metaDict['winTy'] = winTy
        metaDict['winBy'] = winBy
        metaDict['rows'] =subset.shape[0] 
        metaDict['columns'] = subset.shape[1] 
        
        return subset,metaDict

    except:
        print "ERROR"
        if objectHDF:
            objectHDF.close()


def get_hspan(northing,window,hdfFile, bandList = None):
    '''Retreives imagery from a window around a input set of coordinates.
    
    Returns the upper left hand coordinates of the image,the
    subset as a numpy array, and metadata object from HDF for geotiff export
    
    WARNING: Inputing too large of a window can cause a crash.
    '''
    #Use exceptions to close the hdf file in the event of an error
    try:
        
        if not os.path.isfile(hdfFile):
            print "HDF file not found"
            return 

        # Load HDF file
        metaDict =get_metadata(hdfFile)
        objectHDF = h5py.File(hdfFile,'r')
        data = objectHDF[objectHDF.keys()[0]]["Reflectance"]["Reflectance_Data"] 

        # Convert input coordinates into array indices
        yArray = int(metaDict['ulY'] - northing)
        
        # Generate extents of the window T(op), B(ottom), adjust if out of image bounds
        winTy = max(yArray - window,0)
        winBy = min(yArray + window, metaDict['rows'])
        winLx = 0
        winRx = metaDict['columns']
            
        # Slice out subset as float
        if metaDict['bands']:
            subset = data[winTy:winBy,winLx:winRx,bandList]
        else:
            subset = data[winTy:winBy,winLx:winRx,:]

        subset = subset.astype(float)
                
        objectHDF.close()       
        
        # Update metadata dictionary
        metaDict['ulY'] = metaDict['ulY']-winTy
        metaDict['winRx'] = winRx
        metaDict['winLx'] = winLx
        metaDict['winTy'] = winTy
        metaDict['winBy'] = winBy
        metaDict['rows'] = subset.shape[0]
        metaDict['columns'] = subset.shape[1]
        metaDict['bands'] = subset.shape[2]
        
        return subset,metaDict

    except:
        print "ERROR"
        if objectHDF:
            objectHDF.close()




def get_metadata(hdfFile):
    '''Parse image metadta into a dictionary
       :param hdfFile: pathname to hdf file
       :return metaDict: dctionary with file metadata
 
    '''
    #use exceptions to close the hdf file in the event of an error
    try:
        objectHDF = h5py.File(hdfFile,'r')
        
        #metadata dictionary
        metaDict = {}
        
        # Get metadata
        data = objectHDF[objectHDF.keys()[0]]["Reflectance"]["Reflectance_Data"] 
        metadata = objectHDF[objectHDF.keys()[0]]["Reflectance"]["Metadata"]
        metaDict['coord_sys'] = metadata['Coordinate_System']['Coordinate_System_String'].value 
        metaDict['map_info'] = metadata['Coordinate_System']['Map_Info'].value.split(',')   
        metaDict['fwhm'] =  metadata['Spectral_Data']['FWHM'].value
        metaDict['wavelengths'] = metadata['Spectral_Data']['Wavelength'].value.astype(int)
        metaDict['ulX'] = float(metaDict['map_info'][3])
        metaDict['ulY'] = float(metaDict['map_info'][4])
        metaDict['rows'] = data.shape[0]
        metaDict['columns'] = data.shape[1]
        metaDict['bands'] = data.shape[2]
        
        region1 = (metaDict['wavelengths'] >=400) & (metaDict['wavelengths'] <=1330)
        region2 = (metaDict['wavelengths'] >=1430) & (metaDict['wavelengths'] <=1800)
        region3 = (metaDict['wavelengths'] >=1960) & (metaDict['wavelengths'] <=2450)

        metaDict['good_bands'] = region1| region2 | region3 
        

           
        objectHDF.close()       

        return metaDict

    except:
        print "ERROR"
        if objectHDF:
            objectHDF.close()




def get_ndi(band1,band2,srcFile):
    """
    Calculates an normalized difference index given two bands.
     
    Parameters
    ----------
    band1 : Index of first band in NDI.
    band2 : Index of second band in NDI.
    srcFile : Pathname of input HDf file.

    Returns
    -------
    ndi : masked numpy array of NDI
    metaDict : image metadata dictionary
    """

    try:
        
        if not os.path.isfile(srcFile):
            print "HDF file not found"
            return 

        # Open HDF file
        metaDict =get_metadata(srcFile)
        srcHDF = h5py.File(srcFile,'r')
        data = srcHDF[srcHDF.keys()[0]]["Reflectance"]["Reflectance_Data"] 

        # Load bands for NDI to memory
        band1Arr= data[:,:,band1]
        band1Arr= band1Arr.astype(float)
        band2Arr= data[:,:,band2]
        band2Arr= band2Arr.astype(float)

        # Calculate NDI
        ndi = (band1Arr-band2Arr)/(band1Arr+band2Arr)
        ndi[band1Arr == -9999] = -9999
        ndi = np.ma.masked_array(ndi, mask = ndi == -9999.)

        del band1Arr,band2Arr,data
        
        srcHDF.close()       
        
        return ndi,metaDict

    except:
        print "ERROR"
        if srcHDF:
            srcHDF.close()



def array_to_geotiff(array,metaDict,dstFile,datatype = gdal.GDT_Int16):
    """
    Export numpy array as geotiff.

    Parameters
    ----------
    array : Numpy array
    metaDict : Metadata dictionary for array
    dstFile : Pathname of output geotif
    datatype :  Array datatype

    Returns
    -------
    None
    
    Geotiff saved to dstFile
    
    """
    
    # Set the output raster transform and projection properties
    driver = gdal.GetDriverByName("GTIFF")
    tiff = driver.Create(dstFile,metaDict['columns'],metaDict['rows'],metaDict['bands'],gdal.GDT_Int16)
    transform = (metaDict['ulX'],float(metaDict['map_info'][1]),0,metaDict['ulY'],0,-float(metaDict['map_info'][2]))
    tiff.SetGeoTransform(transform)
    tiff.SetProjection(metaDict['coord_sys'])
        
    # Write bands to file
    for band in range(array.shape[2]):
        tiff.GetRasterBand(band +1).WriteArray(array[:,:,band])
        tiff.GetRasterBand(band +1).SetNoDataValue(-9999)
        
    del tiff, driver
    

def hdf_to_geotiff(srcFile,bandList = [27,17,7,47]):
    """
    Export HDF file as geotiff.

    Parameters
    ----------
    srcFile : Pathname of hdf file to be converted to a geotiff.
    bandList : List of band indices to be included in geotiff.

    Returns
    -------
    None
    
    Geotiff saved to same directory as input HDF.
    
    """
    
    outDIR= os.path.split(srcFile)[0]
    basename = os.path.splitext(os.path.basename(srcFile))[0]
    dstFile = "%s/%s.tif" % (outDIR,basename)

    # Load HDF file
    metaDict =get_metadata(srcFile)
    srcHDF = h5py.File(srcFile,'r')
    data = srcHDF[srcHDF.keys()[0]]["Reflectance"]["Reflectance_Data"] 
    
    # Set the output raster transform and projection properties
    driver = gdal.GetDriverByName("GTIFF")
    tiff = driver.Create(dstFile,metaDict['columns'],metaDict['rows'],len(bandList),gdal.GDT_Int16)
    transform = (metaDict['ulX'],float(metaDict['map_info'][1]),0,metaDict['ulY'],0,-float(metaDict['map_info'][2]))
    tiff.SetGeoTransform(transform)
    tiff.SetProjection(metaDict['coord_sys'])
    
    # Cycle through each band
    for i,band in enumerate(bandList):
        print "Saving band %s" % band
        tiff.GetRasterBand(i+1).WriteArray(data[:,:,band])
        tiff.GetRasterBand(i+1).SetNoDataValue(-9999)
    
    #the file is written to the disk once the driver variables are deleted
    del tiff, driver   
    
    srcHDF.close()



def apply_PLSR(srcFile,traitCoeffs):    
    """
    Apply PLSR coefficients to HDF file.

    Parameters
    ----------
    srcFile : Pathname of input file
    traitCoeffs : Pathname of CSV containgin PLSR coefficients

    Returns
    -------
    m x n x 2 numpy array af trait mean and standard deviation
    
    """
    
    # Load trait model
    traitModel = pd.read_csv(traitCoeffs,index_col = 0)
    traitModel = traitModel.T
    wavelengths = [int(x) for x in traitModel.columns if x <> "intercept"]
    wavelengthsStr = [x for x in traitModel.columns if x <> "intercept"]
    coeffs =traitModel[wavelengthsStr].values.reshape((1,500,len(wavelengths)))
    intercept = traitModel["intercept"].values.reshape(1,1,500)
    
    # Open HDF file
    srcHDF = h5py.File(srcFile,'r+')
    data = srcHDF[srcHDF.keys()[0]]["Reflectance"]["Reflectance_Data"]
    yChunk,xChunk,bChunk = data.chunks

    traitArray = np.zeros((data.shape[0],data.shape[1],2))
    
    # Create wavelength mask 
    waveMask = np.arange(data.shape[2])[[x in wavelengths for x in metaDict['wavelengths']]]
    
    # Apply PLSR mode
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
            dataArr =data[yStart:yEnd,xStart:xEnd,waveMask]
            dataArr = ma.masked_array(dataArr, mask = dataArr == -9999).astype(float)
            
            # Apply PLSR coefficients
            traits = np.einsum('jkl,mnl->jkn',dataArr,coeffs )
            traitPred =  traits+intercept
            
            traitMean =traitPred.mean(axis=2)
            traitStd =traitPred.std(axis=2,ddof=1)
          
            # Remask no data    
            traitStd[dataArr.mask[:,:,0]] = -9999
            traitMean[dataArr.mask[:,:,0]]  = -9999
    
            traitArray[yStart:yEnd,xStart:xEnd,0] = traitMean
            traitArray[yStart:yEnd,xStart:xEnd:,1] = traitStd

    srcHDF.close()
    
    return traitArray







    