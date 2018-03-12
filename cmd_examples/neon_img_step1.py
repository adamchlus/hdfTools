import os,shutil,sys,subprocess
import numpy as np, h5py
from hdfTools.resampler import *
from hdfTools.helpers import *
from hdfTools.brdf import *
from hdfTools.kernel import *

"""
This script runs the first step in the 3 step processing stream for NEON imagery.

In the first step there are 4 subprocesses:
    1. Resample imagery to XX nm resolution.
    2. Generate a landcover mask to seperate scattering classes.
    3. Calculate scattering kernels.
    4. Randomly sample imagery and output samples to file.

"""

def main(srcFile):
    """
    srcFile : Filename of input HDF file, assumes script is run from
                same directory
    
    """

    # Substep 1
    #####################
    print "Commencing spectral resampling    %s" % time.ctime()
    dstWaves = np.arange(390,2505,5)
    dstFWHMs = np.ones(dstWaves.shape) * 5  
    srcFile =  '%s/%s' % (os.getcwd(),srcFile)
    resampFile = '%s/%s.h5' % (os.path.dirname(srcFile),
                         os.path.splitext(os.path.basename(srcFile.replace('reflectance','refl_5nm')))[0])
    apply_resample(srcFile,resampFile,dstWaves,dstFWHMs)
    
    # Substep 2
    #####################
    # Generate landcover mask using an NDVI threshold and rename file
    print "Commencing landcover classification    %s" % time.ctime()

    mask,metadata = get_ndi(95,55,resampFile)
    mask[mask > .1] = 1
    mask[(mask <= .1) & (mask >= -1)] = 0
    objectHDF = h5py.File(resampFile,'r+')
    ancData = objectHDF[objectHDF.keys()[0]]["Reflectance"]["Metadata"]['Ancillary_Imagery']
    ancData.create_dataset("Landcover", data = mask.data)
    objectHDF.close()
    
    lcFile = '%s/%s_lc.h5' % (os.getcwd(),os.path.splitext(os.path.basename(resampFile))[0])
    os.rename(resampFile, lcFile)
    
    # Substep 3
    #####################
    print "Commencing kerenl generation    %s" % time.ctime()

    knlFile = write_kernel(lcFile)
    
    # Substep 4
    #####################
    print "Commencing sampling    %s" % time.ctime()

    sample_hdf(knlFile, sampPerc = 0.1)

	  
    print "Step 1 complete   %s" % time.ctime()

if __name__ == "__main__":
   main(sys.argv[1])




