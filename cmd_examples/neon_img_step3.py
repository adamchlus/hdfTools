import os,shutil,sys,time
import pandas as pd
import numpy as np, h5py
from hdfTools.resampler import *
from hdfTools.helpers import *
from hdfTools.brdf import *
from hdfTools.vnorm import *

"""
This script runs the last step in the 3 step processing stream for NEON imagery.

In the first step there are 4 subprocesses:
    1. Apply BRDF correction to imagery.
    2. Vector normalize BRDF corrected data.

"""

def main(srcFile):
    """
    srcFile : Filename of input HDF file, assumes script is run from
                same directory
    
    """

    # Substep 1
    #####################
    print "Beginning BRDF correction    %s" % time.ctime()
    
    brdfCoeffs = '/mnt/gluster/chlus/brdf_coeffs/%s_brdf_coeffs.csv' % srcFile[:26]
    brdfCoeffDF= pd.read_csv(brdfCoeffs,index_col=0)
    srcFile =  '%s/%s' % (os.getcwd(),srcFile)
    brdfFile =  apply_brdf(srcFile,brdfCoeffDF)
    
    # Substep 2
    #####################
    print "BRDF correction, beginning vector normalization    %s" % time.ctime()

    vnorm(brdfFile)

    print "Vector normalization complete    %s" % time.ctime()

	  
if __name__ == "__main__":
   main(sys.argv[1])

