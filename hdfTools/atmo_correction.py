import os,shutil,glob,subprocess,time
import numpy as np, h5py
from .helpers import *


"""
This modules contains functions to export the following files needed for atmospheric correction using ATCOR from HDF
radiance files.

    1. Radiance image (ENVI, BSQ, float32)
    2. DEM Files
        a. Elevation
        b. Slope
        c. Aspect
        d. Skyview
    3. Central wavelength and FWHM (.txt)
    4. 




"""

