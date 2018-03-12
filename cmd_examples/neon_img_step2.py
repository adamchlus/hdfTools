import os,shutil,glob
import numpy as np
import pandas as pd

"""
This script runs the second step in the 3 step processing stream for NEON imagery.

For the given image dataset it runs develops a regression model for each wavelength
and output the results as a csv.

Regression modeling per scattering class, per wavelength
            reflectance = fIso + fVol*kVol +  fGeo*kGeo
            (eq 2. Weyermann et al. IEEE-TGARS 2015)

Weyermann, J., Kneubuhler, M., Schlapfer, D., & Schaepman, M. E. (2015). 
Minimizing Reflectance Anisotropy Effects in Airborne Spectroscopy Data Using Ross-Li Model Inversion 
With Continuous Field Land Cover Stratification. 
IEEE Transactions on Geoscience and Remote Sensing, 53(11), 5814-5823.


"""

# Load volume and geometry kernel samples
kVol = []
kVolFiles =  glob.glob('%s/*kVol.bin' % os.getcwd())
kVolFiles.sort()
print "Loading %s kVol files" % len(kVolFiles)


for kVolFile in kVolFiles:
    kVol += np.fromfile(kVolFile).tolist()

    
kGeo =  []
kGeoFiles =  glob.glob('%s/*kGeo.bin' % os.getcwd())
kGeoFiles.sort()
print "Loading %s kGeo files" % len(kGeoFiles)

for kGeoFile in  kGeoFiles:
    kGeo += np.fromfile(kGeoFile).tolist()

# Reshape for regression
kVol = np.expand_dims(kVol,axis=1)
kGeo = np.expand_dims(kGeo,axis=1)

X = np.concatenate([kVol,kGeo,np.ones(kGeo.shape)],axis=1)

brdfCoeffs = {}

# Build per wavelength regression models
for wavelength in range(390,2501,5):
    print "Building model for %s nm" % wavelength
    y = []    
    waveFiles =  glob.glob('%s/*_%s.bin' % (os.getcwd(),wavelength))
    waveFiles.sort()
    for waveFile in  waveFiles:
        y += np.fromfile(waveFile).tolist()
    y= np.expand_dims(y,axis=1)
    brdfCoeffs[wavelength] = np.linalg.lstsq(X, y)[0].flatten()  

print "Model building complete, saving to disk."

# Save coefficients to CSV
brdfDf =  pd.DataFrame.from_dict(brdfCoeffs).T
brdfDf.columns = ['kVol','kGeo','kIso']
brdfDf.to_csv('%s/%s_brdf_coeffs.csv' % (os.getcwd(),os.path.basename(kVolFile)[:26]))








