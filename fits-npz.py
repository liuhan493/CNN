import numpy
from astropy.io import fits
import numpy as np
import sys
import os
import math
path = "SpecViewer-master/fits/"  # 文件夹目录
files = os.listdir(path)      # 得到文件夹下的所有文件名称



for file in files:
    location = os.path.join(path, file)
    fitsfile = fits.open(location)

    flux = np.array(fitsfile[1].data["FLUX"])
    wav = 10 ** np.array(fitsfile[1].data["LOGLAM"])

    error = np.array(fitsfile[1].data["IVAR"])
    error = np.sqrt(1/error)

    print(error)
    z = np.array(fitsfile[2].data["Z"])
    location = str(location)
    location = location[-25:-5]
    paths = "SpecViewer-master/npz/"+ location + ".npz"
    paths = str(paths)
    print(paths)
    np.savez(paths,wav=wav,flux=flux,err=error,z=z)
