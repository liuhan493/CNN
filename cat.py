from astropy.io import fits


hdul = fits.open("SpecViewer-master/fits/spec-3668-55478-0484.fits")
print(hdul[1].header)