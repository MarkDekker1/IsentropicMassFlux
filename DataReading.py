#Packages
from netCDF4 import Dataset
from pylab import * 

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import scipy

#%%Data inladen (verschilt per computer!)
file = 'C:\Users\Rob\Documents\Localstudy\BoundaryLayers\2011_2016.nc'

#%%
ncdf = Dataset(file, mode='r')

#%%Uithalen variabelen
Lat = ncdf.variables['Latitude'][:]
Lon = ncdf.variables['Longitude'][:]
Temp = ncdf.variables['t'][:]
u = ncdf.variables['u'][:]