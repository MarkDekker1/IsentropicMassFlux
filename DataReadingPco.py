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
file = 'C:\Users\Rob\Documents\Localstudy\BoundaryLayers\DataIsentropicMassFlux.nc'
ncdf = Dataset(file, mode='r')

#%%Uithalen variabelen
Lat = ncdf.variables['latitude'][:]
Lon = ncdf.variables['longitude'][:]
Time = ncdf.variables['time'][:]
Levels = ncdf.variables['level'][:]
Levels=Levels[::-1]

#%%
v = ncdf.variables['v'][Tpoint][Ppoint][:][:]
Temp = ncdf.variables['t'][Tpoint][Ppoint[:][:]

#%%Calculations
Tpoint=1
Ppoint=3
Pref = 1000.
Rcp = 0.286
Theta = Temp*(Pref/Ppoint)**Rcp
g=9.81

def Tempf(plev,tlev,latlev):
    return np.mean(ncdf.variables['t'][tlev][36-plev][latlev][:])

def Theta(plev,tlev,latlev):
    T = Tempf(plev,tlev,latlev)
    return T*(Pref/Levels[plev])**Rcp

def Sigma(plev,tlev,latlev):
    DeltaTh = Theta(plev+1,tlev,latlev)-Theta(plev-1,tlev,latlev)
    Deltap = Levels[plev+1]-Levels[plev-1]
    return -1./g * Deltap/DeltaTh
    
def Heatflux(plev,tlev,latlev):
    V = np.mean(ncdf.variables['v'][tlev][36-plev][latlev][:])
    return V*Sigma(plev,tlev,latlev)


#%% Create vectors
Supermatrix=[]
for t in range(0,3):
    HeatFluxVec=[]
    for i in range(0,91):
        HeatFluxVec.append(Heatflux(2,t,i))
    Supermatrix.append(HeatFluxVec)
    
#%% Plotting Horizontally
Lat_0=np.mean(Lat)
Lon_0=np.mean(Lon)

#Actual plot
plt.figure(num=None, figsize=(10,6),dpi=150, facecolor='w', edgecolor='k')

m = Basemap(llcrnrlon=-180,llcrnrlat=0,urcrnrlon=180,urcrnrlat=90,
            resolution='l',projection='cyl',
            lat_ts=40,lat_0=Lat_0,lon_0=Lon_0)
												
xi, yi = m(Lon, Lat)
xi, yi = np.meshgrid(xi,yi)

Colors = m.contourf(xi,yi,Heatflux(15,1),150,cmap=plt.cm.jet)

#m.fillcontinents(color='grey')
m.drawparallels(np.arange(-90., 91., 30), labels=[1,0,0,0], fontsize=15)
m.drawmeridians(np.arange(-180., 180., 60), labels=[0,0,0,1], fontsize=15)

m.drawcoastlines()
cbar = m.colorbar(Colors, location='bottom', pad="30%",extend='both')
cbar.ax.tick_params(labelsize=15) 
#plt.clim([-5,5])
plt.show()