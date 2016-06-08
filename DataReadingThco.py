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
file = 'C:\Users\Rob\Documents\Localstudy\BoundaryLayers\DataIsentropicMassFluxTh.nc'
ncdf = Dataset(file, mode='r')

#%%Uithalen variabelen
Lat = ncdf.variables['latitude'][:]
Lon = ncdf.variables['longitude'][:]
Time = ncdf.variables['time'][:]
Levels = ncdf.variables['level'][:]
#Levels=Levels[::-1]

#%%Calculations
Tpoint=1
Ppoint=3
Pref = 1000.
Rcp = 0.286
#Theta = Temp*(Pref/Ppoint)**Rcp
g=9.81

def Pres(Thlev,tlev,latlev,longlev):
    return ncdf.variables['pres'][tlev][Thlev][latlev][longlev]

def ZonalmeanV(Thlev,tlev,latlev):
    return np.mean(ncdf.variables['v'][tlev][Thlev][latlev][:])
    
def DeviationV(Thlev,tlev,latlev,longlev):
    return ncdf.variables['v'][tlev][Thlev][latlev][longlev]-ZonalmeanV(Thlev,tlev,latlev)

def Sigma(Thlev,tlev,latlev,longlev):
    DeltaTh = Levels[Thlev+1]-Levels[Thlev-1]
    Deltap = Pres(Thlev+1,tlev,latlev,longlev)-Pres(Thlev-1,tlev,latlev,longlev)
    return -1./g * Deltap/DeltaTh

def ZonalmeanSigma(Thlev,tlev,latlev):
    M=0.
    for i in range(0,360):
        M=M+Sigma(Thlev,tlev,latlev,i)
    M=M/360.
    return M
    
def DeviationSigma(Thlev,tlev,latlev,longlev):
    return Sigma(Thlev,tlev,latlev,longlev)-ZonalmeanSigma(Thlev,tlev,latlev)
        
def MeanHeatflux(Thlev,tlev,latlev):
    return ZonalmeanV(Thlev,tlev,latlev)*ZonalmeanSigma(Thlev,tlev,latlev)
    
def EddyHeatflux(Thlev,tlev,latlev,longlev):
    return DeviationV(Thlev,tlev,latlev,longlev)*DeviationSigma(Thlev,tlev,latlev,longlev)

#%% Vector
Meanvec=np.zeros(91)
for i in range(0,91):
    Meanvec[i]=MeanHeatflux(2,2,i)
#%%
Eddyvec=np.zeros(shape=(91,360))
for i in range(0,360):
    for j in range(0,91):
        Eddyvec[i]=EddyHeatflux(2,2,j,i)
#%% Create vectors for zonal average
Supermatrix=[]
for t in range(0,3):
    HeatFluxVec=[]
    for i in range(0,91):
        ZonalMeanHeatflux.append(Heatflux(2,t,i))
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