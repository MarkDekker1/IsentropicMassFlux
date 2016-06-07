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
Theta = Temp*(Pref/Ppoint)**Rcp
g=9.81

def Pres(Thlev,tlev):
    return ncdf.variables['p'][tlev][Thlev][:][:]
    
def ZonalMeanPres(Thlev,tlev,latlev):
    return np.mean(ncdf.variables['p'][tlev][Thlev][latlev][:])

def Sigma(Thlev,tlev):
    DeltaTh = Levels[Thlev+1]-Levels[Thlev-1]
    Deltap = Pres(Thlev+1,tlev)-Pres(Thlev-1,tlev)
    return -1./g * Deltap/DeltaTh

def V(Thlev,tlev):
    return ncdf.variables['v'][tlev][36-plev][:][:]
    
def ZonalmeanV(Thlev,tlev,latlev):
    return np.mean(ncdf.variables['v'][tlev][Thlev][latlev][:])
    
def DeviationV(Thlev,tlev,latlev,longlev):
    return ncdf.variables['v'][tlev][Thlev][latlev][longlev]-ZonalmeanV(Thlev,tlev,latlev)
    
#def TemporalmeanV(Thlev,latlev,longlev):
#    M=ncdf.variables['v'][0][Thlev][latlev][longlev]
#    for i in range(1,180):
#        M=M+ncdf.variables['v'][i][Thlev][latlev][longlev]
#    M=M/180.
#    return M

#def DeviationV(tlev,Thlev,latlev,longlev):
#    return ncdf.variables['v'][tlev][Thlev][latlev][longlev]-TemporalmeanV(Thlev,latlev,longlev)    
    
#def TemporalmeanSigma(Thlev,latlev,longlev):
#    M=ncdf.variables['v'][0][Thlev][latlev][longlev]
#    for i in range(1,180):
#        M=M+ncdf.variables['v'][i][Thlev][latlev][longlev]
#    M=M/180.
#    return M

def DeviationSigma(tlev,Thlev,latlev,longlev):
    return ncdf.variables['v'][tlev][Thlev][latlev][longlev]-TemporalmeanV(Thlev,latlev,longlev)    
    
def ZonalMeanSigma(Thlev,tlev,latlev):
    DeltaTh = Levels[Thlev+1]-Levels[Thlev-1]
    Deltap = ZonalMeanPres(Thlev+1,tlev,latlev)-ZonalMeanPres(Thlev-1,tlev,latlev)
    return -1./g * Deltap/DeltaTh

def DeviationsSigma(Thlev,tlev,latlev):
    return Sigma(Thlev,tlev)
        
def Heatflux(Thlev,tlev):
    V = ncdf.variables['v'][tlev][36-plev][:][:]
    return V*Sigma(plev,tlev)
    
def ZonalMeanHeatflux(Thlev,tlev,latlev):
    V = np.mean(ncdf.variables['v'][tlev][36-plev][latlev][:])
    return V*ZonalMeanSigma(plev,tlev,latlev)

#%% Vector
Vvec=np.zeros(360)
for i in range(0,360):
    Vvec[i]=DeviationV(2,2,45,i)

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