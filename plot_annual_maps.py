#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset
import geopy.distance
import matplotlib.dates as mdates


import cartopy.crs as ccrs
import cartopy.feature as cfeature

from mpl_toolkits.mplot3d import Axes3D
import dask

#import wrf
#from wrf import getvar

from scipy import stats

import seaborn as sns
from scipy.stats import skew
from scipy.stats import shapiro

from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patches as patches

def run_script(script_path):
    with open(script_path) as script_file:
        exec(script_file.read(), globals())


# In[2]:


# setup some starting data
WRF_3D = Dataset('../../WRF_general_output/wrfout_d03_2014-01-01_00:00:00')
WRF = xr.open_mfdataset(f'../../WRF_general_output/static/wrfout_d03_static')

#WRF_data/tslist/



if __name__ == "__main__":
    run_script('loadData.py')
    run_script('weighted4pts.py')


# In[11]:


exp1='glac2019'; exp2='noice_BT'#glac2100_dem2100'#noice_dtm50'#
#exp1='noice_BT'; exp2='modlakes_noice_BT'

years = np.arange(2007,2023)#23) #2023 or 2012
season = 'all' # SONDJFMAM, DJF, JJA, all, MAMJJA
WRF_hgt_exp1, WRF_hgt_exp2, WRF_lu_exp1, WRF_lu_exp2, WRF_lon, WRF_lat = defineStaticData(exp1=exp1, exp2=exp2)
WRF_precip_exp1, WRF_precip_exp2, WRF_precip_var2_exp1, WRF_precip_var2_exp2, obs_precip = definePrecipData(years=years, season=season, exp1=exp1, exp2=exp2)
WRF_temp_exp1, WRF_temp_exp2, obs_temp = defineTempData(years=years, exp1=exp1, exp2=exp2, season=season)


if exp1 == 'glac2019':
    exp1lab = 'control'
elif exp1 == 'noice_BT':
    exp1lab = 'w/out future lakes'
    
if exp2 == 'noice_BT':
    exp2lab = 'no ice volume'
elif exp2 == 'noice_dtm50':
    exp2lab = 'no ice surface'
elif exp2 == 'glac2100_dem2100':
    exp2lab = 'ice volume 2100'
elif exp2 == 'modlakes_noice_BT':
    exp2lab = 'no ice volume: w/ future lakes'


    
#%%

# general plotting settings

xmin = 5.78; xmax = 8.12
ymin = 61.28; ymax = 61.95
levels2 = np.arange(0,2000,300)

if __name__ == "__main__":
    run_script('plotData.py')

#plotPrecipAbs()#vmax=1000)#1200)#25000)
#plotSnowAbs()#vmax=1000)#1200)#25000)
#plotPrecipDiff(vmin=-10, vmax=10)#vmin=-20, vmax=20)#vmin=-2, vmax=2)#vmin=-2, vmax=2)#
plotSnowDiff(vmin=-50, vmax=50)#vmin=-20, vmax=20)#vmin=-500, vmax=500)#
#plotRainDiff(vmin=-50, vmax=50)#vmin=-20, vmax=20)#vmin=-500, vmax=500)#

#plotTempAbs(temp_model = WRF_temp_exp1, vmin=0,vmax=14)#vmin=-8,vmax=8)
#plotTempDiff(vmin=-1, vmax=1)#vmin=-1, vmax=1)#vmin=-2,vmax=2)#

#plotLUDiff()
#plotLUDiff3D()
#plotHGTDiff()
#plotHGTDiff3D()


#%%

WRF_lon, WRF_lat, (precip_model_exp2-precip_model_exp1)/precip_model_exp1*100

mask_all = WRF_lon.values > 0
mask_west = WRF_lon.values < 6.5
mask_center = (WRF_lon.values >= 6.5) & (WRF_lon.values <= 7.5)
mask_east = WRF_lon.values > 7.5

mask = mask_center
print (np.nanmean((precip_model_exp2[mask]-precip_model_exp1[mask])/precip_model_exp1[mask]*100))

# relative change in annual precip (%)
# all: -0.6612496
# west: -0.123783626
# center: -2.0209389
# east: 0.5548574

#%%

mask = ((WRF_lu_exp1.values == 24) & (WRF_lu_exp2.values != 24))

mask.sum()


#%%

WRF_lon, WRF_lat, (snow_model_exp2-snow_model_exp1)/len(years)

target_lon = 6.971929730189916; target_lat = 61.28864257057141 # Hodlekve
target_lon = 7.217269140081988; target_lat = 61.33605680043416 # Heggis

# Flatten the arrays to make distance calculations easier
lon_flat = WRF_lon.values.flatten()
lat_flat = WRF_lat.values.flatten()

# Calculate the distance (using Euclidean here)
distances = np.sqrt((lon_flat - target_lon)**2 + (lat_flat - target_lat)**2)

# Find the index of the minimum distance
min_index = np.argmin(distances)

print (lon_flat[min_index], lat_flat[min_index], ((snow_model_exp2-snow_model_exp1)/len(years)).flatten()[min_index])

# absolute change in annual snow (mm s.w.e.)
# Hodlekve: -27.262573
# Heggis: -1.2212524

#%%

fig, ax = plt.subplots()

c = ax.pcolormesh(WRF_lon, WRF_lat, (snow_model_exp2-snow_model_exp1)/(((precip_model_exp2-precip_model_exp1)-(snow_model_exp2-snow_model_exp1))),
               vmin=-5, vmax=5,  
               cmap='coolwarm_r')
if exp1 == 'glac2019':
    indices = np.argwhere((WRF_lu_exp1.values == 24) & (WRF_lu_exp2.values != 24))
    mc = 'k'
    for (i, j) in indices:
        ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=.1, zorder=1000)
        
plt.colorbar(c)
        
plt.show()

np.sort(np.array(((snow_model_exp2-snow_model_exp1)[mask]-(((precip_model_exp2-precip_model_exp1)-(snow_model_exp2-snow_model_exp1))[mask]))/((((precip_model_exp2-precip_model_exp1)-(snow_model_exp2-snow_model_exp1))[mask]))))
        
# In[180]:


#WRF_3D_exp1 = Dataset('/home/woody/gwgk/gwgi019h/output_WRF/real/wrflowinp_d03_2016')
#WRF_3D_exp2 = Dataset('/lustre/gwgk/gwgi019h/WRF/run4/wrflowinp_d03')
#WRF_3D_exp1 = Dataset('../../WRF_general_output/restart/nomodlakes_glac2019/wrfrst_d03_2007-01-01_00:00:00')
#WRF_3D_exp2 = Dataset('../../WRF_general_output/restart/nomodlakes_noice_dtm50/wrfrst_d03_2007-01-01_00:00:00')


# In[90]:


# #WRF_3D = Dataset('/home/woody/gwgk/gwgi019h/output_WRF/nomodlakes_glac2019/output_wrfout/wrfout_d03_2010-08-01_00:00:00')

# WRF_3D_exp1 = Dataset('../../WRF_general_output/static/wrf_static_d03_nomodlakes_noice_dtm50')
# WRF_3D_exp2 = Dataset('../../WRF_general_output/static/wrf_static_d03_nomodlakes_noice_BT')
# #WRF_3D_exp1 = Dataset('/home/woody/gwgk/gwgi019h/output_WRF/nomodlakes_glac2019/wrfuserout/wrfuserout_d03_1h_2010-01-01_00:00:00')
# #WRF_3D_exp2 = Dataset('/home/woody/gwgk/gwgi019h/output_WRF/modlakes_glac2019/wrfuserout/wrfuserout_d03_1h_2010-01-01_00:00:00')

# var = "LU_INDEX"

# grid_lon = np.squeeze(getvar(WRF_3D, "XLONG"))
# grid_lat = np.squeeze(getvar(WRF_3D, "XLAT"))
# hgt = np.squeeze(getvar(WRF_3D, "HGT"))
# #lu = np.squeeze(getvar(WRF_3D_exp2, "LU_INDEX"))
# var_exp1 = np.squeeze(getvar(WRF_3D_exp1, var))
# var_exp2 = np.squeeze(getvar(WRF_3D_exp2, var))

# xmin=6.1; xmax=7.8; ymin=61.3; ymax=62

# fig = plt.figure(figsize=(10,18))
# ax = plt.axes()

# print (((var_exp1-var_exp2)>0).sum(), np.sum(var_exp1-var_exp2)/3, np.max(var_exp1-var_exp2))

# ax.contour(grid_lon, grid_lat, hgt)
# cs = ax.pcolormesh(grid_lon, grid_lat, var_exp2,#-var_exp2,
#               #vmin=274, vmax=289, 
#               #vmin=3, vmax=5, 
#               #vmin=15, vmax=17, 
#               #vmin=-.1, vmax=.1, 
#               #vmin=0, vmax=10,
#               #vmin=.01, vmax=.1,
#               cmap='viridis')
# plt.colorbar(cs, orientation='horizontal')

# ax.set_xlim([xmin,xmax])
# ax.set_ylim([ymin,ymax])


# In[72]:


try:
    precip_ice
except:
    precip_ice = {}
    precip_ice_low = {}
    precip_ice_high = {}
    precip_noice = {}
    precip_noice_low = {}
    precip_noice_high = {}
else:
    pass

if exp1 == 'glac2019':
    ice_mask = ((WRF_lu_exp1.values == 24))
    
    thr_ice = np.median(WRF_hgt_exp1.values[ice_mask])
    ice_low_mask = (WRF_hgt_exp1.values < thr_ice)
    ice_high_mask = (WRF_hgt_exp1.values >= thr_ice)
    
    thr_noice = np.median(WRF_hgt_exp1.values[~ice_mask])
    noice_low_mask = (WRF_hgt_exp1.values < thr_noice)
    noice_high_mask = (WRF_hgt_exp1.values >= thr_noice)
        
for exp, WRF_precip_exp in zip([exp1, exp2], [WRF_precip_exp1, WRF_precip_exp2]):
    try:
        precip_ice[exp]
    except:
        precip_ice[exp] = []
        precip_ice_low[exp] = []
        precip_ice_high[exp] = []
        precip_noice[exp] = []
        precip_noice_low[exp] = []
        precip_noice_high[exp] = []

    for key, data in WRF_precip_exp.items():
        precip_ice[exp].append(data[ice_mask].mean())
        precip_ice_low[exp].append(data[ice_mask & ice_low_mask].mean())
        precip_ice_high[exp].append(data[ice_mask & ice_high_mask].mean())
        precip_noice[exp].append(data[~ice_mask].mean())
        precip_noice_low[exp].append(data[~ice_mask & noice_low_mask].mean())
        precip_noice_high[exp].append(data[~ice_mask & noice_high_mask].mean())
    
        


# In[73]:


try:
    temp_ice
except:
    temp_ice = {}
    temp_ice_low = {}
    temp_ice_high = {}
    temp_noice = {}
    temp_noice_low = {}
    temp_noice_high = {}
else:
    pass

if exp1 == 'glac2019':
    ice_mask = ((WRF_lu_exp1.values == 24))
    
    thr_ice = np.median(WRF_hgt_exp1.values[ice_mask])
    ice_low_mask = (WRF_hgt_exp1.values < thr_ice)
    ice_high_mask = (WRF_hgt_exp1.values >= thr_ice)
    
    thr_noice = np.median(WRF_hgt_exp1.values[~ice_mask])
    noice_low_mask = (WRF_hgt_exp1.values < thr_noice)
    noice_high_mask = (WRF_hgt_exp1.values >= thr_noice)
    
for exp, WRF_temp_exp in zip([exp1, exp2], [WRF_temp_exp1, WRF_temp_exp2]):
    try:
        temp_ice[exp]
    except:
        temp_ice[exp] = []
        temp_ice_low[exp] = []
        temp_ice_high[exp] = []
        temp_noice[exp] = []
        temp_noice_low[exp] = []
        temp_noice_high[exp] = []

    for key, data in WRF_temp_exp.items():
        temp_ice[exp].append(data[ice_mask].mean())
        temp_ice_low[exp].append(data[ice_mask & ice_low_mask].mean())
        temp_ice_high[exp].append(data[ice_mask & ice_high_mask].mean())
        temp_noice[exp].append(data[~ice_mask].mean())
        temp_noice_low[exp].append(data[~ice_mask & noice_low_mask].mean())
        temp_noice_high[exp].append(data[~ice_mask & noice_high_mask].mean())
    
        


# In[83]:


fig, ax1 = plt.subplots(figsize=(10,7))

for i, key in enumerate(precip_ice.keys()):
    if i >= 1:
        total_key = np.array(precip_ice[key])+np.array(precip_noice[key])
        total_ref = np.array(precip_ice['glac2019'])+np.array(precip_noice['glac2019'])
        ax1.plot((total_key-total_ref), c=f'C{i-1}', lw=3, alpha=.5, label=key) #/total_ref*100
        ax1.axhline(np.mean(total_key-total_ref), c=f'C{i-1}')
        ax1.plot((np.array(precip_ice[key])-np.array(precip_ice['glac2019'])), c=f'C{i-1}', ls='-.') #/np.array(precip_ice['glac2019'])*100
#        ax1.plot(np.array(precip_ice_low[key])-np.array(precip_ice_low['glac2019']), c=f'C{i-1}', label=key)
#        ax1.plot(np.array(precip_ice_high[key])-np.array(precip_ice_high['glac2019']), c=f'C{i-1}', ls='-.')
        ax1.plot((np.array(precip_noice_low[key])-np.array(precip_noice_low['glac2019'])), c=f'C{i-1}', ls='--') #/np.array(precip_noice_low['glac2019'])*100
        ax1.plot((np.array(precip_noice_high[key])-np.array(precip_noice_high['glac2019'])), c=f'C{i-1}', ls=':') #/np.array(precip_noice_high['glac2019'])*100
        print (np.mean(total_key-total_ref)*12*12)
ax1.plot((),(), c='w', ls='-', label=' ')
ax1.plot((),(), c='grey', lw=3, alpha=.5, ls='-', label='total')
ax1.plot((),(), c='grey', ls='-.', label='over ice')
#ax1.plot((),(), c='grey', ls='-', label='over ice (low)')
#ax1.plot((),(), c='grey', ls='-.', label='over ice (high)')
ax1.plot((),(), c='grey', ls='--', label='outside ice (low)')
ax1.plot((),(), c='grey', ls=':', label='outside ice (high)')
ax1.set_xticks(np.arange(0,len(precip_ice[key])+1,12*2))
ax1.set_xticklabels(np.arange(years[0],years[0]+int(len(precip_ice[key])/12)+1,2))
ax1.set_xlabel('Year')
ax1.set_ylabel('Difference in preciptiation \nrelative to control experiment (mm)')
#ax1.set_xlim(xmin=144, xmax=192)
#ax1.set_ylim(ymin=-10, ymax=10)
ax1.legend(ncols=2)
plt.grid()
plt.show()


# In[80]:


labels = ['ice-volume-2100', 'no-ice-surface', 'no-ice-volume']

fig, ax1 = plt.subplots(figsize=(7,7))

for i, key in enumerate(precip_ice.keys()):
    if i >= 1 and i < 6:
        ax1.plot(np.cumsum([i+n for i,n in zip(precip_ice[key], precip_noice[key])])-np.cumsum([i+n for i,n in zip(precip_ice['glac2019'], precip_noice['glac2019'])]), c=f'C{i-1}', lw=3, alpha=.5, label=labels[i-1])
        ax1.plot(np.cumsum(precip_ice[key])-np.cumsum(precip_ice['glac2019']), c=f'C{i-1}', ls='-.')
        #ax1.plot(np.cumsum(precip_ice_low[key])-np.cumsum(precip_ice_low['glac2019']), c=f'C{i-1}', label=key)
        #ax1.plot(np.cumsum(precip_ice_high[key])-np.cumsum(precip_ice_high['glac2019']), c=f'C{i-1}', ls='-.')
        ax1.plot(np.cumsum(precip_noice_low[key])-np.cumsum(precip_noice_low['glac2019']), c=f'C{i-1}', ls='--')
        ax1.plot(np.cumsum(precip_noice_high[key])-np.cumsum(precip_noice_high['glac2019']), c=f'C{i-1}', ls=':')

        # in percentage:
#        ax1.plot((np.array([i+n for i,n in zip(precip_ice[key], precip_noice[key])])-np.array([i+n for i,n in zip(precip_ice['glac2019'], precip_noice['glac2019'])]))/np.array([i+n for i,n in zip(precip_ice['glac2019'], precip_noice['glac2019'])])*100, c=f'C{i-1}', lw=3, alpha=.5, label=key)
#        ax1.plot((np.array(precip_ice[key])-np.array(precip_ice['glac2019']))/np.array(precip_ice['glac2019'])*100, c=f'C{i-1}', ls='-.')
#        ax1.plot((np.array(precip_noice_low[key])-np.array(precip_noice_low['glac2019']))/np.array(precip_noice_low['glac2019'])*100, c=f'C{i-1}', ls='--')
#        ax1.plot((np.array(precip_noice_high[key])-np.array(precip_noice_high['glac2019']))/np.array(precip_noice_high['glac2019'])*100, c=f'C{i-1}', ls=':')

ax1.plot((),(), c='w', ls='-', label=' ')
ax1.plot((),(), c='grey', ls='-', lw=3, alpha=.5, label='total')
ax1.plot((),(), c='grey', ls='-.', label='over ice')
#ax1.plot((),(), c='grey', ls='-', label='over ice (low)')
#ax1.plot((),(), c='grey', ls='-.', label='over ice (high)')
ax1.plot((),(), c='grey', ls='--', label='outside ice (low)')
ax1.plot((),(), c='grey', ls=':', label='outside ice (high)')
ax1.set_xticks(np.arange(0,len(precip_ice[key])+1,24))
ax1.set_xticklabels(np.arange(years[0],years[0]+int(len(precip_ice[key])/12)+1,2))
ax1.set_xlabel('Year')
ax1.set_ylabel('Difference in accummulated preciptiation \nrelative to control experiment (mm)')
#ax1.set_ylim(-230,50)
ax1.legend(ncols=1, fontsize=14)
plt.grid()
plt.show()


# In[84]:


fig, ax1 = plt.subplots(figsize=(10,7))

for i, key in enumerate(temp_ice.keys()):
    if i >= 1:
        ax1.plot(np.array(temp_ice_low[key])-np.array(temp_ice_low['glac2019']), c=f'C{i-1}', label=key)
        ax1.plot(np.array(temp_ice_high[key])-np.array(temp_ice_high['glac2019']), c=f'C{i-1}', ls='-.')
        ax1.plot(np.array(temp_noice_low[key])-np.array(temp_noice_low['glac2019']), c=f'C{i-1}', ls='--')
        ax1.plot(np.array(temp_noice_high[key])-np.array(temp_noice_high['glac2019']), c=f'C{i-1}', ls=':')

ax1.plot((),(), c='w', ls='-', label=' ')
ax1.plot((),(), c='grey', ls='-', label='over ice (low)')
ax1.plot((),(), c='grey', ls='-.', label='over ice (high)')
ax1.plot((),(), c='grey', ls='--', label='outside ice (low)')
ax1.plot((),(), c='grey', ls=':', label='outside ice (high)')
ax1.set_xticks(np.arange(0,len(temp_ice[key])+1,24))
ax1.set_xticklabels(np.arange(years[0],years[0]+int(len(temp_ice[key])/12)+1,2))
ax1.set_xlabel('Year')
ax1.set_ylabel('Difference in temperature \nrelative to control experiment ($\u00b0$C)')
#ax1.set_xlim(xmin=0, xmax=24)
ax1.set_xlim(xmin=144, xmax=192)
ax1.legend(ncols=2, loc=3)
plt.grid()
plt.show()


# In[78]:


len(temp_ice[key])+1


# In[67]:


fig, ax1 = plt.subplots(figsize=(10,7))

for i, key in enumerate(temp_ice.keys()):
    if i >= 0:
        ax1.plot(np.array(temp_ice_low[key]), c=f'C{i}', label=key)
        ax1.plot(np.array(temp_ice_high[key]), c=f'C{i}', ls='-.')
        ax1.plot(np.array(temp_noice_low[key]), c=f'C{i}', ls='--')
        ax1.plot(np.array(temp_noice_high[key]), c=f'C{i}', ls=':')

ax1.plot((),(), c='w', ls='-', label=' ')
ax1.plot((),(), c='grey', ls='-', label='over ice (low)')
ax1.plot((),(), c='grey', ls='-.', label='over ice (high)')
ax1.plot((),(), c='grey', ls='--', label='outside ice (low)')
ax1.plot((),(), c='grey', ls=':', label='outside ice (high)')
ax1.set_xticks(np.arange(0,len(temp_ice[key])+1,24))
ax1.set_xticklabels(np.arange(years[0],years[0]+int(len(temp_ice[key])/12)+1,2))
ax1.set_xlabel('Year')
ax1.set_ylabel('Temperature (K)')
#ax1.set_xlim(xmin=0, xmax=48)
ax1.legend(loc=1)
plt.grid()
plt.show()


# In[ ]:





# In[138]:


# difference in temperature for outside vs inside ice cap

ice_mask = ((WRF_lu_exp1.values == 24))# & (WRF_lu_exp2.values != 24))
ice_temp = {}
noice_temp = {}

ls = ['-','--']
    
fig, ax1 = plt.subplots(figsize=(10,7))
    
for i, exp in enumerate([WRF_temp_exp1, WRF_temp_exp2]):
    
    ice_temp[str(i)] = []
    noice_temp[str(i)] = []

    for key in exp.keys():
        #print (key, exp[key][ice_mask].mean())
        ice_temp[str(i)].append(exp[key][ice_mask].mean())
        noice_temp[str(i)].append(exp[key][~ice_mask].mean())
        
    x = range(len(WRF_temp_exp1.keys()))

    ax1.plot(x, np.array(ice_temp[str(i)])-273.15, c='C0', ls=ls[i], label='inside ice cap')
    ax1.plot(x, np.array(noice_temp[str(i)])-273.15, c='C1', ls=ls[i], label='outside ice cap')
    ax1.plot((),(), c='grey', ls=ls[i], label='difference')
    ax1.plot((),(), c='k', ls=ls[0], label=exp1)
    ax1.plot((),(), c='k', ls=ls[1], label=exp2)
    ax2 = ax1.twinx()
    ax2.plot(x, np.array(noice_temp[str(i)])-np.array(ice_temp[str(i)]), c='grey', ls=ls[i])

    ax1.set_ylim(-15,30)
    ax2.set_ylim(0,7.5)
    ax1.set_xticks(x[12::24])
    ax1.set_xticklabels([a[:4] for a in list(WRF_temp_exp1.keys())[12::24]])
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Temperature ($\u00b0$)')
    if i == 0:
        ax1.legend(loc=2)

plt.show()


# In[139]:


mean_ice_temp = np.zeros((2, len(years)))
mean_noice_temp = np.zeros((2, len(years)))

for e in range(2):
    for i in range(len(years)):
        mean_ice_temp[e, i]   = np.mean(ice_temp[str(e)][i*12:(i+1)*12])#[i::12])
        mean_noice_temp[e, i] = np.mean(noice_temp[str(e)][i*12:(i+1)*12])#i::12])


# In[140]:


fig, ax1 = plt.subplots()
        
ax1.set_xticks(range(len(years))[::2])
#ax1.set_xticklabels(['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])
ax1.set_xticklabels(np.arange(years[0], years[-1]+1, 2))
ax1.set_yticks([0,1])
ax1.set_yticklabels([exp1, exp2])
temp = ax1.imshow(mean_noice_temp-mean_ice_temp, cmap='viridis')#vmin=4, vmax=6, 
plt.colorbar(temp, orientation='horizontal', label='Temperature difference (K)')
ax1.invert_yaxis()
ax1.set_title('Difference in annual mean temperature \nfor inside vs outside the ice cap') #monthly
plt.show()


# In[141]:


# difference in precipitation for outside vs inside ice cap

ice_mask = ((WRF_lu_exp1.values == 24) & (WRF_lu_exp2.values != 24))
ice_precip = {}
noice_precip = {}

ls = ['-','--']
    
fig, ax1 = plt.subplots(figsize=(10,7))
ax2 = ax1.twinx()
    
for i, exp in enumerate([WRF_precip_exp1, WRF_precip_exp2]):
    
    ice_precip[str(i)] = []
    noice_precip[str(i)] = []

    for key in exp.keys():
        ice_precip[str(i)].append(exp[key][ice_mask].mean())
        noice_precip[str(i)].append(exp[key][~ice_mask].mean())
        
    x = range(len(WRF_temp_exp1.keys()))

    ax1.plot(x, np.array(ice_precip[str(i)]), c='C0', ls=ls[i], label='inside ice cap')
    ax1.plot(x, np.array(noice_precip[str(i)]), c='C1', ls=ls[i], label='outside ice cap')
    ax1.plot((),(), c='grey', ls=ls[i], label='difference')
    ax1.plot((),(), c='k', ls=ls[0], label=f'{exp1}')
    ax1.plot((),(), c='k', ls=ls[1], label=f'{exp2}')
    ax2.plot(x, np.array(noice_precip[str(i)])-np.array(ice_precip[str(i)]), c='grey', ls=ls[i])

    ax1.set_ylim(0,1400)
    ax2.set_ylim(-800,50)
    ax1.set_yticks(np.arange(0,900,200))
    ax2.set_yticks(np.arange(-200,10,100))
    ax1.set_xticks(x[12::24])
    ax1.set_xticklabels([a[:4] for a in list(WRF_precip_exp1.keys())[12::24]])
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Precipitation (mm)                               ')
    if i == 0:
        ax1.legend(loc=5)
        
ax1.grid()
ax2.grid()

plt.show()


# In[ ]:

############################



# In[64]:



# # Create a list to store the temperature data for both experiments
# temp_exp1_list = []
# temp_exp2_list = []

# # Iterate through the dictionary keys and append data to the lists
# for month in WRF_temp_exp1.keys():
#     temp_exp1_list.append(WRF_temp_exp1[month])  # Get 2D data as numpy array
#     temp_exp2_list.append(WRF_temp_exp2[month])  # Get 2D data as numpy array

# # Convert lists to 3D numpy arrays
# temp_exp1_3D = np.array(temp_exp1_list)  # Shape will be (num_months, num_lat, num_lon)
# temp_exp2_3D = np.array(temp_exp2_list)  # Same shape as temp_exp1_3D


# # In[65]:




# # 1. Calculate the differences
# differences = temp_exp1_3D - temp_exp2_3D

# # 2. Flatten the 3D differences array to 1D for analysis
# flat_differences = differences.flatten()

# # 3. Visualize the differences
# plt.figure(figsize=(12, 6))

# # Histogram
# plt.subplot(1, 2, 1)
# sns.histplot(flat_differences, bins=20, kde=True)
# plt.title('Histogram of Temperature Differences')
# plt.xlabel('Differences')
# plt.ylabel('Frequency')

# # Box plot
# plt.subplot(1, 2, 2)
# sns.boxplot(x=flat_differences)
# plt.title('Box Plot of Temperature Differences')
# plt.xlabel('Differences')

# plt.tight_layout()
# plt.show()

# # 4. Calculate Skewness
# skewness = skew(flat_differences)
# print(f'Skewness: {skewness}')


# # In[54]:


# # Initialize a significance map
# num_months, num_lat, num_lon = temp_exp1_3D.shape
# significance_map = np.zeros((num_lat, num_lon))

# alpha = 0.05

# # Loop over each grid cell
# for i in range(num_lat):
#     for j in range(num_lon):
#         # Extract paired samples for the specific grid cell
#         sample_1 = temp_exp1_3D[:, i, j]  # All months for grid cell (i, j) in experiment 1
#         sample_2 = temp_exp2_3D[:, i, j]  # All months for grid cell (i, j) in experiment 2

#         # Ensure we have enough data to perform the test
#         if len(sample_1) < 2 or len(sample_2) < 2:
#             continue  # Not enough data points for t-test

#         # Perform the paired t-test
#         t_statistic, p_value = stats.ttest_rel(sample_1, sample_2)
#         print (t_statistic, p_value)
        
#         # Store significance
#         significance_map[i, j] = 1 if p_value < alpha else 0

# # Now, significance_map tells you where the changes are statistically significant.


# # In[59]:



# # Calculate differences
# differences = temp_exp1_3D - temp_exp2_3D

# # Flatten differences for all grid cells or handle each grid cell separately
# flat_differences = differences.flatten()  # This will create a 1D array of differences for all cells

# # Conduct the Shapiro-Wilk test
# stat, p_value = shapiro(flat_differences)
# print('Shapiro-Wilk Test Statistic:', stat)
# print('P-value:', p_value)


# # In[58]:


# plt.imshow(significance_map, cmap='RdYlGn', interpolation='nearest')
# plt.colorbar(label='Significance (1=Significant, 0=Not Significant)')
# plt.title('Statistical Significance of Temperature Change')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()


# In[ ]:





# In[33]:


# loading time series
inpath = '../../output_subsets/timeseries'
experiments = ['glac2006', 'glac2019', 'glac2019_modlakes', 'glac2100_dtm50', 'glac2100_dem2100', 'noice_dtm50', 'noice_BT', 'noice_BT_modlakes']
experiments = ['glac2006', 'glac2019', 'glac2100_dtm50', 'glac2100_dem2100', 'noice_dtm50', 'noice_BT', 'noice_BT_modlakes']
experiments = ['glac2019', 'glac2100_dem2100', 'noice_dtm50', 'noice_BT']
experiments = ['noice_BT']
experiments = ['glac2019', 'noice_BT']#'noice_dtm50']#
#experiments = ['noice_dtm50', 'noice_dtm50_updated', 'noice_BT', 'noice_BT_updated']
#experiments = ['glac2019']
#experiments = ['glac2006', 'noice_dtm50']

#exp2 = 'noice_BT'#noice_dtm50'

# In[34]:


# load temperature
t2m_1pt = {}
t2m_4pts = {}
for exp in experiments:
    dataframes_1pt = []
    dataframes_4pts = []
    for year in range(2007, 2023):#18):
        df = pd.read_csv(f'{inpath}/temp/T2_1pt_{year}_{exp}.csv')
        dataframes_1pt.append(df)
        #df = pd.read_csv(f'{inpath}/temp/T2_4pts_{year}_{exp}.csv')
        #dataframes_4pts.append(df)

    t2m_1pt[exp] = pd.concat(dataframes_1pt, ignore_index=True)
    #t2m_4pts[exp] = pd.concat(dataframes_4pts, ignore_index=True)


# In[27]:


#if __name__ == "__main__":
#    run_script('getERA5DATA.py')


# In[42]:


for exp in experiments:
    t2m_1pt[exp]['date'] = pd.to_datetime(t2m_1pt[exp]['date'])
#era5['time'] = pd.to_datetime(era5['time'])
#
#fig = plt.subplots(figsize=(15,6))
#
##mask = era5['time'] > datetime.datetime(2013,1,1)
#plt.plot(era5['time'][4000:],era5['t2m'][4000:,era5_1000_j[7],era5_1000_i[7]])
#    
#mask = t2m_1pt['glac2019']['date'] > datetime.datetime(2013,1,1)
#plt.plot(t2m_1pt['glac2019'][mask]['date'],t2m_1pt['glac2019'][mask]['OV']-273.15)
#
#mask = (nve['station_id'] == 'OV') & (nve['date'] > datetime.datetime(2013,1,1)) & (nve['date'] < datetime.datetime(2018,1,1))
#plt.plot(nve[mask]['date'], nve[mask]['t'])
#
#plt.xlim(xmin=datetime.datetime(2013,1,1), xmax=datetime.datetime(2017,12,31))
#
#plt.axvline(datetime.datetime(2014,9,1),c='y')
#plt.axvline(datetime.datetime(2016,3,1),c='y')
#plt.axvline(datetime.datetime(2016,5,1),c='y')
#plt.grid()
#plt.show()
#t2m_1pt_monthly[exp][loc]


# In[35]:


# create monthly mean
t2m_1pt_monthly = {}
t2m_4pts_monthly = {}

for exp in experiments:
    t2m_1pt[exp]['date'] = pd.to_datetime(t2m_1pt[exp]['date'])
    t2m_1pt_monthly[exp] = t2m_1pt[exp].groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()#drop=True)
    t2m_1pt_monthly[exp]['date'] -= pd.offsets.MonthBegin() # let date for each month reflect first date of given month

    t2m_4pts[exp]['date'] = pd.to_datetime(t2m_4pts[exp]['date'])
    t2m_4pts_monthly[exp] = t2m_4pts[exp].groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()#drop=True)
    t2m_4pts_monthly[exp]['date'] -= pd.offsets.MonthBegin() # let date for each month reflect first date of given month


# In[36]:


# collect all observed monthly mean temperature in one df
t2m_monthly_obs = pd.DataFrame()
t2m_monthly_obs['date'] = t2m_1pt_monthly[exp]['date']
for st in ['FB', 'OV', 'LV']:#, 'AS']:
    mask = nve_monthly['station_id'] == st
    met_data = nve_monthly[mask][['date', 't']].copy()
    t2m_monthly_obs = t2m_monthly_obs.merge(met_data, on='date', how='left')
    t2m_monthly_obs.rename(columns={'t': st}, inplace=True)
for st in ['MG', 'JD', 'FL']:
    mask = MET_monthly['station'] == st
    met_data = MET_monthly[mask][['date', 'temp']].copy()
    t2m_monthly_obs = t2m_monthly_obs.merge(met_data.reset_index(drop=True), on='date', how='left')
    t2m_monthly_obs.rename(columns={'temp': st}, inplace=True)
for st in ['NB']:
    met_data = NB_monthly[['date', 't']].copy()
    t2m_monthly_obs = t2m_monthly_obs.merge(met_data, on='date', how='left')
    t2m_monthly_obs.rename(columns={'t': st}, inplace=True)


# In[45]:


######################

# In[56]:


# 10 days of missing data at AS in june 2012
corr_1pt = np.nan*np.ones((len(t2m_monthly_obs.columns[1:])))
corr_4pts = np.nan*np.ones((len(t2m_monthly_obs.columns[1:])))

c = plt.color_sequences["tab10"]

fig, ax1 = plt.subplots(figsize=(12,8))
ax1.axhline(0, color='grey')
for exp in experiments[:1]:
    print (exp)
    for i,loc in enumerate(t2m_monthly_obs.columns[1:]):
        print (loc)
        corrlabel = 'not altitude adjusted'
        corrlabel = 'altitude adjusted'
        corr_1pt[i] = 5*10**(-3)*np.squeeze((WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_hgt']-WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'station_hgt']).values)
        corr_4pts[i] = 5*10**(-3)*np.squeeze((WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_hgt_4pts']-WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'station_hgt']).values)

        pts = '1pt'
        corr = 0
        if corrlabel == 'altitude adjusted':
            corr = corr_1pt[i]
        ax1.plot(t2m_monthly_obs['date'], t2m_monthly_obs[loc], c=c[i], ls=':')#, label=f'{loc}_obs')
        ax1.plot(t2m_monthly_obs['date'], t2m_1pt_monthly[exp][loc]+corr-273.15, c=c[i], ls='-', label=f'{loc}')#_{pts}')

#        pts = '4pts'
#        if corrlabel == 'altitude adjusted':
#            corr = corr_4pts[i]
#        ax1.plot(t2m_monthly_obs['date'], t2m_4pts_monthly[exp][loc]+corr-273.15, c=c[i], ls='--')#, label=f'{loc}_{pts}')
ax1.plot((), (), c='w', label=' ')
ax1.plot((), (), c='grey', ls=':', label='obs')
ax1.plot((), (), c='grey', ls='-', label='WRF')#1 pt')
#ax1.plot((), (), c='grey', ls='--', label='4 pts')
ax1.legend(loc=4, ncols=2, fontsize=10)
#ax1.set_xlim(xmin=datetime.datetime(2013,1,1), xmax=datetime.datetime(2016,1,1))
#ax1.set_ylim(-7,3.5)
ax1.set_xlabel('time')
ax1.set_ylabel('monthly temperature ($\u00b0$C)')
ax1.set_title(f'modelled vs observed temperature ({corrlabel})')

ax1.grid()
plt.show()


# In[60]:

plt.rcParams.update({'font.size': 18})

# 10 days of missing data at AS in june 2012
corr_1pt = np.nan*np.ones((len(t2m_monthly_obs.columns[1:])))
corr_4pts = np.nan*np.ones((len(t2m_monthly_obs.columns[1:])))
error_1pt = np.nan*np.ones((len(t2m_monthly_obs.columns[1:]), len(t2m_monthly_obs['date'])))
error_4pts = np.nan*np.ones((len(t2m_monthly_obs.columns[1:]), len(t2m_monthly_obs['date'])))

#c = plt.color_sequences["tab10"]
c = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf', 'k', 'navy', 'aquamarine']
c = [u'#1f77b4', u'#2ca02c', u'#ff7f0e', u'#7f7f7f', u'#bcbd22', 'navy', u'#17becf']
#     u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', 'k', 'navy', 'aquamarine']

handles = []
names = []

fig, ax1 = plt.subplots(figsize=(12,8))
ax1.axhline(0, color='grey')
for exp in experiments[:]:
    for i,loc in enumerate(t2m_monthly_obs.columns[1:]):
        corrlabel = 'not altitude adjusted'
        corrlabel = 'altitude adjusted'
        corr_1pt[i] = 5*10**(-3)*np.squeeze((WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_hgt']-WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'station_hgt']).values)
        corr_4pts[i] = 5*10**(-3)*np.squeeze((WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_hgt_4pts']-WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'station_hgt']).values)
        
        alp=1
        
        pts = '1pt'
        corr = 0
        if corrlabel == 'altitude adjusted':
            corr = corr_1pt[i]
        abserror = abs(t2m_1pt_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15).mean()
        line, = ax1.plot(t2m_monthly_obs['date'], t2m_1pt_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15, c=c[i], alpha=alp, ls='-', label=f'{loc}: {abserror:.1f}')#_{pts}: {abserror:.1f}')
        handles.append(line)  # Store the handle
        names.append(f'{loc}: {abserror:.1f}')
    
        if exp == 'glac2019':
            error_1pt[i] = t2m_1pt_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15
        
        # pts = '4pts'
        # if corrlabel == 'altitude adjusted':
        #     corr = corr_4pts[i]
        # abserror = abs(t2m_4pts_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15).mean()
        # ax1.plot(t2m_monthly_obs['date'], t2m_4pts_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15, c=c[i], ls='--', label=f'{loc}_{pts}: {abserror:.1f}')
        # if exp == 'glac2019':
        #     error_4pts[i] = t2m_4pts_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15

#ax1.plot((), (), c='w', label=' ')
#ax1.plot((), (), c='grey', ls='-', label='1 pt')
#ax1.plot((), (), c='grey', ls='--', label='4 pts')
ordered_indices = [6, 3, 4, 5, 0, 1, 2]
#ax1.legend(loc=3, ncols=3)
ax1.legend([handles[i] for i in ordered_indices], [names[i] for i in ordered_indices], loc=3, ncols=3)
#ax1.set_ylim(-7,3.5)
ax1.set_xlabel('Time')
ax1.set_ylabel('Error in mean monthly temperature (K)')
ax1.set_title(f'Modelled minus observed temperature ({corrlabel})')

ax1.grid()

plt.savefig(f'figures/temp_error.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()


# In[63]:

handles = []

fig, ax = plt.subplots(figsize=(10,7))

for i,loc in enumerate(t2m_monthly_obs.columns[1:]):
    for m in range(12):
        ax.bar(m+i/8, np.nanmean(error_1pt[i][m::12]), width=.125, color=c[i])
    line, = ax.bar(m+i/8, np.nanmean(error_1pt[i][m::12]), width=.125, color=c[i], label=loc)
    handles.append(line)  # Store the handle

#plt.legend(ncols=2)
ax.legend([handles[i] for i in ordered_indices], [names[i][:2] for i in ordered_indices], fontsize=14.5, loc=4, ncols=3)
ax.set_xticks(np.arange(.375,12,1))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
ax.set_xlabel('Month')
ax.set_ylabel('Mean error in mean monthly temperature (K)')
plt.grid()
plt.savefig(f'figures/temp_monthly_error.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()


# In[9]:


#mask_OV = nve_monthly['station_id'] == 'OV'
#mask_LV = nve_monthly['station_id'] == 'LV'
#mask_FB = nve_monthly['station_id'] == 'FB'
#mask_AS = nve_monthly['station_id'] == 'AS'
#mask_MG = MET_monthly['station'] == 'MG'
#mask_FL = MET_monthly['station'] == 'FL'

ls = ['-','--',':']

fig, ax1 = plt.subplots(figsize=(12,8))
for i,loc in enumerate(t2m_monthly_obs.columns[1:]):
    ax1.plot(t2m_monthly_obs['date'], t2m_1pt_monthly[experiments[1]][loc]+corr_1pt[i]-273.15-t2m_monthly_obs[loc], c=c[i], alpha=.2, lw=4, ls='-')#, label=loc)
    ax1.plot(t2m_1pt_monthly[exp]['date'], t2m_1pt_monthly[experiments[1]][loc]-t2m_1pt_monthly[experiments[2]][loc], c=c[i], lw=2, ls='-', label=loc)#, label=f'{experiments[0]}-{experiments[1]})
ax1.plot((), (), c='w', label=' ')
ax1.plot((), (), c='grey', ls=ls[0], alpha=.2, lw=4, label=f'{experiments[1]}-obs')
ax1.plot((), (), c='grey', ls=ls[0], lw=2, label=f'{experiments[1]}-{experiments[2]}')

#    for e,exp in enumerate(experiments):
#        ax1.scatter(t2m_monthly_obs['date'], t2m_monthly_obs[loc], c=c[i])#, ls='-', label=loc)
#        if e == 0:
#            ax1.plot(t2m_1pt_monthly[exp]['date'], t2m_1pt_monthly[exp][loc]-273.15, c=c[i], ls=ls[e], label=loc)
#        else:
#            ax1.plot(t2m_1pt_monthly[exp]['date'], t2m_1pt_monthly[exp][loc]-273.15, c=c[i], ls=ls[e])
#if len(experiments) > 1:
#    ax1.plot((), (), c='w', label=' ')
#    ax1.plot((), (), c='grey', ls=ls[0], label=f'{experiments[0]}')
#    ax1.plot((), (), c='grey', ls=ls[1], label=f'{experiments[1]}')

ax1.legend()
#ax1.set_ylim(-5.5,3.5)
ax1.set_xlabel('time')
ax1.set_ylabel('temperature ($\u00b0$C)')
ax1.set_title('experiment sensitivity relative to model bias (after altitude correction)')

plt.show()


# In[25]:





# In[38]:


ls = ['-','--',':','-.']
corr_1pt = np.nan*np.ones((len(t2m_monthly_obs.columns[1:])))

fig, ax1 = plt.subplots(figsize=(12,8))
for e,exp in enumerate(experiments): #'glac2006', ['glac2019','glac2100_dtm50', 'noice_dtm50', 'noice_BT']):#
    print (exp)
    for i,loc in enumerate(['OV', 'FB']):#'LV']):#, 'FL'
        corrlabel = 'altitude adjusted'
        corr_1pt[i] = 5*10**(-3)*(WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_hgt']-WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'station_hgt']).values
        
        if i == 0:
            ax1.plot(t2m_1pt_monthly[exp]['date'], t2m_1pt_monthly[exp][loc]+corr_1pt[i]-273.15, marker='.', c=c[e], lw=2, ls=ls[i], label=exp)#, label=loc)
        else:
            ax1.plot(t2m_1pt_monthly[exp]['date'], t2m_1pt_monthly[exp][loc]+corr_1pt[i]-273.15, marker='.', c=c[e], lw=2, ls=ls[i])#, label=exp)#, label=loc)
        #ax1.plot(t2m_1pt_monthly[exp]['date'], t2m_1pt_monthly[exp][loc], c=c[e], lw=2, ls=ls[1])#, label=loc)#, label=f'{experiments[0]}-{experiments[1]})
ax1.plot((), (), c='w', label=' ')
ax1.plot((), (), c='k', ls=ls[0], lw=2, label=f'OV')
ax1.plot((), (), c='k', ls=ls[1], lw=2, label=f'FB')
#ax1.plot((), (), c='k', ls=ls[2], lw=2, label=f'LV')
#ax1.plot((), (), c='k', ls=ls[3], lw=2, label=f'FL')

ax1.legend()#loc=2)
#ax1.set_xlim(datetime.datetime(2007,9,1),datetime.datetime(2010,4,1))
#ax1.set_ylim(-10,5)
ax1.set_xlabel('time')
ax1.set_ylabel('temperature ($\u00b0$C)')
ax1.set_title('experiment sensitivity')

#ax1.set_xlim(xmin=datetime.datetime(2013,1,1))
plt.grid()
plt.show()


# In[ ]:





# In[8]:


# load precip

c = plt.color_sequences["tab10"]
years = range(2007, 2023)

prec_1pt = {}
prec_4pts = {}
for exp in experiments:
    dataframes_1pt = []
    dataframes_4pts = []
    for year in years:
        df = pd.read_csv(f'{inpath}/precip/RAINNC_1pt_{year}_{exp}.csv')
        dataframes_1pt.append(df)
        df = pd.read_csv(f'{inpath}/precip/RAINNC_4pts_{year}_{exp}.csv')
        dataframes_4pts.append(df)

    prec_1pt[exp] = pd.concat(dataframes_1pt, ignore_index=True)
    prec_4pts[exp] = pd.concat(dataframes_4pts, ignore_index=True)
    
for exp in experiments:
    prec_1pt[exp]['date'] = pd.to_datetime(prec_1pt[exp]['date'])
    prec_4pts[exp]['date'] = pd.to_datetime(prec_4pts[exp]['date'])


# In[14]:


stations = ['MG', 'FL', 'VS', 'OD', 'SJ']
c = plt.color_sequences["tab10"]
for exp in experiments:
    for s, st in enumerate(stations):
        plt.plot(prec_1pt[exp]['date'][6::24], prec_1pt[exp][st][6::24], c=c[s], label=st)
        plt.plot(prec_4pts[exp]['date'][6::24], prec_4pts[exp][st][6::24], c=c[s], ls = '--')
plt.plot((),(), c='white', ls = '-', label=' ')
plt.plot((),(), c='grey', ls = '-', label='1pt')
plt.plot((),(), c='grey', ls = '--', label='4pts')
plt.legend()


# In[ ]:























#%%

# inpath = '../../output_subsets/timeseries'
# experiments = ['glac2100_dem2100']#'noice_dtm50']#'glac2019']#,'noice_BT']#


# for exp in experiments:
#     for var in ['RAINNC']:
#         dataframes_1pt = []
#         #dataframes_4pts = []
#         for year in range(2007, 2023):
#             df = pd.read_csv(f'{inpath}/precip/original/{var}_1pt_{year}_{exp}.csv')
#             df2 = pd.read_csv(f'{inpath}/precip/PEAK/{var}_1pt_{year}_{exp}.csv')
#             df['date'] = pd.to_datetime(df['date'])
#             df2['date'] = pd.to_datetime(df2['date'])
#             df = pd.merge(df, df2, on='date', how='inner')

#             df.to_csv(f'{inpath}/precip/{var}_1pt_{year}_{exp}.csv')
            


# In[26]:


# load wind and precip timeseries

inpath = '../../output_subsets/timeseries'
experiments = ['glac2019','noice_BT']#'noice_dtm50']#
precip_1pt = {}
snow_1pt = {}

for exp in experiments:
    for var in ['WS', 'WD']:
        dataframes_1pt = []
        #dataframes_4pts = []
        for year in range(2007, 2023):
            df = pd.read_csv(f'{inpath}/wind/{var}_1pt_{year}_{exp}.csv')
            df['date'] = pd.to_datetime(df['date'])
            dataframes_1pt.append(df)
            #df = pd.read_csv(f'{inpath}/wind/{var}_4pts_2006_{exp}.csv')
            #df['date'] = pd.to_datetime(df['date'])
            #dataframes_4pts.append(df)
        if var == 'WS':
            ws_1pt = pd.concat(dataframes_1pt, ignore_index=True)
            #ws_4pts = pd.concat(dataframes_4pts, ignore_index=True)
        elif var == 'WD':
            wd_1pt = pd.concat(dataframes_1pt, ignore_index=True)
            #wd_4pts = pd.concat(dataframes_4pts, ignore_index=True)
    for var in ['RAINNC']:
        dataframes_1pt = []
        #dataframes_4pts = []
        for year in range(2007, 2023):
            df = pd.read_csv(f'{inpath}/precip/{var}_1pt_{year}_{exp}.csv')
            df = df[['date', 'MG', 'FL', 'OD', 'VS', 'SJ', 'PEAK']]
            #df[['date', 'MG', 'FL', 'OD', 'VS', 'SJ', 'PEAK']].to_csv(f'{inpath}/precip/{var}_1pt_{year}_{exp}.csv')
            df['date'] = pd.to_datetime(df['date'])
            dataframes_1pt.append(df)
        precip_1pt[exp] = pd.concat(dataframes_1pt, ignore_index=True)
    for var in ['SNOWNC']:
        dataframes_1pt = []
        #dataframes_4pts = []
        for year in range(2007, 2023):
            df = pd.read_csv(f'{inpath}/precip/{var}_1pt_{year}_{exp}.csv')
            df = df[['date', 'MG', 'FL', 'OD', 'VS', 'SJ', 'PEAK']]
            #df[['date', 'MG', 'FL', 'OD', 'VS', 'SJ', 'PEAK']].to_csv(f'{inpath}/precip/{var}_1pt_{year}_{exp}.csv')
            df['date'] = pd.to_datetime(df['date'])
            dataframes_1pt.append(df)
        snow_1pt[exp] = pd.concat(dataframes_1pt, ignore_index=True)
#    for var in ['T2']:
#        dataframes_1pt = []
#        #dataframes_4pts = []
#        for year in range(2007, 2023):
#            df = pd.read_csv(f'{inpath}/temp/{var}_1pt_{year}_{exp}.csv')
#            df['date'] = pd.to_datetime(df['date'])
#            dataframes_1pt.append(df)
#        temp_1pt = pd.concat(dataframes_1pt, ignore_index=True)
        
ws_1pt['year'] = ws_1pt['date'].dt.year
ws_1pt['month'] = ws_1pt['date'].dt.month
ws_1pt['day'] = ws_1pt['date'].dt.day

# In[38]:


# Function to classify wind direction
def classify_wd(wd):
    if (0 <= wd < 45) or (315 <= wd <= 360):
        return 'N'
    elif 45 <= wd < 135:
        return 'E'
    elif 135 <= wd < 225:
        return 'S'
    elif 225 <= wd < 315:
        return 'W'
    else:
        return 'Invalid'
    
def classify_wd_v2(wd):
    if (0 <= wd < 90):
        return 'NE'
    elif 90 <= wd < 180:
        return 'SE'
    elif 180 <= wd < 270:
        return 'SW'
    elif 270 <= wd <= 360:
        return 'NW'
    else:
        return 'Invalid'
    
def classify_wd_v3(wd):
    if 360*15/16 <= wd or wd < 360*1/16:
        return 'N'
    elif 360*1/16 <= wd < 360*3/16:
        return 'NE'
    elif 360*3/16 <= wd < 360*5/16:
        return 'E'
    elif 360*5/16 <= wd < 360*7/16:
        return 'SE'
    elif 360*7/16 <= wd < 360*9/16:
        return 'S'
    elif 360*9/16 <= wd < 360*11/16:
        return 'SW'
    elif 360*11/16 <= wd < 360*13/16:
        return 'W'
    elif 360*13/16 <= wd < 360*15/16:
        return 'NW'
    else:
        return 'Invalid'
    

wind = wd_1pt[['date', 'PEAK']]
wind.rename(columns={'PEAK': 'wd'}, inplace=True)
wind = pd.merge(wind, ws_1pt[['date', 'PEAK']], on='date', how='inner')
wind.rename(columns={'PEAK': 'ws'}, inplace=True)
wind = pd.merge(wind, precip_1pt[exp1], on='date', how='inner')

# apply wd classes
wind['class'] = wind['wd'].apply(classify_wd)

for loc in precip_1pt[exp].columns[1:]:
    wind[loc][np.where(wind[loc]<wind[loc][0])[0][0]] = np.nan
    wind[loc] = wind[loc].diff()
    wind.loc[wind[loc]<0, loc] = np.nan # remove negative values (that arise due to restart?)
    wind.loc[wind[loc]>100, loc] = np.nan # remove unrealistically high values (that arise due to restart?)


# In[39]:


mask = MET_hourly_precip['station_id'] == 'SN55430'
precip_MG = pd.merge(MET_hourly_precip[mask], wind[['date', 'class']], on='date', how='inner')

mask = MET_hourly_precip['station_id'] == 'SN55820'
precip_FL = pd.merge(MET_hourly_precip[mask], wind[['date', 'class']], on='date', how='inner')

c = plt.color_sequences["tab10"]

fig, ax = plt.subplots(2,2, figsize=(12,8))
ax = ax.flatten()

for a, cl in enumerate(['N', 'E', 'S', 'W']):
#for a, cl in enumerate(['NW', 'NE', 'SW', 'SE']):
    mask = precip_FL['class'] == cl
    ax[a].scatter(precip_FL[mask]['date'], precip_FL[mask]['precip'], 
                  s=40*precip_FL[mask]['precip']/precip_FL[mask]['precip'].max(), 
                  c=c[1], label='FL')
    mask = precip_MG['class'] == cl
    ax[a].scatter(precip_MG[mask]['date'], precip_MG[mask]['precip'], 
                  s=40*precip_MG[mask]['precip']/precip_MG[mask]['precip'].max(), 
                  c=c[0], label='MG')
    ax[a].set_ylim(0,15)
    #ax[a].set_xticks(np.arange(2008,2023,4))
    ax[a].set_title(f'Wind direction from {cl}')
for a in [0,2]:
    ax[a].set_ylabel('Hourly precipitation (mm)')
for a in [1,3]:
    ax[a].set_yticklabels([], visible=False)
for a in [0,1]:
    ax[a].set_xticklabels([], visible=False)
for a in [2,3]:
    ax[a].set_xlabel('Time')
ax[1].legend()
plt.tight_layout()
plt.show()


# In[40]:


c = plt.color_sequences["tab10"]

fig, ax = plt.subplots(2,2, figsize=(12,8))
ax = ax.flatten()

for a, cl in enumerate(['N', 'E', 'W', 'S']):
#for a, cl in enumerate(['NW', 'NE', 'SW', 'SE']):
    mask = wind['class'] == cl
    for i, loc in enumerate(precip_1pt[exp].columns[1:]):
        ax[a].scatter(wind[mask]['date'], wind[mask][loc], 
                      s=40*wind[mask][loc]/wind[mask][loc].max(), 
                      c=c[i], label=loc)
    ax[a].set_ylim(0,103)
    #ax[a].set_xticks(np.arange(2008,2023,4))
    ax[a].set_title(f'Wind direction from {cl}')
for a in [0,2]:
    ax[a].set_ylabel('Hourly precipitation (mm)')
for a in [1,3]:
    ax[a].set_yticklabels([], visible=False)
for a in [0,1]:
    ax[a].set_xticklabels([], visible=False)
for a in [2,3]:
    ax[a].set_xlabel('Time')
ax[1].legend()
plt.tight_layout()
plt.show()


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           # In[41]:


# import seaborn as sns

# fig, ax = plt.subplots(2,2, figsize=(12,8))
# ax = ax.flatten()

# for a, cl in enumerate(['N', 'E', 'S', 'W']): #(['NW', 'NE', 'SW', 'SE']):
#     mask = wind['class'] == cl
#     for i, loc in enumerate(precip_1pt.columns[1:]):
#         sns.violinplot(x=wind[mask][loc], ax=ax[a])
#     ax[a].set_xlim(0,5)


# In[12]:


# apply wd classes on observations
#MET_hourly['class'] = MET_hourly['wd'].apply(classify_wd_v2)


# In[ ]:





# In[ ]:





# In[30]:


run_script('loadData.py')

summer = False
winter = False

if summer == True:
    winter = False
    ws_1pt = ws_1pt[ws_1pt['date'].dt.month.isin([6,7,8])]
    wd_1pt = wd_1pt[wd_1pt['date'].dt.month.isin([6,7,8])]
    nve_hourly = nve_hourly[(nve_hourly['date'].dt.month.isin([6,7,8]))]
    MET_hourly = MET_hourly[(MET_hourly['date'].dt.month.isin([6,7,8]))]
    NB_hourly = NB_hourly[(NB_hourly['date'].dt.month.isin([6,7,8]))]
    SM = SM[(SM['date'].dt.month.isin([6,7,8]))]
elif winter == True:
    ws_1pt = ws_1pt[ws_1pt['date'].dt.month.isin([12,1,2])]
    wd_1pt = wd_1pt[wd_1pt['date'].dt.month.isin([12,1,2])]
    nve_hourly = nve_hourly[(nve_hourly['date'].dt.month.isin([12,1,2]))]
    MET_hourly = MET_hourly[(MET_hourly['date'].dt.month.isin([12,1,2]))]
    NB_hourly = NB_hourly[(NB_hourly['date'].dt.month.isin([12,1,2]))]
    SM = SM[(SM['date'].dt.month.isin([12,1,2]))]
#MET_hourly.head(60)


# In[7]:


# plot wind observations as timeseries

fig, ax1 = plt.subplots(figsize=(10,7))

for st in ['AS']:#, 'FL', 'FB', 'SB']:
    lon, lat = WRF1000_ts.loc[WRF1000_ts['station_id']==st, 'station_lon'], WRF1000_ts.loc[WRF1000_ts['station_id']==st, 'station_lat']
    if st == 'FL':
        data = MET_hourly.loc[(MET_hourly['station_id'] == 'SN55820') & (MET_hourly['date'] >= datetime.datetime(years[0],1,1)) & (MET_hourly['date'] < datetime.datetime(years[-1],1,1)), ['date','ws','wd']]
    elif st == 'AS':
        data = nve_hourly.loc[(nve_hourly['station_id'] == st) & (nve_hourly['date'] >= datetime.datetime(years[0],1,1)) & (nve_hourly['date'] < datetime.datetime(years[-1],1,1)), ['date','ws','wd']]
    else:
        data = NB_hourly.loc[(NB_hourly['date'] >= datetime.datetime(years[0],1,1)) & (NB_hourly['date'] < datetime.datetime(years[-1],1,1)), ['date','ws','wd']]

    print (st, (data['ws']).max())
    ax1.scatter(data['date'][::6], data['wd'][::6], s=1, label=st, vmin=0, vmax=7, c=data['ws'][::6])
#    ax1.scatter(data['date'][::1], data['ws'][::1], s=1, vmin=0, vmax=5, label=st)
plt.legend()
#ax1.set_xlim(datetime.datetime(2011,4,24), datetime.datetime(2011,5,10))
#ax1.set_ylim(0,30)
plt.show()


# In[8]:


fig, ax1 = plt.subplots(figsize=(10,7))

for st in ['FL', 'FB']:#, 'AS']:
    #print (wd_1pt['date'][::24], wd_1pt[st][::24])
    ax1.scatter(wd_1pt['date'][::6], wd_1pt[st][::6], s=.5, label=st)
plt.legend()
#ax1.set_xticks([datetime.datetime(2015,7,1)])
#ax1.set_xlim(datetime.datetime(2014,4,1), datetime.datetime(2016,1,1))
plt.show()


# In[14]:


#from matplotlib.patches import Circle

def wind_rose(ws, wd, ax, speed_bins=[0,5,10,20,60], num_dirs=36, cmap='summer', ec='k'):

    direction_bins = (np.linspace(0, 360, num_dirs + 1)) # + 22.5) % 360  # offset for bin edges

    hist, _, _ = np.histogram2d(wd, ws, bins=[direction_bins, speed_bins])#, density=False)
    if hist.sum() != 0:
        hist /= hist.sum()            # Normalize to a fraction of 1
    #print (hist.sum())
    
#    radii = hist.sum(axis=1)
#    radii_ratio = radii / radii.sum()  # Normalize to 1

#    theta_mid = np.deg2rad(direction_bins[:-1] + 180 / num_dirs)

    direction_angles = np.deg2rad(direction_bins)
    width = 2 * np.pi / num_dirs
    #print (counts, radii)

    cmap = plt.get_cmap(cmap)
    norm = plt.cm.colors.BoundaryNorm(speed_bins, plt.colormaps['summer'].N)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='summer')
        
    # Plot each direction segment with wind speed bins
    for i in range(num_dirs):
        for j in range(len(speed_bins)-1):
            ax.bar(direction_angles[i], hist[i, j], width=width,
                   bottom=np.sum(hist[i, :j]),  # stack bars
                   color=sm.to_rgba(speed_bins[j]),#counts[i, j]), 
                   edgecolor='k', linewidth=0.1)#, alpha=0.7)
    
#    circle = Circle((0, 0), radius=max(speed_bins), edgecolor=ec, facecolor='none', linewidth=1.5)
#    ax.add_patch(circle)
    
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)

    # Hide labels
    ax.set_xticklabels([])  # Remove angular ticks
    ax.set_yticklabels([])  # Remove radial ticks
    ax.set_yticks([])  # Remove radial ticks
    #ax.set_ylim(0,0.34)
    
    
    # Set the color of the outer edge of the polar plot
    for spine in ax.spines.values():
        spine.set_visible(True)  # Make the spine visible
        spine.set_edgecolor(ec)  # Set the edge color
        spine.set_linewidth(2)  # Set line width for the edge

    return ax


# In[15]:


def estimate_wind_regime(wd_array):
    
    four_bins = True
    # Initialize a dictionary to hold counts for each regime
    if four_bins == True:
        regime_counts = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
    else:
        regime_counts = {'N': 0, 'NE': 0, 'E': 0, 'SE': 0, 'S': 0, 'SW': 0, 'W': 0, 'NW': 0}
    
    # Count the occurrences of wind directions in each regime
    for wd in wd_array:
        
        if four_bins == True:
            if wd > 315 or wd < 45:  # North
                regime_counts['N'] += 1
            elif 45 <= wd < 135:      # East
                regime_counts['E'] += 1
            elif 135 <= wd < 225:     # South
                regime_counts['S'] += 1
            elif 225 <= wd < 315:     # West
                regime_counts['W'] += 1

        else:
            if 360*15/16 <= wd or wd < 360*1/16:  # North
                regime_counts['N'] += 1
            elif 360*1/16 <= wd < 360*3/16:
                regime_counts['NE'] += 1
            elif 360*3/16 <= wd < 360*5/16:
                regime_counts['E'] += 1
            elif 360*5/16 <= wd < 360*7/16:
                regime_counts['SE'] += 1
            elif 360*7/16 <= wd < 360*9/16:
                regime_counts['S'] += 1
            elif 360*9/16 <= wd < 360*11/16:
                regime_counts['SW'] += 1
            elif 360*11/16 <= wd < 360*13/16:
                regime_counts['W'] += 1
            elif 360*13/16 <= wd < 360*15/16:
                regime_counts['NW'] += 1

    # Determine the dominant regime
    dominant_regime = max(regime_counts, key=regime_counts.get)
    
    return dominant_regime


# In[43]:




# In[64]:


import cmocean
cmap=cmocean.cm.phase

#def wind_color(dominant_regime):  
#    if dominant_regime == 'N':
#        col = cmap(.5)#'darkturquoise'
#    elif dominant_regime == 'E':
#        col = cmap(.25)#'gold'
#    elif dominant_regime == 'S':
#        col = cmap(0)#'orangered'
#    elif dominant_regime == 'W':
#        col = cmap(.75)#'darkslateblue'
#    return col

offset=.1 # should be less than 0.125 / 0.25

def wind_color(dominant_regime): 
    
    four_bins = True
    
    if four_bins == True:
        if dominant_regime == 'N':
            col = cmap(4/8+offset)#'C9'#
        elif dominant_regime == 'E':
            col = 'k'#cmap(2/8+offset)
        elif dominant_regime == 'S':
            col = cmap(0/8+offset)#'C1'#
        elif dominant_regime == 'W':
            col = cmap(6/8+offset)#'C2'#
            
    else:
        if dominant_regime == 'N':
            col = cmap(4/8+offset)
        elif dominant_regime == 'NE':
            col = cmap(3/8+offset)
        elif dominant_regime == 'E':
            col = cmap(2/8+offset)
        elif dominant_regime == 'SE':
            col = cmap(1/8+offset)
        elif dominant_regime == 'S':
            col = cmap(0/8+offset)
        elif dominant_regime == 'SW':
            col = cmap(7/8+offset)
        elif dominant_regime == 'W':
            col = cmap(6/8+offset)
        elif dominant_regime == 'NW':
            col = cmap(5/8+offset)
    return col


# In[85]:


months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
months = np.arange(1,13,1)
num_months = 12
if season == 'DJF':
    months_names = ['Dec', 'Jan', 'Feb']
    months = [12,1,2]
    num_months = 3
elif season == 'JJA':
    months_names = ['Jun', 'Jul', 'Aug']
    months = [6,7,8]
    num_months = 3

regimes = {}

plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(16, num_months, subplot_kw=dict(projection='polar'), figsize=(16, 20))
ax = ax.flatten()

speed_bins = [0, 2.5, 5, 10, 15, 20]

for y in years[:]:
    for m, month in enumerate(months):
        mask = (ws_1pt['year'] == y) & (ws_1pt['month'] == month)
        
        dominant_regime = estimate_wind_regime(wd_1pt[mask]['PEAK'])
        regimes[f'{y}_{month:02}'] = dominant_regime

        sm = wind_rose(ws_1pt[mask]['PEAK'], wd_1pt[mask]['PEAK'], ax[(y-years[0])*num_months+m], speed_bins, ec=wind_color(dominant_regime))

for y, year in enumerate(years):
    ax[y*num_months].set_ylabel(f'{year}', fontsize=12, labelpad=20)

for m, month_name in enumerate(months_names):
    ax[(len(years)-1)*num_months+m].set_xlabel(month_name, fontsize=12)
    

#ax.title('wind roses from model')
plt.show()

#%%

regimes_daily = {}

months = np.arange(1,13,1)
days = np.arange(1,32,1)


speed_bins = [0, 2.5, 5, 10, 15, 20]

for y in years[:]:
    for m, month in enumerate(months):
        for d, day in enumerate(days):
            try:
                date_obj = datetime.datetime(y, month, day)
                mask = (ws_1pt['year'] == y) & (ws_1pt['month'] == month) & (ws_1pt['day'] == day)
            except ValueError:
                continue
        
            dominant_regime = estimate_wind_regime(wd_1pt[mask]['PEAK'])
            regimes_daily[f'{y}_{month:02}_{day:02}'] = dominant_regime



# In[23]:


if exp1 == 'glac2019':
    ice_mask = ((WRF_lu_exp1.values == 24))
    
    thr_ice = np.median(WRF_hgt_exp1.values[ice_mask])
    ice_low_mask = (WRF_hgt_exp1.values < thr_ice)
    ice_high_mask = (WRF_hgt_exp1.values >= thr_ice)
    
    thr_noice = np.median(WRF_hgt_exp1.values[~ice_mask])
    noice_low_mask = (WRF_hgt_exp1.values < thr_noice)
    noice_high_mask = (WRF_hgt_exp1.values >= thr_noice)


#%%

exp = 'glac2019'

t2m_daily = t2m_1pt[exp].groupby(pd.Grouper(key='date', freq='D')).mean().reset_index()
#precip_daily = precip_1pt.groupby(pd.Grouper(key='date', freq='D')).mean().reset_index()

precip_daily = precip_1pt[exp].copy()
precip_daily.set_index('date', inplace=True)
precip_daily = precip_daily[precip_daily.index.hour == 6]

for col in precip_daily.columns:
    precip_daily[col] = precip_daily[col].diff()
    
precip_daily = precip_daily.where((precip_daily >= 0) & (precip_daily <= 140)) # remove a very few outliers that are probably related to restart

precip_daily = precip_daily.reset_index()

for key in regimes_daily.keys():
    date = datetime.datetime(int(key[:4]), int(key[5:7]), int(key[8:10]))
    #mask = t2m_daily['date'] == date 
    t2m_daily.loc[t2m_daily['date'] == date, 'class'] = regimes_daily[key]
    precip_daily.loc[precip_daily['date']-pd.Timedelta(hours=6) == date, 'class'] = regimes_daily[key]
    #print (date, regimes_daily[key])

#%%


t2m_exp2_daily = t2m_1pt[exp2].groupby(pd.Grouper(key='date', freq='D')).mean().reset_index()

precip_exp2_daily = precip_1pt[exp2].copy()
precip_exp2_daily.set_index('date', inplace=True)
precip_exp2_daily = precip_exp2_daily[precip_exp2_daily.index.hour == 6]

for col in precip_exp2_daily.columns:
    precip_exp2_daily[col] = precip_exp2_daily[col].diff()
    
precip_exp2_daily = precip_exp2_daily.where((precip_exp2_daily >= 0) & (precip_exp2_daily <= 140)) # remove a very few outliers that are probably related to restart

precip_exp2_daily = precip_exp2_daily.reset_index()

for key in regimes_daily.keys():
    date = datetime.datetime(int(key[:4]), int(key[5:7]), int(key[8:10]))
    t2m_exp2_daily.loc[t2m_exp2_daily['date'] == date, 'class'] = regimes_daily[key]
    precip_exp2_daily.loc[precip_exp2_daily['date']-pd.Timedelta(hours=6) == date, 'class'] = regimes_daily[key]
    
    
#%%

# shift precip data to the day before for best match with wind class

precip_daily.iloc[:, 1:-1] = precip_daily.iloc[:, 1:-1].shift(-1)
precip_exp2_daily.iloc[:, 1:-1] = precip_exp2_daily.iloc[:, 1:-1].shift(-1)


#%%


fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for i, cl in enumerate(t2m_daily['class']):
    print (wind_color(cl))
    ax.scatter(t2m_daily['OV'][i]-273.15, precip_daily['OD'][i], 
               c=wind_color(cl), s=10)
for cl in ['N', 'E', 'S', 'W']:
#for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), c=wind_color(cl), label=cl)
plt.legend()
ax.set_xlabel('Mean daily temperature (\u00b0C)')
ax.set_ylabel('Mean daily precipitation (mm)')
if season == 'all':
    ax.set_title(f'Mean values at Oldedalen (control experiment)')
else:
    ax.set_title(f'Mean {season} values at Oldedalen (control experiment)')
plt.show()

# In[81]:


fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for key in regimes.keys():
    ax.scatter(WRF_temp_exp1[key][ice_mask].mean()-273.15, WRF_precip_exp1[key][ice_mask].mean(), 
               c=wind_color(regimes[key]), s=60)
for cl in ['N', 'E', 'S', 'W']:
#for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), c=wind_color(cl), label=cl)
plt.legend()
ax.set_xlabel('Mean monthly temperature (\u00b0C)')
ax.set_ylabel('Mean monthly precipitation (mm)')
if season == 'all':
    ax.set_title(f'Mean values over ice cap (control experiment)')
else:
    ax.set_title(f'Mean {season} values over ice cap (control experiment)')
plt.show()


# In[82]:


fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for key in regimes.keys():
    ax.scatter((WRF_temp_exp2[key][ice_mask]-WRF_temp_exp1[key][ice_mask]).mean(), 
               (WRF_precip_exp2[key][ice_mask]-WRF_precip_exp1[key][ice_mask]).mean(), 
               c=wind_color(regimes[key]), s=60)
for cl in ['N', 'E', 'S', 'W']:
#for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), c=wind_color(cl), label=cl)
plt.legend()
ax.set_xlabel('Difference in mean monthly temperature (\u00b0C)')
ax.set_ylabel('Difference in mean monthly precipitation (mm)')
if season == 'all':
    ax.set_title(f'Difference in temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')
else:
    ax.set_title(f'Difference in {season} temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')
plt.show()


# In[67]:


if season == 'all':
    data = pd.merge(MET_monthly.loc[MET_monthly['station'] == 'OD'], nve_monthly.loc[nve_monthly['station_id'] == 'OV', ['date', 't']], on='date', how='inner')
elif season == 'DJF':
    data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'OD')&(MET_monthly['month'].isin([12,1,2]))], nve_monthly.loc[(nve_monthly['station_id'] == 'OV') & (nve_monthly['month'].isin([12,1,2])), ['date', 't']], on='date', how='inner')
elif season == 'JJA':
    data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'OD')&(MET_monthly['month'].isin([6,7,8]))], nve_monthly.loc[(nve_monthly['station_id'] == 'OV') & (nve_monthly['month'].isin([6,7,8])), ['date', 't']], on='date', how='inner')
data = pd.merge(data, wind[['date', 'class']], on='date', how='inner')

grid_i = WRF1000_ts.loc[WRF1000_ts['station_id']=='OD', 'grid_i']
grid_j = WRF1000_ts.loc[WRF1000_ts['station_id']=='OD', 'grid_j']

fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for key in regimes.keys():
    ax.scatter(WRF_temp_exp1[key][grid_j, grid_i]-273.15, 
               WRF_precip_exp1[key][grid_j, grid_i], 
               c=wind_color(regimes[key]), s=60)
for i, cl in enumerate(data['class']):
    ax.scatter(data['t'][i], data['precip'][i], c='none', ec=wind_color(cl), marker='o', s=90)
for cl in ['N', 'E', 'S', 'W']:
#for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), c=wind_color(cl), label=cl)
plt.legend()
ax.set_xlabel('Mean monthly temperature (\u00b0C)')
ax.set_ylabel('Mean monthly precipitation (mm)')
if season == 'all':
    ax.set_title(f'Mean values in Oldedalen (control experiment and observations)')
else:
    ax.set_title(f'Mean {season} values in Oldedalen (control experiment and observations)')
plt.show()


# In[68]:


if season == 'all':
    data = pd.merge(MET_monthly.loc[MET_monthly['station'] == 'FL'], wind[['date', 'class']], on='date', how='inner')
elif season == 'DJF':
    data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'FL')&(MET_monthly['month'].isin([12,1,2]))], wind[['date', 'class']], on='date', how='inner')
elif season == 'JJA':
    data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'FL')&(MET_monthly['month'].isin([6,7,8]))], wind[['date', 'class']], on='date', how='inner')
    
grid_i = WRF1000_ts.loc[WRF1000_ts['station_id']=='FL', 'grid_i']
grid_j = WRF1000_ts.loc[WRF1000_ts['station_id']=='FL', 'grid_j']

fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for key in regimes.keys():
    ax.scatter(WRF_temp_exp1[key][grid_j, grid_i]-273.15, 
               WRF_precip_exp1[key][grid_j, grid_i], 
               c=wind_color(regimes[key]), s=60)
for i, cl in enumerate(data['class']):
    ax.scatter(data['temp'][i], data['precip'][i], c='none', ec=wind_color(cl), marker='o', s=90)
for cl in ['N', 'E', 'S', 'W']:
#for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), c=wind_color(cl), label=cl)
plt.legend()
ax.set_xlabel('Mean monthly temperature (\u00b0C)')
ax.set_ylabel('Mean monthly precipitation (mm)')
if season == 'all':
    ax.set_title(f'Mean {season} values in Fjærland (control experiment and observations)')
else:
    ax.set_title(f'Mean {season} values in Fjærland (control experiment and observations)')
plt.show()


# In[71]:
    
loc = 'SJ'#'OD'#
mos = ([12, 1, 2])
#mos = ([6, 7, 8])
mos = list(np.arange(1,13,1))
    
if loc == 'PEAK':
    xmin = -20
    xverymin = -30
    xrange = 40
    xmax = 10
elif loc == 'FL' or loc == 'SJ':
    xmin = -20
    xverymin = -25
    xrange = 50
    xmax = 20
elif loc == 'OD':
    xmin = -20
    xverymin = -25
    xrange = 50
    xmax = 20

# calculate trend
def trend(x_values, y_values):
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    slope, intercept = np.polyfit(x_values, y_values, 1)  # 1 for linear fit
    
    trend_line_x = np.linspace(x_values.min(), x_values.max(), 100)  # 100 points for smooth line
    trend_line_y = slope * trend_line_x + intercept
    
    return trend_line_x, trend_line_y

    
x_values = []
y_values = []    
x_values_N = []
y_values_N = []  
x_values_E = []
y_values_E = []  
x_values_S = []
y_values_S = []  
x_values_W = []
y_values_W = []  

fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for i, cl in enumerate(t2m_daily['class']):
    if precip_daily['date'][i].month in mos:
        # Append the x and y values for the trend line
        x = t2m_daily[loc][i] - 273.15  # Convert to Celsius
        y = t2m_exp2_daily[loc][i] - t2m_daily[loc][i]
        x_values.append(x)
        y_values.append(y)
        if cl == 'N':
            x_values_N.append(x)
            y_values_N.append(y)
        if cl == 'E':
            x_values_E.append(x)
            y_values_E.append(y)
        if cl == 'S':
            x_values_S.append(x)
            y_values_S.append(y)
        if cl == 'W':
            x_values_W.append(x)
            y_values_W.append(y)
        
        os = 0
        if cl == 'E':
            os = xrange
        elif cl == 'S':
            os = 2*xrange
        elif cl == 'W':
            os = 3*xrange
            
        ax.scatter(x+os, y, c=wind_color(cl), s=10)
    
    
ax.set_xticks(np.arange(xmin, 4*xrange+xmin, 10))
ax.set_xticklabels(4*list(np.arange(xmin, xmax+1, 10)))

if loc == 'PEAK':
    for label in (ax.xaxis.get_ticklabels()[3::4]):
        label.set_visible(False)
        
# for x_values, y_values, col in zip([x_values, x_values_N, x_values_E, x_values_S, x_values_W], [y_values, y_values_N, y_values_E, y_values_S, y_values_W], ['grey', wind_color('N'), wind_color('E'), wind_color('S'), wind_color('W')]):
#     trend_x, trend_y = trend(x_values, y_values)
#     ax.plot(trend_x, trend_y, color=col, linestyle='--', zorder=10000)

#ax.set_ylim(0,5)

# for i, cl in enumerate(t2m_daily['class']):
#     #print (wind_color(cl))
#     ax.scatter(t2m_daily['PEAK'][i]-273.15, 
#                t2m_exp2_daily['PEAK'][i] - t2m_daily['PEAK'][i], 
#                c=wind_color(cl), s=10)
#for key in regimes.keys():
#    ax.scatter((WRF_temp_exp1[key][ice_mask]).mean()-273.15, 
#               (WRF_temp_exp2[key][ice_mask]-WRF_temp_exp1[key][ice_mask]).mean(), 
#               c=wind_color(regimes[key]), s=60)



ax.axhline(y=np.mean(y_values_N), c=wind_color('N'), ls='--', xmin=0, xmax=.25)
ax.axhline(y=np.mean(y_values_E), c=wind_color('E'), ls='--', xmin=.25, xmax=.5)
ax.axhline(y=np.mean(y_values_S), c=wind_color('S'), ls='--', xmin=.5, xmax=.75)
ax.axhline(y=np.mean(y_values_W), c=wind_color('W'), ls='--', xmin=.75, xmax=1)

ax.axvline(x=np.mean(x_values_N)+0*xrange, c=wind_color('N'), ls='--')
ax.axvline(x=np.mean(x_values_E)+1*xrange, c=wind_color('E'), ls='--')
ax.axvline(x=np.mean(x_values_S)+2*xrange, c=wind_color('S'), ls='--')
ax.axvline(x=np.mean(x_values_W)+3*xrange, c=wind_color('W'), ls='--')

for cl in ['N', 'E', 'S', 'W']:
#for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), c=wind_color(cl), label=cl)
#plt.legend()
if loc == 'PEAK':
    ytext = -11.5
elif loc == 'FL' or loc == 'SJ':
    ytext = -11.5
elif loc == 'OD':
    ytext = -21.5
    
# plt.text(xverymin+3+0*xrange, ytext, f'N: {np.mean(x_values_N):.1f}, {np.mean(y_values_N):.1f}', c=wind_color('N'), fontsize=26)
# plt.text(xverymin+3+1*xrange, ytext, f'E: {np.mean(x_values_E):.1f}, {np.mean(y_values_E):.1f}', c=wind_color('E'), fontsize=26)
# plt.text(xverymin+3+2*xrange, ytext, f'S: {np.mean(x_values_S):.1f}, {np.mean(y_values_S):.1f}', c=wind_color('S'), fontsize=26)
# plt.text(xverymin+3+3*xrange, ytext, f'W: {np.mean(x_values_W):.1f}, {np.mean(y_values_W):.1f}', c=wind_color('W'), fontsize=26)

plt.text(xverymin+3+0*xrange, ytext, f'N', c=wind_color('N'), fontsize=26)
plt.text(xverymin+3+1*xrange, ytext, f'E', c=wind_color('E'), fontsize=26)
plt.text(xverymin+3+2*xrange, ytext, f'S', c=wind_color('S'), fontsize=26)
plt.text(xverymin+3+3*xrange, ytext, f'W', c=wind_color('W'), fontsize=26)

ax.set_xlabel('Mean daily temperature for control experiment (\u00b0C)')
ax.set_ylabel('Difference in mean daily temperature (K)')

loclabel = loc
if loc == 'PEAK':
    loclabel = 'peak of ice cap'

if season == 'all':
#    ax.set_title(f'Difference in temperature vs control temperature at peak of ice cap\n for different wind regimes (no ice volume - control)')
    if exp2 == 'noice_BT':
        ax.set_title(f'Temperature at {loclabel} (no-ice-volume vs. control)')
    elif exp2 == 'noice_dtm50':
        ax.set_title(f'Temperature at {loclabel} (no-ice-surface vs. control)')
#else:
#    ax.set_title(f'Difference in {season} temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')

right_edge = ax.get_xlim()[1]

ax.axhline(0, c='grey', alpha=0.5, zorder=-100)    
#ax.axvspan(0, right_edge, color='grey', alpha=.2, zorder=-1000)
ax.set_xlim(xmax=right_edge)

ax.set_xlim(xverymin, 4*xrange+xverymin)
ax.set_ylim(-12, 17)

for tick in np.arange(xverymin, 4*xrange, xrange):
    plt.axvline(x=tick, color='gray', linestyle='-', alpha=1)#0.5)
    
# if loc == 'PEAK': 
#     for tick in np.arange(-10, 11, 10):
#         plt.axhline(y=tick, color='gray', linestyle='-', alpha=0.5)
# if loc == 'FL': 
#     for tick in np.arange(-20, 11, 10):
#         plt.axhline(y=tick, color='gray', linestyle='-', alpha=0.5)
# if loc == 'OD': 
#     for tick in np.arange(-20, 11, 10):
#         plt.axhline(y=tick, color='gray', linestyle='-', alpha=0.5)

plt.show()


#%%

loc = 'SJ'#'OD'#'FL'#

mos = ([12, 1, 2])
mos = ([6, 7, 8])
mos = list(np.arange(1,13,1))
    
if loc == 'PEAK':
    xmin = 0
    xrange = 140
elif loc == 'FL' or loc == 'SJ':
    xmin = 0
    xrange = 90
elif loc == 'OD':
    xmin = 0
    xrange = 140

x_values = []
y_values = []    
x_values_N = []
y_values_N = []  
x_values_E = []
y_values_E = []  
x_values_S = []
y_values_S = []  
x_values_W = []
y_values_W = []  

fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for i, cl in enumerate(precip_daily['class']):
    if precip_daily['date'][i].month in mos:
        if precip_daily[loc][i] >= 0:
            # Append the x and y values for the trend line
            x = precip_daily[loc][i]  # Convert to Celsius
            y = precip_exp2_daily[loc][i] - precip_daily[loc][i]
            x_values.append(x)
            y_values.append(y)
            if cl == 'N':
                x_values_N.append(x)
                y_values_N.append(y)
            if cl == 'E':
                x_values_E.append(x)
                y_values_E.append(y)
            if cl == 'S':
                x_values_S.append(x)
                y_values_S.append(y)
            if cl == 'W':
                x_values_W.append(x)
                y_values_W.append(y)
            
            os = 0
            if cl == 'E':
                os = xrange
            elif cl == 'S':
                os = 2*xrange
            elif cl == 'W':
                os = 3*xrange
                
            ax.scatter(x+os, y, c=wind_color(cl), s=10)
#ax.set_xticks(np.arange(30, 4*140+1, 140))
ax.set_xticks(np.arange(xmin, 4*xrange, int(xrange/2)))
#ax.set_xticklabels(['N', 'E', 'S', 'W'])
ax.set_xticklabels(4*list(np.arange(xmin, int(xrange/2)+1,int(xrange/2))))
    
# if loc == 'PEAK':
#     for label in (ax.xaxis.get_ticklabels()[3::4]):
#         label.set_visible(False)

# for x_values, y_values, col in zip([x_values, x_values_N, x_values_E, x_values_S, x_values_W], [y_values, y_values_N, y_values_E, y_values_S, y_values_W], ['grey', wind_color('N'), wind_color('E'), wind_color('S'), wind_color('W')]):
#     trend_x, trend_y = trend(x_values, y_values)
#     ax.plot(trend_x, trend_y, color=col, linestyle='--', zorder=10000)

#ax.set_ylim(0,5)

# for i, cl in enumerate(t2m_daily['class']):
#     #print (wind_color(cl))
#     ax.scatter(t2m_daily['PEAK'][i]-273.15, 
#                t2m_exp2_daily['PEAK'][i] - t2m_daily['PEAK'][i], 
#                c=wind_color(cl), s=10)
#for key in regimes.keys():
#    ax.scatter((WRF_temp_exp1[key][ice_mask]).mean()-273.15, 
#               (WRF_temp_exp2[key][ice_mask]-WRF_temp_exp1[key][ice_mask]).mean(), 
#               c=wind_color(regimes[key]), s=60)


ax.axhline(y=np.nanmean(y_values_N), c=wind_color('N'), ls='--', xmin=0, xmax=.25)
ax.axhline(y=np.nanmean(y_values_E), c=wind_color('E'), ls='--', xmin=.25, xmax=.5)
ax.axhline(y=np.nanmean(y_values_S), c=wind_color('S'), ls='--', xmin=.5, xmax=.75)
ax.axhline(y=np.nanmean(y_values_W), c=wind_color('W'), ls='--', xmin=.75, xmax=1)

ax.axvline(x=np.nanmean(x_values_N)+0*xrange, c=wind_color('N'), ls='--')
ax.axvline(x=np.nanmean(x_values_E)+1*xrange, c=wind_color('E'), ls='--')
ax.axvline(x=np.nanmean(x_values_S)+2*xrange, c=wind_color('S'), ls='--')
ax.axvline(x=np.nanmean(x_values_W)+3*xrange, c=wind_color('W'), ls='--')

for cl in ['N', 'E', 'S', 'W']:
#for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), s=50, c=wind_color(cl), label=cl)
#plt.legend(ncols=4, columnspacing=5.5, loc='lower center')

if loc == 'PEAK':
    ytext = -86#-111
elif loc == 'FL' or loc == 'SJ':
    ytext = -87
elif loc == 'OD':
    ytext = -87

plt.text(xmin+20, ytext, 'N', c=wind_color('N'), fontsize=26)
plt.text(xmin+20+xrange, ytext, 'E', c=wind_color('E'), fontsize=26)
plt.text(xmin+20+2*xrange, ytext, 'S', c=wind_color('S'), fontsize=26)
plt.text(xmin+20+3*xrange, ytext, 'W', c=wind_color('W'), fontsize=26)

ax.set_xlabel('Daily precipitation for control experiment (mm)')
ax.set_ylabel('Difference in daily precipitation (mm)')

loclabel = loc
if loc == 'PEAK':
    loclabel = 'peak of ice cap'

if season == 'all':
#    ax.set_title(f'Difference in precipitation vs control precipitation at peak of ice cap\n for different wind regimes (no ice volume - control)')
    if exp2 == 'noice_BT':
        ax.set_title(f'Precipitation at {loclabel} (no-ice-volume vs. control)')
    elif exp2 == 'noice_dtm50':
        ax.set_title(f'Precipitation at {loclabel} (no-ice-surface vs. control)')
#else:
#    ax.set_title(f'Difference in {season} temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')

right_edge = ax.get_xlim()[1]

ax.axhline(0, c='grey', zorder=-100)    
#ax.axvspan(0, right_edge, color='grey', alpha=.2, zorder=-1000)
ax.set_xlim(xmax=right_edge)


#ax.set_xlim(-2,140)
ax.set_xlim(xmin-1,4*xrange+1)

if loc == 'PEAK':
    ax.set_ylim(-90,90) #-115
elif loc == 'FL' or loc == 'OD' or loc == 'SJ':
    ax.set_ylim(-90,90)#50)

# 6 points excluded from this range

ax.set_yticks(np.arange(-80,81,40))

for tick in np.arange(xmin, 4*xrange, xrange):
    plt.axvline(x=tick, color='gray', linestyle='-', alpha=1, zorder=-100)#0.5)
    
# if loc == 'PEAK':
#     for tick in np.arange(-100, 100, 50):
#         plt.axhline(y=tick, color='gray', linestyle='-', alpha=0.5)
# elif loc == 'FL' or loc == 'OD':
#     for tick in np.arange(-80, 50, 40):
#         plt.axhline(y=tick, color='gray', linestyle='-', alpha=0.5)                

plt.show()



# In[71]:


fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for key in regimes.keys():
    ax.scatter((WRF_precip_exp1[key][ice_mask]).mean(), 
               (WRF_precip_exp2[key][ice_mask]-WRF_precip_exp1[key][ice_mask]).mean(), 
               c=wind_color(regimes[key]), s=60)
for cl in ['N', 'E', 'S', 'W']:
#for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), c=wind_color(cl), label=cl)
plt.legend()

ax.set_xlabel('Mean monthly precipitation for control experiment (mm)')
ax.set_ylabel('Difference in mean monthly precipitation (mm)')

if season == 'all':
    ax.set_title(f'Difference in precipitation vs control precipitation over ice cap\n for different wind regimes (no ice volume - control)')
#else:
#    ax.set_title(f'Difference in {season} temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')

right_edge = ax.get_xlim()[1]
    
#ax.axvspan(0, right_edge, color='grey', alpha=.2, zorder=-1000)
ax.set_xlim(xmax=right_edge)

plt.show()

# In[71]:


fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for key in regimes.keys():
    ax.scatter((WRF_temp_exp1[key][ice_mask]).mean()-273.15, 
               (WRF_precip_exp2[key][ice_mask]-WRF_precip_exp1[key][ice_mask]).mean(), 
               c=wind_color(regimes[key]), s=60)
for cl in ['N', 'E', 'S', 'W']:
#for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), c=wind_color(cl), label=cl)
plt.legend()

ax.set_xlabel('Mean monthly temperature for control experiment (\u00b0C)')
ax.set_ylabel('Difference in mean monthly precipitation (mm)')

if season == 'all':
    ax.set_title(f'Difference in precipitation vs temperature over ice cap\n for different wind regimes (no ice volume - control)')
else:
    ax.set_title(f'Difference in {season} temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')

right_edge = ax.get_xlim()[1]
    
ax.axvspan(0, right_edge, color='grey', alpha=.2, zorder=-1000)
ax.set_xlim(xmax=right_edge)

plt.show()


# In[72]:


fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for key in regimes.keys():
    ax.scatter((WRF_temp_exp2[key][ice_mask]-WRF_temp_exp1[key][ice_mask]).mean(), 
               (WRF_precip_exp1[key][ice_mask]).mean(), 
               c=wind_color(regimes[key]), s=60)
for cl in ['N', 'E', 'S', 'W']:
#for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), c=wind_color(cl), label=cl)
plt.legend()
ax.set_xlabel('Difference in mean monthly temperature (\u00b0C)')
ax.set_ylabel('Mean monthly precipitation (mm)')
if season == 'all':
    ax.set_title(f'Difference in temperature vs precipitation over ice cap\n for different wind regimes (no ice volume - control)')
else:
    ax.set_title(f'Difference in {season} temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')
plt.show()


# In[ ]:


if season == 'all':
    data = pd.merge(MET_monthly.loc[MET_monthly['station'] == 'OD'], nve_monthly.loc[nve_monthly['station_id'] == 'OV', ['date', 't']], on='date', how='inner')
elif season == 'DJF':
    data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'OD')&(MET_monthly['month'].isin([12,1,2]))], nve_monthly.loc[(nve_monthly['station_id'] == 'OV') & (nve_monthly['month'].isin([12,1,2])), ['date', 't']], on='date', how='inner')
elif season == 'JJA':
    data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'OD')&(MET_monthly['month'].isin([6,7,8]))], nve_monthly.loc[(nve_monthly['station_id'] == 'OV') & (nve_monthly['month'].isin([6,7,8])), ['date', 't']], on='date', how='inner')
data = pd.merge(data, wind[['date', 'class']], on='date', how='inner')

data#['t'], data['precip']


# In[ ]:


grid_i = WRF1000_ts.loc[WRF1000_ts['station_id']=='OD', 'grid_i']
grid_j = WRF1000_ts.loc[WRF1000_ts['station_id']=='OD', 'grid_j']

fig, ax = plt.subplots(figsize=(14,10))
plt.rcParams.update({'font.size': 22})

for key in regimes.keys():
    ax.scatter(WRF_temp_exp1[key][grid_j, grid_i]-273.15, 
               WRF_precip_exp1[key][grid_j, grid_i], 
               c=wind_color(regimes[key]), s=60)
for i, cl in enumerate(data['class']):
    ax.scatter(data['t'][i], data['precip'][i], c='none', ec=wind_color(cl), marker='o', s=90)
#for cl in ['N', 'E', 'S', 'W']:
for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
    ax.scatter((), (), c=wind_color(cl), label=cl)
plt.legend()
ax.set_xlabel('Mean monthly temperature (\u00b0C)')
ax.set_ylabel('Mean monthly precipitation (mm)')
if season == 'all':
    ax.set_title(f'Mean values in Oldedalen (control experiment and observations)')
else:
    ax.set_title(f'Mean {season} values in Oldedalen (control experiment and observations)')
plt.show()


# In[ ]:





# In[ ]:





# In[24]:


clo = 7
cla = 62

plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(16, 20))
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=clo, central_latitude=cla))

ax.set_extent([6.3, 7.7, 61.25, 62.05], crs=ccrs.PlateCarree())

ax.contourf(WRF_lon, WRF_lat, WRF_hgt_exp1, cmap='Greys_r', alpha=.5, levels=15, transform=ccrs.PlateCarree())



speed_bins = [0, 2.5, 5, 10, 15]

c = [u'#7f7f7f', u'#17becf', 'k', 'navy', u'#e377c2', u'#9467bd', 'k', u'#2ca02c', u'#ff7f0e', 'k', 'k', 'k', u'#d62728', 'k', 'k', u'#1f77b4', 'aquamarine', u'#8c564b', 'lightgrey']

for s, st in enumerate(stations):
    if st in ['FL_ol', 'NV', 'BH', 'JD', 'SV', 'FN']: #SM
        continue
    
    print (st)
    
    #else:#if st == 'AS':
    lon, lat = WRF1000_ts.loc[WRF1000_ts['station_id']==st, 'grid_lon'], WRF1000_ts.loc[WRF1000_ts['station_id']==st, 'grid_lat']
    ax.plot(lon, lat, marker='X', color='k', markersize=25, transform=ccrs.PlateCarree())#LambertConformal(central_longitude=clo, central_latitude=cla))#

    display_coords = ax.transData.transform(ax.projection.transform_point(lon, lat, ccrs.PlateCarree()))

    # Fractional position in figure coordinates
    fig_coords = fig.transFigure.inverted().transform(display_coords)

    # Set a smaller inset axes size
    inset_width = 0.08  # Width as a fraction of figure width
    inset_height = 0.08  # Height as a fraction of figure height

    # Create polar inset axes
    if st != 'NB':
        inset_ax = fig.add_axes([fig_coords[0] - inset_width / 2, fig_coords[1] - inset_height / 2,
                                 inset_width, inset_height], projection='polar', label=st, zorder=100)
    else:
        inset_ax = fig.add_axes([fig_coords[0] - inset_width / 2, fig_coords[1] - inset_height / 2,
                                 inset_width, inset_height], projection='polar', label=st, zorder=1000)
        inset_ax.set_facecolor((1, 1, 1, 0.7))  # Set the background color with alpha

    sm = wind_rose(ws_1pt[st], wd_1pt[st], inset_ax, speed_bins, cmap='summer', ec=c[s])
    

glaciers.to_crs(epsg=4326).plot(ax=ax, color='w', edgecolor='k', transform=ccrs.PlateCarree(), zorder=90)

#ax.add_feature(cfeature.LAND)
#ax.add_feature(cfeature.OCEAN)
#ax.add_feature(cfeature.COASTLINE)
    
gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray')
gl.top_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.right_labels = False

gl.xlocator = plt.MaxNLocator(nbins=4)  # Control number of x-ticks
gl.ylocator = plt.MaxNLocator(nbins=4)  # Control number of y-ticks

gl.ylocator = mticker.FixedLocator(np.arange(61.3,61.95,.2))

ax.set_aspect('equal')  # or use 'equal' as required for your requirement
inset_ax.set_aspect('equal')  # or use 'equal' as required for your requirement


#ax.title('wind roses from model')


# In[56]:


#run_script('loadData.py')
SM.loc[SM['wd']<0, ['ws','wd']] = np.nan
SM.loc[SM['ws']<1, 'wd'] = np.nan
SM.loc[SM['ws']==4.4, ['ws','wd']] = np.nan

fig = plt.subplots(figsize=(15,10))
plt.scatter(SM['date'], SM['wd'], s=1)
#plt.gca().set_ylim(0,360)


# In[75]:


clo = 7
cla = 62

plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(16, 20))
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=clo, central_latitude=cla))

ax.set_extent([6.3, 7.7, 61.25, 62.05], crs=ccrs.PlateCarree())

ax.contourf(WRF_lon, WRF_lat, WRF_hgt_exp1, cmap='Greys_r', alpha=.5, levels=15, transform=ccrs.PlateCarree())


speed_bins = [0, 2.5, 5, 10, 15]


c = [u'#1f77b4', 'navy', u'#d62728', u'#8c564b', 'k', u'#17becf']


for s, st in enumerate(['FB', 'FL', 'AS', 'SB', 'SM', 'NB']):
    lon, lat = WRF1000_ts.loc[WRF1000_ts['station_id']==st, 'station_lon'], WRF1000_ts.loc[WRF1000_ts['station_id']==st, 'station_lat']
    if st == 'FL':
        data = MET_hourly.loc[(MET_hourly['station_id'] == 'SN55820') & (MET_hourly['date'] >= datetime.datetime(years[0],1,1)) & (MET_hourly['date'] < datetime.datetime(years[-1]+1,1,1)), ['date','ws','wd']]
    elif st == 'SB':
        data = MET_hourly.loc[(MET_hourly['station_id'] == 'SN55425') & (MET_hourly['date'] >= datetime.datetime(years[0],1,1)) & (MET_hourly['date'] < datetime.datetime(years[-1]+1,1,1)), ['date','ws','wd']]        
    elif st == 'NB':
        data = NB_hourly.loc[(NB_hourly['date'] >= datetime.datetime(years[0],1,1)) & (NB_hourly['date'] < datetime.datetime(years[-1],1,1)), ['date','ws','wd']]
    elif st == 'SM':
        data = SM.loc[(SM['date'] >= datetime.datetime(years[0],1,1)) & (SM['date'] < datetime.datetime(years[-1],1,1)), ['date','ws','wd']]
    else:
        data = nve_hourly.loc[(nve_hourly['station_id'] == st) & (nve_hourly['date'] >= datetime.datetime(years[0],1,1)) & (nve_hourly['date'] < datetime.datetime(years[-1],1,1)), ['date','ws','wd']]
    
    #ax.plot(lon, lat, marker='X', color='k', markersize=25, transform=ccrs.PlateCarree())#LambertConformal(central_longitude=clo, central_latitude=cla))#
    
    
    display_coords = ax.transData.transform(ax.projection.transform_point(lon, lat, ccrs.PlateCarree()))

    # Fractional position in figure coordinates
    fig_coords = fig.transFigure.inverted().transform(display_coords)
    
    # Set a smaller inset axes size
    inset_width = 0.08  # Width as a fraction of figure width
    inset_height = 0.08  # Height as a fraction of figure height
    
    # Create polar inset axes
    inset_ax = fig.add_axes([fig_coords[0] - inset_width / 2, fig_coords[1] - inset_height / 2,
                             inset_width, inset_height], projection='polar', label=st)
    if st == 'NB':
        inset_ax.set_facecolor((1, 1, 1, 0.7))  # Set the background color with alpha
        
    
    sm = wind_rose(data['ws'], data['wd'], inset_ax, speed_bins, cmap='summer_r', ec=c[s])
    
    
#ax.title('wind roses from observations')


glaciers.to_crs(epsg=4326).plot(ax=ax, color='w', edgecolor='k', transform=ccrs.PlateCarree(), zorder=90)

#ax.add_feature(cfeature.LAND)
#ax.add_feature(cfeature.OCEAN)
#ax.add_feature(cfeature.COASTLINE)
    
gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray')
gl.top_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.right_labels = False

gl.xlocator = plt.MaxNLocator(nbins=4)  # Control number of x-ticks
gl.ylocator = plt.MaxNLocator(nbins=4)  # Control number of y-ticks

gl.ylocator = mticker.FixedLocator(np.arange(61.3,61.95,.2))

ax.set_aspect('equal')  # or use 'equal' as required for your requirement
inset_ax.set_aspect('equal')  # or use 'equal' as required for your requirement


# In[ ]:
    
    
    
# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


##SM_2015 = SM.loc[(SM['date']>=datetime.datetime(2019,12,28))&(SM['date']<datetime.datetime(2019,12,31))]
SM_2019 = SM.loc[(SM['date']>=datetime.datetime(2019,12,28))&(SM['date']<datetime.datetime(2019,12,31))]
SM_2020 = SM.loc[(SM['date']>=datetime.datetime(2020,11,17))&(SM['date']<datetime.datetime(2020,11,20))]
SM_2022 = SM.loc[(SM['date']>=datetime.datetime(2022,11,9))&(SM['date']<datetime.datetime(2022,11,12))]


# In[36]:


#plt.scatter(SM_2019['date'].values, SM_2019['ws'].values)
#plt.scatter(SM_2020['date'].values, SM_2020['ws'])
#plt.scatter(SM_2022['date'].values, SM_2022['ws'])
#plt.scatter(SM_2019['date'].values, SM_2019['wd'].values)
#plt.scatter(SM_2020['date'].values, SM_2020['wd'])
#plt.scatter(SM_2022['date'].values, SM_2022['wd'])


# In[39]:


# test of some case studies with extreme precipitation

year = 2022; month = 11; day = 11; obs = {'OD': 51.2, 'VS': 51.7, 'FL': 67, 'MG': 37.8, 'JD': np.nan, 'SJ': 39}      # medium wind from SW
#year = 2020; month = 11; day = 19; obs = {'OD': 96, 'VS': 61.5, 'FL': 57.9, 'MG': np.nan, 'JD': 40.6, 'SJ': 65.5}   # low to medium wind from S to W to N (SW)
year = 2019; month = 12; day = 30; obs = {'OD': 64.2, 'VS': 67.2, 'FL': 81.4, 'MG': np.nan, 'JD': 35.6, 'SJ': 69.1} # medium to high, from S to W (W-SW)
#year = 2015; month = 3; day = 8; obs = {'OD': 83.5, 'VS': 69.7, 'FL': 66.5, 'MG': np.nan, 'JD': 66.5, 'SJ': 53.1}   # 

WRF1 = Dataset(f'/home/woody/gwgk/gwgi019h/output_WRF/nomodlakes_glac2019/output_wrfout/wrfout_d03_{year}-{month:02}-{day-1:02}_06:00:00')
WRF1 = WRF1['RAINNC'][0]
WRF2 = Dataset(f'/home/woody/gwgk/gwgi019h/output_WRF/nomodlakes_glac2019/output_wrfout/wrfout_d03_{year}-{month:02}-{day:02}_06:00:00')
lons = WRF2['XLONG'][0]
lats = WRF2['XLAT'][0]
WRF2 = WRF2['RAINNC'][0]

vmin=0; vmax=100

fig, (ax1) = plt.subplots(figsize=(15,10))
plt.rcParams.update({'font.size': 16}) 

cf = ax1.pcolormesh(lons, lats, WRF2-WRF1, vmin=vmin, vmax=vmax)
ax1.contour(lons, lats, WRF_hgt_exp1, levels=np.arange(0,2101,300), cmap='terrain')

indices = np.argwhere(WRF_lu_exp1.values == 24)
for (i, j) in indices:
    ax1.scatter(lons[i,j], lats[i,j], color='k', s=2, zorder=100000)
    
for key in obs.keys():
    lat = WRF1000_ts.loc[WRF1000_ts['station_id'] == key, 'station_lat']
    lon = WRF1000_ts.loc[WRF1000_ts['station_id'] == key, 'station_lon']
    ax1.scatter(lon, lat, c=obs[key], ec='k', s=100, vmin=vmin, vmax=vmax, zorder=1000)
plt.colorbar(cf)

ax1.set_xlabel('Longitude ($\u00b0$)')
ax1.set_ylabel('Latitude ($\u00b0$)')
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_title(f'{year}.{month}.{day}')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[167]:


condition_hgt = WRF_hgt_exp1.values > 1850
condition_lat = (WRF_lat.values > 61.6) & (WRF_lat.values < 61.8)
condition_lon = (WRF_lon.values > 6.8) & (WRF_lon.values < 7.3)

combined_condition = condition_hgt & condition_lat & condition_lon
#combined_condition.values.sum()

indices = np.argwhere(combined_condition)
indices


# In[156]:


fig, ax = plt.subplots(figsize=(15,10))

masked_hgt = np.where(WRF_hgt_exp1 > 1800 & WRF_lon > 6.8, WRF_hgt_exp1, np.nan)
ax.pcolormesh(WRF_lon, WRF_lat, masked_hgt, vmin=-100, vmax=2000)

#indices = np.argwhere(WRF_lu_exp1.values == 24)
#for (i, j) in indices:
#    ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color='k', s=2)
    
ax.axhline(61.6)
ax.axhline(61.8)
ax.axvline(6.8)
ax.axvline(7.3)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_xlabel('Longitude ($\u00b0$)')
ax.set_ylabel('Latitude ($\u00b0$)')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[86]:


# plotting SON, DJF, MAM together with winter mass balance
# comparing experiments
#
# precipitation during these three seasons is very likely higher than observed values 
# due to time period of 9 months instead of approx. 7 months


#xlat, xlon = np.meshgrid(WRF_rain['XLAT'],WRF_rain['XLONG'])
vmin = -150; vmax = 150

y = 2007
#years = [2007, 2008, 2009, 2010, 2011, 2012]
exp2 = 'noice_BT'#dtm50'#
variable = 'RAINNC'#'SNOWNC' #

exp = 'glac2006'
seasons = ['_SON_', '_DJF_', '_MAM_']
months = [1,2,3,4,5]
months_prevyear = [9,10,11,12]

winterprecip_all = {}
wintersnow_all = {}
winterprecip_exp2_all = {}
obs_precip_all = np.nan*np.ones((7, len(years)))

for y in years:
    
    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams.update({'font.size': 16})

    WRF = xr.open_mfdataset('/home/titan/gwgk/gwgi019h/validation/output/wrfout_d03_static')
    WRF_hgt = np.squeeze(WRF['HGT'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/{variable}_{y-1}{seasons[0]}{exp}')
    WRF_lon = np.squeeze(WRF['XLONG'])
    WRF_lat = np.squeeze(WRF['XLAT'])
    #WRF_hgt = np.squeeze(WRF['HGT_M'])
    WRF_precip_SON = np.squeeze(WRF[variable])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/{variable}_{y}{seasons[1]}{exp}')
    WRF_precip_DJF = np.squeeze(WRF[variable])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/{variable}_{y}{seasons[2]}{exp}')
    WRF_precip_MAM = np.squeeze(WRF[variable])

    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y-1}{seasons[0]}{exp}')
    WRF_snow_SON = np.squeeze(WRF['SNOWNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y}{seasons[1]}{exp}')
    WRF_snow_DJF = np.squeeze(WRF['SNOWNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y}{seasons[2]}{exp}')
    WRF_snow_MAM = np.squeeze(WRF['SNOWNC'])

    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/{variable}_{y-1}{seasons[0]}{exp2}')
    WRF_precip_SON_exp2 = np.squeeze(WRF[variable])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/{variable}_{y}{seasons[1]}{exp2}')
    WRF_precip_DJF_exp2 = np.squeeze(WRF[variable])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/{variable}_{y}{seasons[2]}{exp2}')
    WRF_precip_MAM_exp2 = np.squeeze(WRF[variable])

    winterprecip = (WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)
    wintersnow = (WRF_snow_SON+WRF_snow_DJF+WRF_snow_MAM)
    winterprecip_exp2 = (WRF_precip_SON_exp2+WRF_precip_DJF_exp2+WRF_precip_MAM_exp2)
    
    winterprecip_all[y-years[0]] = winterprecip
    wintersnow_all[y-years[0]] = wintersnow
    winterprecip_exp2_all[y-years[0]] = winterprecip_exp2
    
    cm = ax.pcolormesh(WRF_lon, WRF_lat, winterprecip-winterprecip_exp2,
                  vmin=vmin, vmax=vmax,  
                  cmap='coolwarm_r')
    ax.contour(WRF_lon, WRF_lat, WRF_hgt, cmap='terrain')
    ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000)

    for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ']):
        if st not in [' ']:#'SB','SJ']:#~np.array([st]).isin(['SB','SJ']):
            mask = (MET_monthly['station'] == st) & (((MET_monthly['year'] == y)&(MET_monthly['month'].isin(months))) | ((MET_monthly['year'] == y-1)&(MET_monthly['month'].isin(months_prevyear))))
            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'][0])
            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'][0])
            if MET_monthly[mask]['precip'].sum() > 0:
                ax.scatter(lon, lat, c='k', zorder=1000)#, MET_monthly[mask]['precip'].sum(), ec='k', lw=.6,
                           #vmin=vmin, vmax=vmax,  
                           #cmap='viridis')
                #grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'][0])
                #grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'][0])
                #wrf_snow = ((WRF_snow_SON+WRF_snow_DJF+WRF_snow_MAM)[grid_j, grid_i]).values
                #wrf_precip = ((WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)[grid_j, grid_i]).values
                met_precip = MET_monthly[mask]['precip'].sum()
                obs_precip_all[s, y-years[0]] = met_precip
                ax.text(lon, lat, #WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i], 
                        f'   {met_precip:.0f}', fontsize=14, va='center') #{wrf_precip:.0f}/
    

    # figure specifications
    
    cb = plt.colorbar(cm)
    cb.set_label('Difference in precipitation (mm)', rotation=270, labelpad=30)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')
    ax.set_title(f'Diff. in precip. ({exp} - {exp2}) for ext. {y-1}/{y} winter (Sep-May)')#' (extended winter)')#{seasonstr}')
    plt.show()
    


# In[ ]:





# In[6]:


#xlat, xlon = np.meshgrid(WRF_rain['XLAT'],WRF_rain['XLONG'])
vmin = 0; vmax = 6000

y = 2007
for y in [2011, 2012]:
    exp = 'glac2006'
    season = '_'#'_DJF_'#'_SON_'#'_JJA_'#'_MAM_'#
    months = np.arange(1,13,1)
    months_prevyear = []
    if season == '_DJF_':
        months = [1,2]
        months_prevyear = [12]
        vmax = 2000
        print (f'looking at DJF for {y-1}-{y}')
    elif season == '_MAM_':
        months = [3,4,5]
        vmax = 1500
        print (f'looking at MAM for {y}')
    elif season == '_JJA_':
        months = [6,7,8]
        vmax = 700
        print (f'looking at JJA for {y}')
    elif season == '_SON_':
        months = [9,10,11]
        vmax = 2000
        print (f'looking at SON for {y}')

    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams.update({'font.size': 16})

    WRF_rain = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y}{season}{exp}')#,combine="nested",concat_dim="Time")
    cm = ax.pcolormesh(np.squeeze(WRF_rain['XLONG']), np.squeeze(WRF_rain['XLAT']), np.squeeze(WRF_rain['RAINNC']),
                  vmin=vmin, vmax=vmax,  
                  cmap='viridis')

    for st in MET_monthly['station'].unique():
        if st not in [' ']:#'SB','SJ']:#~np.array([st]).isin(['SB','SJ']):
            mask = (MET_monthly['station'] == st) & (((MET_monthly['year'] == y)&(MET_monthly['month'].isin(months))) | ((MET_monthly['year'] == y-1)&(MET_monthly['month'].isin(months_prevyear))))
            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'][0])
            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'][0])
            if MET_monthly[mask]['precip'].sum() > 0:
                ax.scatter(lon, lat, c=MET_monthly[mask]['precip'].sum(), ec='k', lw=.6,
                           vmin=vmin, vmax=vmax,  
                           cmap='viridis')
                grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'][0])
                grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'][0])
                wrf_precip = (WRF_rain['RAINNC'][0, grid_j, grid_i]).values
                met_precip = MET_monthly[mask]['precip'].sum()
                ax.text(WRF_rain['XLONG'][0, grid_j, grid_i], WRF_rain['XLAT'][0, grid_j, grid_i], 
                        f'  {wrf_precip:.0f}/{met_precip:.0f}', fontsize=12, va='center')

    cb = plt.colorbar(cm)
    cb.set_label('Annual precipitation (mm)', rotation=270, labelpad=30)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')
    seasonstr = season[:-1]
    ax.set_title(f'Precipitation in {y}{seasonstr}')
    plt.show()


# In[7]:


# plotting SON, DJF, MAM together with winter mass balance
#
# precipitation during these three seasons is very likely higher than observed values 
# due to time period of 9 months instead of approx. 7 months


#xlat, xlon = np.meshgrid(WRF_rain['XLAT'],WRF_rain['XLONG'])
vmin = 0; vmax = 5000

y = 2007
for y in [2007, 2008, 2009, 2010, 2011, 2012]:
    exp = 'glac2006'
    seasons = ['_SON_', '_DJF_', '_MAM_']
    months = [1,2,3,4,5]
    months_prevyear = [9,10,11,12]


    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams.update({'font.size': 16})

#    for season in seasons:
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y-1}{seasons[0]}{exp}')
    WRF_lon = np.squeeze(WRF['XLONG'])
    WRF_lat = np.squeeze(WRF['XLAT'])
    WRF_precip_SON = np.squeeze(WRF['RAINNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y}{seasons[1]}{exp}')
    WRF_precip_DJF = np.squeeze(WRF['RAINNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y}{seasons[2]}{exp}')
    WRF_precip_MAM = np.squeeze(WRF['RAINNC'])

    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y-1}{seasons[0]}{exp}')
    WRF_snow_SON = np.squeeze(WRF['SNOWNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y}{seasons[1]}{exp}')
    WRF_snow_DJF = np.squeeze(WRF['SNOWNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y}{seasons[2]}{exp}')
    WRF_snow_MAM = np.squeeze(WRF['SNOWNC'])
    
    cm = ax.pcolormesh(WRF_lon, WRF_lat, WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM,
                  vmin=vmin, vmax=vmax,  
                  cmap='viridis')
    
    for st in MET_monthly['station'].unique():
        if st not in [' ']:#'SB','SJ']:#~np.array([st]).isin(['SB','SJ']):
            mask = (MET_monthly['station'] == st) & (((MET_monthly['year'] == y)&(MET_monthly['month'].isin(months))) | ((MET_monthly['year'] == y-1)&(MET_monthly['month'].isin(months_prevyear))))
            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'][0])
            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'][0])
            if MET_monthly[mask]['precip'].sum() > 0:
                ax.scatter(lon, lat, c=MET_monthly[mask]['precip'].sum(), ec='k', lw=.6,
                           vmin=vmin, vmax=vmax,  
                           cmap='viridis')
                grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'][0])
                grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'][0])
                #wrf_snow = ((WRF_snow_SON+WRF_snow_DJF+WRF_snow_MAM)[grid_j, grid_i]).values
                wrf_precip = ((WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)[grid_j, grid_i]).values
                met_precip = MET_monthly[mask]['precip'].sum()
                ax.text(WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i]-.01, 
                        f'  {wrf_precip:.0f}/{met_precip:.0f}', fontsize=12, va='center')
    
    # adding winter mass balance
    
    # at Austdalsbreen
    #lat = 61.819722; lon = 7.350495
    lon = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'lon']
    lat = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'lat']
    #wmb_ab = 2530 # for 2006/07
    wmb_ab = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'SWE(mm)']
    ax.scatter(lon, lat, c=wmb_ab, ec='k', lw=.6, marker='s',
               vmin=vmin, vmax=vmax,  
               cmap='viridis')   
    grid_i = 93; grid_j = 71
    wrf_snow = ((WRF_snow_SON+WRF_snow_DJF+WRF_snow_MAM)[grid_j, grid_i]).values
    #wrf_precip = ((WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)[grid_j, grid_i]).values
    ax.text(WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i]+.005, 
            f'  {wrf_snow:.0f}/{wmb_ab.values[0]:.0f}', fontsize=12, va='center')

    # at Nigardsbreen
    #lat = 61.713528; lon = 7.084047 # some variation, consider updating each year
    lon = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'lon']
    lat = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'lat']
    #wmb_nb = 3235
    wmb_nb = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'SWE(mm)']
    ax.scatter(lon, lat, c=wmb_nb, ec='k', lw=.6, marker='s',
               vmin=vmin, vmax=vmax,  
               cmap='viridis')        
    grid_i = 79; grid_j = 59
    wrf_snow = ((WRF_snow_SON+WRF_snow_DJF+WRF_snow_MAM)[grid_j, grid_i]).values
    #wrf_precip = ((WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)[grid_j, grid_i]).values
    ax.text(WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i]+.005, 
            f'  {wrf_snow:.0f}/{wmb_nb.values[0]:.0f}', fontsize=12, va='center')
    
    cb = plt.colorbar(cm)
    cb.set_label('Precipitation (mm)', rotation=270, labelpad=30)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')
    #seasonstr = season[:-1]
    ax.set_title(f'Precipitation in extended {y-1}/{y} winter (Sep-May)')#' (extended winter)')#{seasonstr}')
    plt.show()


# In[87]:


# plot all years difference
    
fig, ax = plt.subplots(figsize=(15,10))
plt.rcParams.update({'font.size': 16}) 

winterprecip_all_sum = xr.zeros_like(next(iter(winterprecip_all.values())))
for arr in winterprecip_all.values():
    winterprecip_all_sum += arr
    
wintersnow_all_sum = xr.zeros_like(next(iter(winterprecip_all.values())))
for arr in wintersnow_all.values():
    wintersnow_all_sum += arr

winterprecip_exp2_all_sum = xr.zeros_like(next(iter(winterprecip_exp2_all.values())))
for arr in winterprecip_exp2_all.values():
    winterprecip_exp2_all_sum += arr

if exp2 == 'noice_BT':
    vmin = -2000; vmax = 2000
elif exp2 == 'noice_dtm50':
    vmin = -300; vmax = 300
cm = ax.pcolormesh(WRF_lon, WRF_lat, winterprecip_all_sum-winterprecip_exp2_all_sum,
              vmin=vmin, vmax=vmax,  
              cmap='coolwarm_r')
ax.contour(WRF_lon, WRF_lat, WRF_hgt, cmap='terrain')
ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000)

for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ']):
    if st not in [' ']:
        mask = (MET_monthly['station'] == st) & (((MET_monthly['year'] == y)&(MET_monthly['month'].isin(months))) | ((MET_monthly['year'] == y-1)&(MET_monthly['month'].isin(months_prevyear))))
        lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'][0])
        lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'][0])
        if MET_monthly[mask]['precip'].sum() > 0:
            ax.scatter(lon, lat, c='k', zorder=1000)
            met_precip = MET_monthly[mask]['precip'].sum()
            obs_precip_all[s, y-years[0]] = met_precip
            ax.text(lon, lat, 
                    f'   {obs_precip_all.sum(axis=1)[s]:.0f}', fontsize=14, va='center')

cb = plt.colorbar(cm)
cb.set_label('Difference in precipitation (mm)', rotation=270, labelpad=30)

ax.set_xlabel('Longitude ($\u00b0$)')
ax.set_ylabel('Latitude ($\u00b0$)')
startyear = years[0]
endyear = years[-1]
ax.set_title(f'Difference in precipitation ({exp} - {exp2})\nfor extended winter seasons (Sep-May) from {startyear-1} to {endyear}')#' (extended winter)')#{seasonstr}')
plt.show()

# plot all years absolute values

fig, ax = plt.subplots(figsize=(15,10))
plt.rcParams.update({'font.size': 16}) 


vmin = 0; vmax = 25000
cm = ax.pcolormesh(WRF_lon, WRF_lat, winterprecip_all_sum,
              vmin=vmin, vmax=vmax,  
              cmap='viridis')
ax.contour(WRF_lon, WRF_lat, WRF_hgt, cmap='terrain')
ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000)

for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ']):
    if st not in [' '] and ~np.isnan(obs_precip_all.sum(axis=1)[s]):
        #mask = (MET_monthly['station'] == st) & (((MET_monthly['year'] == y)&(MET_monthly['month'].isin(months))) | ((MET_monthly['year'] == y-1)&(MET_monthly['month'].isin(months_prevyear))))
        lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'][0])
        lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'][0])
        if MET_monthly[mask]['precip'].sum() > 0:
            ax.scatter(lon, lat, c=obs_precip_all.sum(axis=1)[s], ec='k', lw=.6, s=50,
                       vmin=vmin, vmax=vmax,  
                       cmap='viridis', zorder=1000)
            grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'][0])
            grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'][0])
            #wrf_snow = ((wintersnow_all_sum)[grid_j, grid_i]).values
            wrf_precip = ((winterprecip_all_sum)[grid_j, grid_i]).values
            #met_precip = MET_monthly[mask]['precip'].sum()
            ax.text(WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i]-.01, 
                    f'  {wrf_precip:.0f} / {obs_precip_all.sum(axis=1)[s]:.0f}', 
                    fontsize=14, va='center',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))            
            
for s in range(5,7):
    if s == 5: # Austdalsbreen    
        lon = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'lon']
        lat = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'lat']
        grid_i = 93; grid_j = 71 
    
    elif s == 6: # Nigardsbreen
        lon = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'lon']
        lat = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'lat']
        grid_i = 79; grid_j = 59
        
#    ax.scatter(lon, lat, c=obs_precip_all.sum(axis=1)[s], ec='k', lw=.6, marker='s', s=50,
#                       vmin=vmin, vmax=vmax,  
#                       cmap='viridis', zorder=1000)
#
#    wrf_snow = ((wintersnow_all_sum)[grid_j, grid_i]).values
#    ax.text(WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i]+.01, 
#            f'  {wrf_snow:.0f} / {obs_precip_all.sum(axis=1)[s]:.0f}', 
#            fontsize=14, va='center',
#            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 

cb = plt.colorbar(cm)
cb.set_label('Precipitation (mm)', rotation=270, labelpad=30)

ax.set_xlabel('Longitude ($\u00b0$)')
ax.set_ylabel('Latitude ($\u00b0$)')
startyear = years[0]
endyear = years[-1]
ax.set_title(f'Precipitation for extended winter seasons (Sep-May) from {startyear-1} to {endyear}')#' (extended winter)')#{seasonstr}')

plt.show()


# In[76]:


# plot difference between two years

import matplotlib.colors as mcolors
    
fig, ax = plt.subplots(figsize=(15,10))
plt.rcParams.update({'font.size': 16}) 

y1 = 2007; y2 = 2010

vmin = 0; vmax = 2500

original_cmap = plt.get_cmap('coolwarm_r')
colors = original_cmap(np.linspace(0.5, 1, 256)) 
custom_cmap = mcolors.ListedColormap(colors)

cm = ax.pcolormesh(WRF_lon, WRF_lat, winterprecip_all[y1-years[0]]-winterprecip_all[y2-years[0]],
              vmin=vmin, vmax=vmax,  
              cmap='viridis')
ax.contour(WRF_lon, WRF_lat, WRF_hgt, cmap='terrain')
ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000)

for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ']):
    if st not in [' ']:
        mask = (MET_monthly['station'] == st) & (((MET_monthly['year'] == y)&(MET_monthly['month'].isin(months))) | ((MET_monthly['year'] == y-1)&(MET_monthly['month'].isin(months_prevyear))))
        lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'][0])
        lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'][0])
        if MET_monthly[mask]['precip'].sum() > 0:
            #met_precip = MET_monthly[mask]['precip'].sum()
            #obs_precip_all[s, y-years[0]] = met_precip
            obsdiff = obs_precip_all[s][y1-years[0]]-obs_precip_all[s][y2-years[0]]
            ax.scatter(lon, lat, c=obsdiff, ec='k', lw=.6, s=50,
                       vmin=vmin, vmax=vmax,  
                       cmap='viridis', zorder=1000)
            grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'][0])
            grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'][0])
            wrfdiff_precip = ((winterprecip_all[y1-years[0]]-winterprecip_all[y2-years[0]])[grid_j, grid_i]).values
            ax.text(lon, lat-.01, 
                    f'   {wrfdiff_precip:.0f}/{obsdiff:.0f}', fontsize=14, va='center')

for s in range(5,7):
    if s == 5: # Austdalsbreen    
        lon = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'lon']
        lat = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'lat']
        grid_i = 93; grid_j = 71 
    
    elif s == 6: # Nigardsbreen
        lon = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'lon']
        lat = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'lat']
        grid_i = 79; grid_j = 59
    
    obsdiff = obs_precip_all[s][y1-years[0]]-obs_precip_all[s][y2-years[0]]
                
    ax.scatter(lon, lat, c=obsdiff, ec='k', lw=.6, marker='s', s=50,
               vmin=vmin, vmax=vmax,  
               cmap='viridis', zorder=1000)

    wrfdiff_snow = ((wintersnow_all[y1-years[0]]-wintersnow_all[y2-years[0]])[grid_j, grid_i]).values
    ax.text(WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i], 
            f'  {wrfdiff_snow:.0f}/{obsdiff:.0f}', fontsize=14, va='center') 

    
cb = plt.colorbar(cm)
cb.set_label('Difference in precipitation (mm)', rotation=270, labelpad=30)

ax.set_xlabel('Longitude ($\u00b0$)')
ax.set_ylabel('Latitude ($\u00b0$)')
startyear = years[0]
endyear = years[-1]
ax.set_title(f'Difference in precipitation ({y1} - {y2})\nfor extended winter seasons (Sep-May)')#' (extended winter)')#{seasonstr}')
plt.show()


# In[49]:


winterprecip_all[y1-years[0]]


# In[45]:


obs_precip_all[s][y1-years[0]]


# In[26]:


obs_precip_all


# In[90]:


met_precip, met_precip_all.sum(axis=1)[s]


# In[67]:


#winterprecip_all[0]
#total_sum = xr.DataArray(0, dims=winterprecip_all[0].dims)  # Initialize with zero DataArray
for arr in winterprecip_all.values():
    total_sum += arr


# In[33]:


fig, ax = plt.subplots(figsize=(7,7))

#years = [2007, 2008, 2009, 2010, 2011]

import seaborn as sns
import matplotlib as mpl

sns.set_palette("tab10", plt.cm.tab10.N )
c = plt.rcParams['axes.prop_cycle'].by_key()['color']
c = list(map(mpl.colors.rgb2hex, c))

obs_precip_all = np.nan*np.ones((7, len(years)))
wrf_precip_all = np.nan*np.ones((7, len(years)))
wrf_snow_all   = np.nan*np.ones((7, len(years)))

exp = 'glac2006'
seasons = ['_SON_', '_DJF_', '_MAM_']
months = [1,2,3,4,5]
months_prevyear = [9,10,11,12]

for y in years:

    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y-1}{seasons[0]}{exp}')
    WRF_lon = np.squeeze(WRF['XLONG'])
    WRF_lat = np.squeeze(WRF['XLAT'])
    WRF_precip_SON = np.squeeze(WRF['RAINNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y}{seasons[1]}{exp}')
    WRF_precip_DJF = np.squeeze(WRF['RAINNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y}{seasons[2]}{exp}')
    WRF_precip_MAM = np.squeeze(WRF['RAINNC'])

    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y-1}{seasons[0]}{exp}')
    WRF_snow_SON = np.squeeze(WRF['SNOWNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y}{seasons[1]}{exp}')
    WRF_snow_DJF = np.squeeze(WRF['SNOWNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y}{seasons[2]}{exp}')
    WRF_snow_MAM = np.squeeze(WRF['SNOWNC'])
    
    for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ']):#MET_monthly['station'].unique()):
        if st not in [' ']:#'SB','SJ']:#~np.array([st]).isin(['SB','SJ']):
            mask = (MET_monthly['station'] == st) & (((MET_monthly['year'] == y)&(MET_monthly['month'].isin(months))) | ((MET_monthly['year'] == y-1)&(MET_monthly['month'].isin(months_prevyear))))
            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'][0])
            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'][0])
            if MET_monthly[mask]['precip'].sum() > 0:
                grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'][0])
                grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'][0])
                wrf_snow = ((WRF_snow_SON+WRF_snow_DJF+WRF_snow_MAM)[grid_j, grid_i]).values
                wrf_precip = ((WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)[grid_j, grid_i]).values
                met_precip = MET_monthly[mask]['precip'].sum()
                obs_precip_all[s, y-years[0]] = (met_precip)
                wrf_precip_all[s, y-years[0]] = (wrf_precip)
                wrf_snow_all[s, y-years[0]] = (wrf_snow)

    # adding winter mass balance
    obs_precip_all[5, y-years[0]] = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'SWE(mm)']
    obs_precip_all[6, y-years[0]] = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'SWE(mm)']
        
    grid_i = 93; grid_j = 71 ###########################
    wrf_precip_all[5, y-years[0]] = ((WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)[grid_j, grid_i]).values
    wrf_snow_all[5, y-years[0]] = ((WRF_snow_SON+WRF_snow_DJF+WRF_snow_MAM)[grid_j, grid_i]).values

    grid_i = 79; grid_j = 59
    wrf_precip_all[6, y-years[0]] = ((WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)[grid_j, grid_i]).values 
    wrf_snow_all[6, y-years[0]] = ((WRF_snow_SON+WRF_snow_DJF+WRF_snow_MAM)[grid_j, grid_i]).values           
                
for s in range(5):
    ax.plot(years, obs_precip_all[s], c=c[s], marker='o')
    ax.plot(years, wrf_precip_all[s], c=c[s], marker='.', ls='--')
    #ax.plot(years, wrf_snow_all[s],   c=c[s], marker='s', ls='--')
    
for s in range(5,7):
    ax.plot(years, obs_precip_all[s], c=c[s], marker='o')
    #ax.plot(years, wrf_precip_all[s], c=c[s], marker='.', ls='--')
    ax.plot(years, wrf_snow_all[s],   c=c[s], marker='s', ls='--')

ax.scatter((), (), c='grey', marker='o', label='obs precip')
ax.scatter((), (), c='grey', marker='.', label='wrf precip')
ax.scatter((), (), c='grey', marker='s', label='wrf snow')
ax.scatter((), (), c='w', label=' ')
for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ', 'AB*', 'NB*']):
    ax.scatter((), (), c=c[s], marker='*', label=st)
            
ax.set_xlabel('Year')
ax.set_ylabel('Precipitation for ext. winter season (mm)')
plt.legend(bbox_to_anchor=(1.0, 1.0))

plt.show()


fig, ax = plt.subplots(figsize=(7,7))

for s in range(5):
    ax.plot(years, wrf_precip_all[s]-obs_precip_all[s], c=c[s], marker='.')
    
for s in range(5,7):
    ax.plot(years, wrf_snow_all[s]-obs_precip_all[s], c=c[s], marker='s')

#ax.scatter((), (), c='grey', marker='o', label='obs precip')
ax.scatter((), (), c='grey', marker='.', label='wrf precip')
ax.scatter((), (), c='grey', marker='s', label='wrf snow')
ax.scatter((), (), c='w', label=' ')
for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ', 'AB*', 'NB*']):
    ax.scatter((), (), c=c[s], marker='*', label=st)
            
ax.set_xlabel('Year')
ax.set_ylabel('Difference in precip. between WRF and OBS \n for ext. winter season (mm)')
plt.legend(bbox_to_anchor=(1.0, 1.0))

plt.show()


fig, ax = plt.subplots(figsize=(7,7))

for s in range(5):
    ax.plot(years, (wrf_precip_all[s]-obs_precip_all[s])/obs_precip_all[s]*100, c=c[s], marker='.')
    
for s in range(5,7):
    ax.plot(years, (wrf_snow_all[s]-obs_precip_all[s])/obs_precip_all[s]*100, c=c[s], marker='s')

#ax.scatter((), (), c='grey', marker='o', label='obs precip')
ax.scatter((), (), c='grey', marker='.', label='wrf precip')
ax.scatter((), (), c='grey', marker='s', label='wrf snow')
ax.scatter((), (), c='w', label=' ')
for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ', 'AB*', 'NB*']):
    ax.scatter((), (), c=c[s], marker='*', label=st)
            
ax.set_xlabel('Year')
ax.set_ylabel('Rel. diff. in precip. between WRF and OBS \n for ext. winter season (%)')
plt.legend(bbox_to_anchor=(1.0, 1.0))

plt.show()


# In[20]:


fig, ax = plt.subplots(figsize=(7,7))

obs_precip_all_exp2 = np.nan*np.ones((7, len(years)))
wrf_precip_all_exp2 = np.nan*np.ones((7, len(years)))
wrf_snow_all_exp2   = np.nan*np.ones((7, len(years)))
obs_precip_all_exp3 = np.nan*np.ones((7, len(years)))
wrf_precip_all_exp3 = np.nan*np.ones((7, len(years)))
wrf_snow_all_exp3   = np.nan*np.ones((7, len(years)))

exp2 = 'noice_dtm50'
exp3 = 'noice_BT'

for y in years:

    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y-1}{seasons[0]}{exp2}')
    WRF_precip_SON_exp2 = np.squeeze(WRF['RAINNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y}{seasons[1]}{exp2}')
    WRF_precip_DJF_exp2 = np.squeeze(WRF['RAINNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y}{seasons[2]}{exp2}')
    WRF_precip_MAM_exp2 = np.squeeze(WRF['RAINNC'])

    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y-1}{seasons[0]}{exp2}')
    WRF_snow_SON_exp2 = np.squeeze(WRF['SNOWNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y}{seasons[1]}{exp2}')
    WRF_snow_DJF_exp2 = np.squeeze(WRF['SNOWNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y}{seasons[2]}{exp2}')
    WRF_snow_MAM_exp2 = np.squeeze(WRF['SNOWNC'])
    
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y-1}{seasons[0]}{exp3}')
    WRF_precip_SON_exp3 = np.squeeze(WRF['RAINNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y}{seasons[1]}{exp3}')
    WRF_precip_DJF_exp3 = np.squeeze(WRF['RAINNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/RAINNC_{y}{seasons[2]}{exp3}')
    WRF_precip_MAM_exp3 = np.squeeze(WRF['RAINNC'])

    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y-1}{seasons[0]}{exp3}')
    WRF_snow_SON_exp3 = np.squeeze(WRF['SNOWNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y}{seasons[1]}{exp3}')
    WRF_snow_DJF_exp3 = np.squeeze(WRF['SNOWNC'])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/SNOWNC_{y}{seasons[2]}{exp3}')
    WRF_snow_MAM_exp3 = np.squeeze(WRF['SNOWNC'])
    
    for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ']):#MET_monthly['station'].unique()):
        if st not in [' ']:#'SB','SJ']:#~np.array([st]).isin(['SB','SJ']):
            mask = (MET_monthly['station'] == st) & (((MET_monthly['year'] == y)&(MET_monthly['month'].isin(months))) | ((MET_monthly['year'] == y-1)&(MET_monthly['month'].isin(months_prevyear))))
            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'][0])
            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'][0])
            if MET_monthly[mask]['precip'].sum() > 0:
                grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'][0])
                grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'][0])
                wrf_snow_exp2 = ((WRF_snow_SON_exp2    +WRF_snow_DJF_exp2  +WRF_snow_MAM_exp2)  [grid_j, grid_i]).values
                wrf_precip_exp2 = ((WRF_precip_SON_exp2+WRF_precip_DJF_exp2+WRF_precip_MAM_exp2)[grid_j, grid_i]).values
                wrf_snow_exp3 = ((WRF_snow_SON_exp3    +WRF_snow_DJF_exp3  +WRF_snow_MAM_exp3)  [grid_j, grid_i]).values
                wrf_precip_exp3 = ((WRF_precip_SON_exp3+WRF_precip_DJF_exp3+WRF_precip_MAM_exp3)[grid_j, grid_i]).values
                #met_precip = MET_monthly[mask]['precip'].sum()
                #obs_precip_all[s, y-years[0]] = (met_precip)
                wrf_precip_all_exp2[s, y-years[0]] = (wrf_precip_exp2)
                wrf_snow_all_exp2  [s, y-years[0]] = (wrf_snow_exp2)
                wrf_precip_all_exp3[s, y-years[0]] = (wrf_precip_exp3)
                wrf_snow_all_exp3  [s, y-years[0]] = (wrf_snow_exp3)

    # adding winter mass balance
    obs_precip_all[5, y-years[0]] = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'SWE(mm)']
    obs_precip_all[6, y-years[0]] = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'SWE(mm)']
    
    grid_i = 93; grid_j = 71 ###########################
    wrf_precip_all_exp2[5, y-years[0]] = ((WRF_precip_SON_exp2+WRF_precip_DJF_exp2+WRF_precip_MAM_exp2)[grid_j, grid_i]).values
    wrf_snow_all_exp2  [5, y-years[0]] = ((WRF_snow_SON_exp2  +WRF_snow_DJF_exp2  +WRF_snow_MAM_exp2)  [grid_j, grid_i]).values
    wrf_precip_all_exp3[5, y-years[0]] = ((WRF_precip_SON_exp3+WRF_precip_DJF_exp3+WRF_precip_MAM_exp3)[grid_j, grid_i]).values
    wrf_snow_all_exp3  [5, y-years[0]] = ((WRF_snow_SON_exp3  +WRF_snow_DJF_exp3  +WRF_snow_MAM_exp3)  [grid_j, grid_i]).values

    grid_i = 79; grid_j = 59
    wrf_precip_all_exp2[6, y-years[0]] = ((WRF_precip_SON_exp2+WRF_precip_DJF_exp2+WRF_precip_MAM_exp2)[grid_j, grid_i]).values 
    wrf_snow_all_exp2  [6, y-years[0]] = ((WRF_snow_SON_exp2  +WRF_snow_DJF_exp2  +WRF_snow_MAM_exp2)  [grid_j, grid_i]).values           
    wrf_precip_all_exp3[6, y-years[0]] = ((WRF_precip_SON_exp3+WRF_precip_DJF_exp3+WRF_precip_MAM_exp3)[grid_j, grid_i]).values 
    wrf_snow_all_exp3  [6, y-years[0]] = ((WRF_snow_SON_exp3  +WRF_snow_DJF_exp3  +WRF_snow_MAM_exp3)  [grid_j, grid_i]).values           
                
for s in range(5):
    ax.plot(years, obs_precip_all[s], c=c[s], marker='o')
    ax.plot(years, wrf_precip_all[s], c=c[s], marker='.', ls='--')
    ax.plot(years, wrf_precip_all_exp2[s], c=c[s], marker='.', ls=':')
    ax.plot(years, wrf_precip_all_exp3[s], c=c[s], marker='.', ls='-.')
    #ax.plot(years, wrf_snow_all[s], c=c[s], marker='s', ls='--')
    
for s in range(5,7):
    ax.plot(years, obs_precip_all[s], c=c[s], marker='o')
    #ax.plot(years, wrf_precip_all[s], c=c[s], marker='.', ls='--')
    ax.plot(years, wrf_snow_all[s], c=c[s], marker='s', ls='--')
    ax.plot(years, wrf_snow_all_exp2[s], c=c[s], marker='s', ls=':')
    ax.plot(years, wrf_snow_all_exp3[s], c=c[s], marker='s', ls='-.')

ax.scatter((), (), c='grey', marker='o', label='obs precip')
ax.scatter((), (), c='grey', marker='.', label='wrf precip')
ax.scatter((), (), c='grey', marker='s', label='wrf snow')
ax.scatter((), (), c='w', label=' ')

ax.plot((), (), c='grey', ls='--', label='wrf glac2006')
ax.plot((), (), c='grey', ls=':', label='wrf noice')
ax.plot((), (), c='grey', ls='-.', label='wrf noice_lowalt')
ax.plot((), (), c='w', label=' ')

for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ', 'AB*', 'NB*']):
    ax.scatter((), (), c=c[s], marker='*', label=st)
            
ax.set_xlabel('Year')
ax.set_ylabel('Precipitation for ext. winter season (mm)')
plt.legend(bbox_to_anchor=(1.0, 1.0))

plt.show()


# In[ ]:





# In[49]:


# plotting SON, DJF, MAM together with winter mass balance
#
# precipitation during these three seasons is very likely higher than observed values 
# due to time period of 9 months instead of approx. 7 months

variable = 'RAINNC' # RAINNC, SNOWNC, PREC_ACC_NC, PRECNC

#xlat, xlon = np.meshgrid(WRF_rain['XLAT'],WRF_rain['XLONG'])
vmin = 0; vmax = 6000

y = 2007
for y in [2007]:
    exp = 'glac2006'
    seasons = ['_SON_', '_DJF_', '_MAM_']
    months = [1,2,3,4,5]
    months_prevyear = [9,10,11,12]


    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams.update({'font.size': 16})

#    for season in seasons:
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/{variable}_{y-1}{seasons[0]}{exp}')
    WRF_lon = np.squeeze(WRF['XLONG'])
    WRF_lat = np.squeeze(WRF['XLAT'])
    WRF_precip_SON = np.squeeze(WRF[variable])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/{variable}_{y}{seasons[1]}{exp}')
    WRF_precip_DJF = np.squeeze(WRF[variable])
    WRF = xr.open_mfdataset(f'/home/vault/gwgi/gwgi019h/output_WRF/output_subsets/maps/{variable}_{y}{seasons[2]}{exp}')
    WRF_precip_MAM = np.squeeze(WRF[variable])
    
    cm = ax.pcolormesh(WRF_lon, WRF_lat, WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM,
                  vmin=vmin, vmax=vmax,  
                  cmap='viridis')
    
    for st in MET_monthly['station'].unique():
        if st not in [' ']:#'SB','SJ']:#~np.array([st]).isin(['SB','SJ']):
            mask = (MET_monthly['station'] == st) & (((MET_monthly['year'] == y)&(MET_monthly['month'].isin(months))) | ((MET_monthly['year'] == y-1)&(MET_monthly['month'].isin(months_prevyear))))
            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'][0])
            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'][0])
            if MET_monthly[mask]['precip'].sum() > 0:
                ax.scatter(lon, lat, c=MET_monthly[mask]['precip'].sum(), ec='k', lw=.6,
                           vmin=vmin, vmax=vmax,  
                           cmap='viridis')
                grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'][0])
                grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'][0])
                wrf_precip = ((WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)[grid_j, grid_i]).values
                met_precip = MET_monthly[mask]['precip'].sum()
                ax.text(WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i], 
                        f'  {wrf_precip:.0f}/{met_precip:.0f}', fontsize=12, va='center')
    
    # adding winter mass balance
    
    # at Austdalsbreen
    #lat = 61.819722; lon = 7.350495
    lon = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'lon']
    lat = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'lat']
    #wmb_ab = 2530 # for 2006/07
    wmb_ab = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'SWE(mm)']
    ax.scatter(lon, lat, c=wmb_ab, ec='k', lw=.6,
               vmin=vmin, vmax=vmax,  
               cmap='viridis')   
    grid_i = 93; grid_j = 71
    wrf_precip = ((WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)[grid_j, grid_i]).values
    ax.text(WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i], 
            f'  {wrf_precip:.0f}/{wmb_ab.values[0]:.0f}', fontsize=12, va='center')

    # at Nigardsbreen
    #lat = 61.713528; lon = 7.084047 # some variation, consider updating each year
    lon = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'lon']
    lat = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'lat']
    #wmb_nb = 3235
    wmb_nb = massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'SWE(mm)']
    ax.scatter(lon, lat, c=wmb_nb, ec='k', lw=.6,
               vmin=vmin, vmax=vmax,  
               cmap='viridis')        
    grid_i = 79; grid_j = 59
    wrf_precip = ((WRF_precip_SON+WRF_precip_DJF+WRF_precip_MAM)[grid_j, grid_i]).values
    ax.text(WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i], 
            f'  {wrf_precip:.0f}/{wmb_nb.values[0]:.0f}', fontsize=12, va='center')
    
    cb = plt.colorbar(cm)
    cb.set_label('Winter precipitation (mm)', rotation=270, labelpad=30)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')
    #seasonstr = season[:-1]
    ax.set_title(f'Snow in {y} (extended winter)')#{seasonstr}')
    plt.show()


# In[50]:


lat = 61.819722; lon = 7.350495
lat = 61.713528; lon = 7.084047

grid_i = 79; grid_j = 59

WRF_lat[grid_j, grid_i].values, WRF_lon[grid_j, grid_i].values


# In[29]:


lon_idx = (np.abs(WRF_lon.compute() - lon)).argmin()#.item()
lat_idx = (np.abs(WRF_lat.compute() - lat)).argmin()#.item()
lon_idx, lat_idx


# In[ ]:




#%%

# model performance for 12/all wettest days in Oldedalen
import os

dates = ['2020-11-19', '2015-03-08', '2014-10-28', '2013-11-16', '2008-01-29', '2019-01-01', '2022-01-13', '2014-10-29', '2010-07-22', '2019-09-15', '2008-11-27', '2019-12-30']
obs_precip = [96, 83.5, 74.8, 73.8, 72.3, 71.3, 71.3, 67, 66.2, 65.8, 64.5, 64.2]
wrf_precip = []

dates = pd.date_range(start='2007-01-02', end='2023-01-01', freq='D')
dates = dates.strftime('%Y-%m-%d').tolist()

for date in dates:
    if os.path.exists(f'/home/woody/gwgk/gwgi019h/output_WRF/nomodlakes_glac2019/surface_levels/wrfout_d03_SL_{date}'):
        WRF = Dataset(f'/home/woody/gwgk/gwgi019h/output_WRF/nomodlakes_glac2019/surface_levels/wrfout_d03_SL_{date}')
    else:
        now = np.nan
    try:
        now = WRF['RAINNC'][6,57,65]
    except IndexError:
        now = np.nan
        
    date = str(pd.to_datetime(date)-pd.Timedelta(hours=24))[:10]
    if os.path.exists(f'/home/woody/gwgk/gwgi019h/output_WRF/nomodlakes_glac2019/surface_levels/wrfout_d03_SL_{date}'):
        WRF = Dataset(f'/home/woody/gwgk/gwgi019h/output_WRF/nomodlakes_glac2019/surface_levels/wrfout_d03_SL_{date}')
    else:
        yesterday = np.nan
    try:
        yesterday = WRF['RAINNC'][6,57,65]
    except IndexError:
        yesterday = np.nan
    #yesterday = WRF['RAINNC'][6,57,65]
    wrf_precip.append(now-yesterday)
    
#%%
    
MET_daily = pd.read_csv(f"MET_data/obs_daily.csv", sep='\t', decimal=',', na_values='-', skipfooter=1, engine='python')
MET_daily = MET_daily.rename({'Navn': 'station_name', 'Stasjon': 'station_id', 'Tid(norsk normaltid)': 'date', 'Nedbør (døgn)': 'obs'}, axis=1)
MET_daily['date'] = pd.to_datetime(MET_daily['date'], format="%d.%m.%Y")

precip_obs = MET_daily.loc[(MET_daily['station_name'] == 'Oldedalen')&(MET_daily['date'] > datetime.datetime(2007,1,1))&(MET_daily['date'] <= datetime.datetime(2023,1,1)), ['date', 'obs']]

precip_obs = pd.merge(pd.DataFrame({'date': pd.date_range(start='2007-01-02', end='2023-01-01', freq='D')}), 
                      precip_obs, on='date', how='left')

precip_obs['wrf'] = np.array(wrf_precip)
precip_obs['diff'] = precip_obs['wrf']-precip_obs['obs']

#%%
precip_obs = pd.read_csv('../../WRF_general_output/precip_OD.csv')

#%%

np.sum(precip_obs[precip_obs['diff'].notna()].sort_values(by='diff').iloc[:100]), np.sum(precip_obs[precip_obs['diff'].notna()].sort_values(by='diff').iloc[-100:])
precip_obs[precip_obs['diff'].notna()].sort_values(by='diff').iloc[-60:]
