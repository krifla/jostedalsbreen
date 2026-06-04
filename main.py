#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import pandas as pd
import datetime
from netCDF4 import Dataset

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import matplotlib as mpl

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import geopy.distance
from mpl_toolkits.mplot3d import Axes3D
import dask

from scipy import stats
from scipy.stats import skew
from scipy.stats import shapiro

import seaborn as sns
import cmocean

def run_script(script_path):
    with open(script_path) as script_file:
        exec(script_file.read(), globals())


#%%


# setup some starting data
WRF_3D = Dataset('data/wrfout_d03_2014-01-01')
WRF = xr.open_mfdataset(f'data/static/wrfout_d03_static')


if __name__ == "__main__":
    run_script('loadData.py') # load observational data and define functions for downloading model data
    run_script('weighted4pts.py') # define function for finding inverse distance weighted data


# In[11]:

# define experiments to compare
exp1='glac2019'; exp2='noice_dtm50'#noice_BT'#glac2100_dem2100'#
#exp1='noice_BT'; exp2='modlakes_noice_BT' # years must stop at 2012!

# define time period
years = np.arange(2007,2023) #2023 or 2012 (for modlakes)
season = 'all' # SONDJFMAM, DJF, JJA, all, MAMJJA

# load model data
WRF_hgt_exp1, WRF_hgt_exp2, WRF_lu_exp1, WRF_lu_exp2, WRF_lon, WRF_lat = defineStaticData(exp1=exp1, exp2=exp2)
WRF_precip_exp1, WRF_precip_exp2, WRF_precip_var2_exp1, WRF_precip_var2_exp2, obs_precip = definePrecipData(years=years, season=season, exp1=exp1, exp2=exp2)
WRF_temp_exp1, WRF_temp_exp2, obs_temp = defineTempData(years=years, exp1=exp1, exp2=exp2, season=season)

# define labels for experiments
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
   
    
#%%

# plot temperature, precipitation, snow and static maps for experiments

plotPrecipAbs()
plotSnowAbs()
plotPrecipDiff(vmin=-10, vmax=10)
# plotSnowDiff(vmin=-50, vmax=50)
plotRainDiff(vmin=-50, vmax=50)

# plotTempAbs(temp_model = WRF_temp_exp1, vmin=0,vmax=14)
plotTempDiff(vmin=-1, vmax=1)

# plotLUDiff()
# plotLUDiff3D()
# plotHGTDiff()
# plotHGTDiff3D()



#%%

#########################

# if __name__ == "__main__":
#     run_script('plotTimeseries.py')

#########################


# defineTimeseries()
# plotTimeseries(precip_ice, temp_ice)
# loadTempseries()


#%%


# loading time series
inpath = 'data/output_subsets/timeseries'
experiments = ['glac2019', 'noice_BT']#noice_dtm50']#glac2100_dem2100']#
experiments = [exp1, exp2]

#exp2 = 'noice_BT'#noice_dtm50'

#%%

# load temperature time series 
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

    t2m_1pt[exp]['date'] = pd.to_datetime(t2m_1pt[exp]['date'])

#%%

# create monthly mean
t2m_1pt_monthly = {}
t2m_4pts_monthly = {}

for exp in experiments:
    t2m_1pt[exp]['date'] = pd.to_datetime(t2m_1pt[exp]['date'])
    t2m_1pt_monthly[exp] = t2m_1pt[exp].groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()#drop=True)
    t2m_1pt_monthly[exp]['date'] -= pd.offsets.MonthBegin() # let date for each month reflect first date of given month

    # t2m_4pts[exp]['date'] = pd.to_datetime(t2m_4pts[exp]['date'])
    # t2m_4pts_monthly[exp] = t2m_4pts[exp].groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()#drop=True)
    # t2m_4pts_monthly[exp]['date'] -= pd.offsets.MonthBegin() # let date for each month reflect first date of given month


#%%

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

#%%

plt.rcParams.update({'font.size': 18})

# 10 days of missing data at AS in june 2012
corr_1pt = np.nan*np.ones((len(t2m_monthly_obs.columns[1:])))
corr_4pts = np.nan*np.ones((len(t2m_monthly_obs.columns[1:])))
error_1pt = np.nan*np.ones((len(t2m_monthly_obs.columns[1:]), len(t2m_monthly_obs['date'])))
error_4pts = np.nan*np.ones((len(t2m_monthly_obs.columns[1:]), len(t2m_monthly_obs['date'])))

#c = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf', 'k', 'navy', 'aquamarine']
c = [u'#1f77b4', u'#2ca02c', u'#ff7f0e', u'#7f7f7f', u'#bcbd22', 'navy', u'#17becf']

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
        biaserror = (t2m_1pt_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15).mean()
        line, = ax1.plot(t2m_monthly_obs['date'], t2m_1pt_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15, c=c[i], alpha=alp, ls='-', label=f'{loc}: {abserror:.1f} / {biaserror:.1f}')#_{pts}: {abserror:.1f}')
        handles.append(line)  # Store the handle
        names.append(f'{loc}: {abserror:.1f} / {biaserror:.1f}')
    
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


#%%

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
ax.set_ylabel('Mean bias error\nof mean monthly temperature (K)')
plt.grid()
plt.savefig(f'figures/temp_monthly_error.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()








#%%

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




#%%

# load wind and precip timeseries

inpath = 'data/output_subsets/timeseries'
experiments = ['glac2019','noice_BT']#noice_dtm50']#glac2100_dem2100']#
experiments = [exp1, exp2]

precip_1pt = {}
snow_1pt = {}
ws_1pt = {}
wd_1pt = {}

for exp in experiments:
    for var in ['WS', 'WD']:
        dataframes_1pt = []
        #dataframes_4pts = []
        for year in range(2007, 2023):
            df = pd.read_csv(f'{inpath}/wind/{var}_1pt_{year}_{exp}.csv')
            if exp == exp1 and var == 'WS' and year == 2007:
                print (df.columns)
            df['date'] = pd.to_datetime(df['date'])
            dataframes_1pt.append(df)
            #df = pd.read_csv(f'{inpath}/wind/{var}_4pts_2006_{exp}.csv')
            #df['date'] = pd.to_datetime(df['date'])
            #dataframes_4pts.append(df)
        if var == 'WS':
            ws_1pt[exp] = pd.concat(dataframes_1pt, ignore_index=True)
            #ws_4pts = pd.concat(dataframes_4pts, ignore_index=True)
        elif var == 'WD':
            wd_1pt[exp] = pd.concat(dataframes_1pt, ignore_index=True)
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
      
    ws_1pt[exp]['year'] = ws_1pt[exp]['date'].dt.year
    ws_1pt[exp]['month'] = ws_1pt[exp]['date'].dt.month
    ws_1pt[exp]['day'] = ws_1pt[exp]['date'].dt.day


#%%

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
    

wind = wd_1pt[exp1][['date', 'PEAK']]
wind.rename(columns={'PEAK': 'wd'}, inplace=True)
wind = pd.merge(wind, ws_1pt[exp1][['date', 'PEAK']], on='date', how='inner')
wind.rename(columns={'PEAK': 'ws'}, inplace=True)
wind = pd.merge(wind, precip_1pt[exp1], on='date', how='inner')

# apply wd classes
wind['class'] = wind['wd'].apply(classify_wd)

for loc in precip_1pt[exp].columns[1:]:
    wind[loc][np.where(wind[loc]<wind[loc][0])[0][0]] = np.nan
    wind[loc] = wind[loc].diff()
    wind.loc[wind[loc]<0, loc] = np.nan # remove negative values (that arise due to restart?)
    wind.loc[wind[loc]>100, loc] = np.nan # remove unrealistically high values (that arise due to restart?)


#%%

#run_script('loadData.py')

summer = False
winter = False

if summer == True:
    winter = False
    ws_1pt[exp1] = ws_1pt[exp1][ws_1pt['date'].dt.month.isin([6,7,8])]
    wd_1pt[exp1] = wd_1pt[exp1][wd_1pt['date'].dt.month.isin([6,7,8])]
    nve_hourly = nve_hourly[(nve_hourly['date'].dt.month.isin([6,7,8]))]
    MET_hourly = MET_hourly[(MET_hourly['date'].dt.month.isin([6,7,8]))]
    NB_hourly = NB_hourly[(NB_hourly['date'].dt.month.isin([6,7,8]))]
    SM = SM[(SM['date'].dt.month.isin([6,7,8]))]
elif winter == True:
    ws_1pt[exp1] = ws_1pt[exp1][ws_1pt['date'].dt.month.isin([12,1,2])]
    wd_1pt[exp1] = wd_1pt[exp1][wd_1pt['date'].dt.month.isin([12,1,2])]
    nve_hourly = nve_hourly[(nve_hourly['date'].dt.month.isin([12,1,2]))]
    MET_hourly = MET_hourly[(MET_hourly['date'].dt.month.isin([12,1,2]))]
    NB_hourly = NB_hourly[(NB_hourly['date'].dt.month.isin([12,1,2]))]
    SM = SM[(SM['date'].dt.month.isin([12,1,2]))]



#%%

#from matplotlib.patches import Circle

def wind_rose(ws, wd, ax, speed_bins=[0,5,10,20,60], num_dirs=36, cmap='summer', ec='k'):

    direction_bins = (np.linspace(0, 360, num_dirs + 1)) # + 22.5) % 360  # offset for bin edges

    hist, _, _ = np.histogram2d(wd, ws, bins=[direction_bins, speed_bins])#, density=False)
    if hist.sum() != 0:
        hist /= hist.sum()            # Normalize to a fraction of 1

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
    
    
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)

    # Hide labels
    ax.set_xticklabels([])  # Remove angular ticks
    ax.set_yticklabels([])  # Remove radial ticks
    ax.set_yticks([])  # Remove radial ticks
    
    # Set the color of the outer edge of the polar plot
    for spine in ax.spines.values():
        spine.set_visible(True)  # Make the spine visible
        spine.set_edgecolor(ec)  # Set the edge color
        spine.set_linewidth(2)  # Set line width for the edge

    return ax


#%%

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


#%%


cmap=cmocean.cm.phase

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


#%%

# months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
#                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# months = np.arange(1,13,1)
# num_months = 12
# if season == 'DJF':
#     months_names = ['Dec', 'Jan', 'Feb']
#     months = [12,1,2]
#     num_months = 3
# elif season == 'JJA':
#     months_names = ['Jun', 'Jul', 'Aug']
#     months = [6,7,8]
#     num_months = 3

# regimes = {}

# plt.rcParams.update({'font.size': 22})

# fig, ax = plt.subplots(16, num_months, subplot_kw=dict(projection='polar'), figsize=(16, 20))
# ax = ax.flatten()

# speed_bins = [0, 2.5, 5, 10, 15, 20]

# for y in years[:]:
#     for m, month in enumerate(months):
#         mask = (ws_1pt['year'] == y) & (ws_1pt['month'] == month)
        
#         dominant_regime = estimate_wind_regime(wd_1pt[mask]['PEAK'])
#         regimes[f'{y}_{month:02}'] = dominant_regime

#         sm = wind_rose(ws_1pt[mask]['PEAK'], wd_1pt[mask]['PEAK'], ax[(y-years[0])*num_months+m], speed_bins, ec=wind_color(dominant_regime))

# for y, year in enumerate(years):
#     ax[y*num_months].set_ylabel(f'{year}', fontsize=12, labelpad=20)

# for m, month_name in enumerate(months_names):
#     ax[(len(years)-1)*num_months+m].set_xlabel(month_name, fontsize=12)
    
# plt.show()

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
                mask = (ws_1pt[exp1]['year'] == y) & (ws_1pt[exp1]['month'] == month) & (ws_1pt[exp1]['day'] == day)
            except ValueError:
                continue
        
            dominant_regime = estimate_wind_regime(wd_1pt[exp1][mask]['PEAK'])
            regimes_daily[f'{y}_{month:02}_{day:02}'] = dominant_regime

#%%

if exp1 == 'glac2019':
    ice_mask = ((WRF_lu_exp1.values == 24))
    
    thr_ice = np.median(WRF_hgt_exp1.values[ice_mask])
    ice_low_mask = (WRF_hgt_exp1.values < thr_ice)
    ice_high_mask = (WRF_hgt_exp1.values >= thr_ice)
    
    thr_noice = np.median(WRF_hgt_exp1.values[~ice_mask])
    noice_low_mask = (WRF_hgt_exp1.values < thr_noice)
    noice_high_mask = (WRF_hgt_exp1.values >= thr_noice)


#%%

exp = exp1 #'glac2019'

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
    t2m_daily.loc[t2m_daily['date'] == date, 'class'] = regimes_daily[key]
    precip_daily.loc[precip_daily['date']-pd.Timedelta(hours=6) == date, 'class'] = regimes_daily[key]

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
    
# temperature for different wind regimes

plt.rcParams.update({'font.size': 22})

loc = 'PEAK'#'OD'#'FL'#
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
        
ax.axhline(y=np.mean(y_values_N), c=wind_color('N'), ls='--', xmin=0, xmax=.25)
ax.axhspan(xmin=0., xmax=.25, ymin=np.percentile(y_values_N, 10), ymax=np.percentile(y_values_N, 90), fc=wind_color('N'), alpha=.4)
ax.axhline(y=np.mean(y_values_E), c=wind_color('E'), ls='--', xmin=.25, xmax=.5)
ax.axhspan(xmin=.25, xmax=.5, ymin=np.percentile(y_values_E, 10), ymax=np.percentile(y_values_E, 90), fc=wind_color('E'), alpha=.4)
ax.axhline(y=np.mean(y_values_S), c=wind_color('S'), ls='--', xmin=.5, xmax=.75)
ax.axhspan(xmin=.5, xmax=.75, ymin=np.percentile(y_values_S, 10), ymax=np.percentile(y_values_S, 90), fc=wind_color('S'), alpha=.4)
ax.axhline(y=np.mean(y_values_W), c=wind_color('W'), ls='--', xmin=.75, xmax=1)
ax.axhspan(xmin=.75, xmax=1., ymin=np.percentile(y_values_W, 10), ymax=np.percentile(y_values_W, 90), fc=wind_color('W'), alpha=.4)
print (np.mean(y_values_N), np.mean(y_values_E), np.mean(y_values_S), np.mean(y_values_W))

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
    elif exp2 == 'glac2100_dem2100':
        ax.set_title(f'Temperature at {loclabel} (2100-volume vs. control)')
#else:
#    ax.set_title(f'Difference in {season} temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')

right_edge = ax.get_xlim()[1]

ax.axhline(0, c='grey', alpha=0.5, zorder=-100)    
#ax.axvspan(0, right_edge, color='grey', alpha=.2, zorder=-1000)
ax.set_xlim(xmax=right_edge)

ax.set_xlim(xverymin, 4*xrange+xverymin)
ax.set_ylim(-12, 17)

for tick in np.arange(xverymin, 4*xrange, xrange):
    plt.axvline(x=tick, color='gray', linestyle='-', alpha=1, zorder=-100)#0.5)

plt.show()


#%%

# precipitation for different wind regimes

loc = 'PEAK'#'OD'#'FL'#'FL'#

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
    
ax.axhline(y=np.nanmean(y_values_N), c=wind_color('N'), ls='--', xmin=0, xmax=.25)
ax.axhspan(xmin=0., xmax=.25, ymin=np.nanpercentile(y_values_N, 10), ymax=np.nanpercentile(y_values_N, 90), fc=wind_color('N'), alpha=.4)
ax.axhline(y=np.nanmean(y_values_E), c=wind_color('E'), ls='--', xmin=.25, xmax=.5)
ax.axhspan(xmin=.25, xmax=.5, ymin=np.nanpercentile(y_values_E, 10), ymax=np.nanpercentile(y_values_E, 90), fc=wind_color('E'), alpha=.4)
ax.axhline(y=np.nanmean(y_values_S), c=wind_color('S'), ls='--', xmin=.5, xmax=.75)
ax.axhspan(xmin=.5, xmax=.75, ymin=np.nanpercentile(y_values_S, 10), ymax=np.nanpercentile(y_values_S, 90), fc=wind_color('S'), alpha=.4)
ax.axhline(y=np.nanmean(y_values_W), c=wind_color('W'), ls='--', xmin=.75, xmax=1)
ax.axhspan(xmin=.75, xmax=1., ymin=np.nanpercentile(y_values_W, 10), ymax=np.nanpercentile(y_values_W, 90), fc=wind_color('W'), alpha=.4)
print (np.nanmean(y_values_N), np.nanmean(y_values_E), np.nanmean(y_values_S), np.nanmean(y_values_W))

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
    if exp2 == 'noice_BT':
        ax.set_title(f'Precipitation at {loclabel} (no-ice-volume vs. control)')
    elif exp2 == 'noice_dtm50':
        ax.set_title(f'Precipitation at {loclabel} (no-ice-surface vs. control)')
    elif exp2 == 'glac2100_dem2100':
        ax.set_title(f'Precipitation at {loclabel} (2100-volume vs. control)')
#else:
#    ax.set_title(f'Difference in {season} temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')

right_edge = ax.get_xlim()[1]

ax.axhline(0, c='grey', zorder=-100)
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
                 

plt.show()


#%%

# testing significance

for x in [y_values_N, y_values_E, y_values_S, y_values_W]:

    #np.nanmean(y_values_N)
    x = np.array(x)
    x = x[~np.isnan(x)]
    
    n = len(x)
    xbar = x.mean()
    s = x.std(ddof=1)
    
    tstat, pval = stats.ttest_1samp(x, 0.0)
    
    print (n, xbar, s, tstat, pval)


#%%

# checking slope

for x,y in zip([x_values_N, x_values_E, x_values_S, x_values_W], [y_values_N, y_values_E, y_values_S, y_values_W]):

    res = stats.linregress(x, y)
    slope = res.slope
    
    print (slope)
    
#%%

if season == 'all':
    data = pd.merge(MET_monthly.loc[MET_monthly['station'] == 'OD'], nve_monthly.loc[nve_monthly['station_id'] == 'OV', ['date', 't']], on='date', how='inner')
elif season == 'DJF':
    data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'OD')&(MET_monthly['month'].isin([12,1,2]))], nve_monthly.loc[(nve_monthly['station_id'] == 'OV') & (nve_monthly['month'].isin([12,1,2])), ['date', 't']], on='date', how='inner')
elif season == 'JJA':
    data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'OD')&(MET_monthly['month'].isin([6,7,8]))], nve_monthly.loc[(nve_monthly['station_id'] == 'OV') & (nve_monthly['month'].isin([6,7,8])), ['date', 't']], on='date', how='inner')
data = pd.merge(data, wind[['date', 'class']], on='date', how='inner')


#%%


for s, st in enumerate(['FB', 'FL', 'AS', 'SB', 'SM', 'NB']):
#    ax = axs[s]
    
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



#%%


fig, axs = plt.subplots(2, 7, subplot_kw={'projection': 'polar'}, figsize=(11, 3.5), gridspec_kw={'hspace': 0, 'wspace': 0.26})

speed_bins = [0, 2.5, 5, 10, 100]
c = [u'#1f77b4', 'navy', u'#d62728', u'#8c564b', 'k', u'#17becf', 'grey']


for s, st in enumerate(['FB', 'FL', 'AS', 'SB', 'SM', 'NB', 'PEAK']):
    for row in range(2):
        
        ax = axs[row, s]
        
        if row == 0 and st != 'PEAK':
            
            lon, lat = WRF1000_ts.loc[WRF1000_ts['station_id']==st, 'station_lon'], WRF1000_ts.loc[WRF1000_ts['station_id']==st, 'station_lat']
            if st == 'FL':
                data = MET_hourly.loc[(MET_hourly['station_id'] == 'SN55820') & (MET_hourly['date'] >= datetime.datetime(years[0],1,1)) & (MET_hourly['date'] < datetime.datetime(years[-1]+1,1,1)), ['date','ws','wd']]
            elif st == 'SB':
                data = MET_hourly.loc[(MET_hourly['station_id'] == 'SN55425') & (MET_hourly['date'] >= datetime.datetime(years[0],1,1)) & (MET_hourly['date'] < datetime.datetime(years[-1]+1,1,1)), ['date','ws','wd']]        
            elif st == 'NB':
                data = NB_hourly.loc[(NB_hourly['date'] >= datetime.datetime(years[0],1,1)) & (NB_hourly['date'] < datetime.datetime(years[-1]+1,1,1)), ['date','ws','wd']] #datetime.datetime(years[-2],10,27)
            elif st == 'SM':
                data = SM.loc[(SM['date'] >= datetime.datetime(years[0],1,1)) & (SM['date'] < datetime.datetime(years[-1]+1,1,1)), ['date','ws','wd']]
            else:
                data = nve_hourly.loc[(nve_hourly['station_id'] == st) & (nve_hourly['date'] >= datetime.datetime(years[0],1,1)) & (nve_hourly['date'] < datetime.datetime(years[-1]+1,1,1)), ['date','ws','wd']]
            
            data = data.dropna(subset=['ws','wd'])
            
            sm = wind_rose(data['ws'], data['wd'], ax, speed_bins, cmap='summer_r', ec=c[s])
        
            merged_ws = pd.merge(ws_1pt[exp1][['date',st]], data[['date', 'ws']], on='date', how='inner')
            merged_ws['diff'] = merged_ws[st]-merged_ws['ws']
            bias = np.nanmean(merged_ws['diff'])
            abserror = np.nanmean(np.abs(merged_ws['diff']))    
            
            ax.set_title(f'{st}', color=c[s], fontsize=20, pad=10)
            
        elif row == 1:
            ax = axs[row,s]
            
            #lon, lat = WRF1000_ts.loc[WRF1000_ts['station_id']==st, 'grid_lon'], WRF1000_ts.loc[WRF1000_ts['station_id']==st, 'grid_lat']
            
            #sm = wind_rose(ws_1pt[st], wd_1pt[st], ax, speed_bins, cmap='summer', ec=c[s])
            #ax.set_title(st, fontsize=10, pad=10)
        
            
            merged_ws = pd.merge(ws_1pt[exp1][['date',st]], data[['date', 'ws']], on='date', how='inner')
            merged_wd = pd.merge(wd_1pt[exp1][['date',st]], data[['date', 'wd']], on='date', how='inner')
            
            if st != 'PEAK':
                wind_rose(merged_ws[st], merged_wd[st], ax, speed_bins, cmap='summer', ec=c[s])
            
            merged_ws['diff'] = merged_ws[st]-merged_ws['ws']
            bias = np.nanmean(merged_ws['diff'])
            abserror = np.nanmean(np.abs(merged_ws['diff']))    
            
            days = ((data['date'].reset_index(drop=True).iloc[-1]-data['date'].reset_index(drop=True).iloc[0]).days)
            
            #ax.set_title(f'WS error:\n{bias:.1f} / {abserror:.1f}', fontsize=10, pad=10)
            if st != 'PEAK':
                ax.text(
                    0.5, -0.15,                      # x, y in axes coords (0–1); y < 0 is below
                    #f'Bias / abs. WS error:\n
                    f'{bias:.1f} / {abserror:.1f} ms$^{{-1}}$\n({days} days)',
                    transform=ax.transAxes, ha='center', va='top', color=c[s], fontsize=11.8
                )
            else:
                wind_rose(ws_1pt[exp1][st], wd_1pt[exp1][st], ax, speed_bins, cmap='summer', ec=c[s])
                
                empty_ax = axs[0,s]
                
                #axs[0,s].set_visible(False)   # completely hide that axes
                
                empty_ax.set_facecolor('none')
                empty_ax.grid(False)
                
                empty_ax.set_xticks([])
                empty_ax.set_yticks([])
                empty_ax.set_xticklabels([])
                empty_ax.set_yticklabels([])
                
                for spine in empty_ax.spines.values():
                    spine.set_visible(False)
                
                axs[0,s].set_title(f'{st}', color=c[s], fontsize=20, pad=10)

        axs[0,0].set_ylabel(f'Obs.', color='k', fontsize=20)
        axs[1,0].set_ylabel(f'Model', color='k', fontsize=20)
        
norm       = mpl.colors.BoundaryNorm(speed_bins, plt.get_cmap('summer').N)

mappable = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('summer'))
mappable.set_array([])

cbar = fig.colorbar(mappable, ax=axs, orientation='vertical',
                    fraction=0.025, pad=0.02,
                    boundaries=speed_bins, ticks=speed_bins[:-1], extend='max')
cbar.set_label('Wind speed (m s$^{-1}$)', fontsize=14, rotation=270, labelpad=23)
cbar.ax.tick_params(labelsize=14)
#cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    
fig.tight_layout()

fig.savefig(f'figures/windroses.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

    




#%%

# # Oldedalen - temperature and precipitation for all wind regimes

# fig, ax = plt.subplots(figsize=(14,10))
# plt.rcParams.update({'font.size': 22})

# for i, cl in enumerate(t2m_daily['class']):
#     print (wind_color(cl))
#     ax.scatter(t2m_daily['OV'][i]-273.15, precip_daily['OD'][i], 
#                c=wind_color(cl), s=10)
# for cl in ['N', 'E', 'S', 'W']:
# #for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
#     ax.scatter((), (), c=wind_color(cl), label=cl)
# plt.legend()
# ax.set_xlabel('Mean daily temperature (\u00b0C)')
# ax.set_ylabel('Mean daily precipitation (mm)')
# if season == 'all':
#     ax.set_title(f'Mean values at Oldedalen (control experiment)')
# else:
#     ax.set_title(f'Mean {season} values at Oldedalen (control experiment)')
# plt.show()

#%%

# # Peak of ice cap - temperature and precipitation for all wind regimes

# fig, ax = plt.subplots(figsize=(14,10))
# plt.rcParams.update({'font.size': 22})

# for key in regimes.keys():
#     ax.scatter(WRF_temp_exp1[key][ice_mask].mean()-273.15, WRF_precip_exp1[key][ice_mask].mean(), 
#                c=wind_color(regimes[key]), s=60)
# for cl in ['N', 'E', 'S', 'W']:
# #for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
#     ax.scatter((), (), c=wind_color(cl), label=cl)
# plt.legend()
# ax.set_xlabel('Mean monthly temperature (\u00b0C)')
# ax.set_ylabel('Mean monthly precipitation (mm)')
# if season == 'all':
#     ax.set_title(f'Mean values over ice cap (control experiment)')
# else:
#     ax.set_title(f'Mean {season} values over ice cap (control experiment)')
# plt.show()


#%%

# differences in temperature and precipitation for all wind regimes

# fig, ax = plt.subplots(figsize=(14,10))
# plt.rcParams.update({'font.size': 22})

# for key in regimes.keys():
#     ax.scatter((WRF_temp_exp2[key][ice_mask]-WRF_temp_exp1[key][ice_mask]).mean(), 
#                (WRF_precip_exp2[key][ice_mask]-WRF_precip_exp1[key][ice_mask]).mean(), 
#                c=wind_color(regimes[key]), s=60)
# for cl in ['N', 'E', 'S', 'W']:
# #for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
#     ax.scatter((), (), c=wind_color(cl), label=cl)
# plt.legend()
# ax.set_xlabel('Difference in mean monthly temperature (\u00b0C)')
# ax.set_ylabel('Difference in mean monthly precipitation (mm)')
# if season == 'all':
#     ax.set_title(f'Difference in temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')
# else:
#     ax.set_title(f'Difference in {season} temperature/precipitation over ice cap\n for different wind regimes (no ice volume - control)')
# plt.show()


#%%

# Oldedalen temperature and precipitation for observations vs model and all wind regimes

# if season == 'all':
#     data = pd.merge(MET_monthly.loc[MET_monthly['station'] == 'OD'], nve_monthly.loc[nve_monthly['station_id'] == 'OV', ['date', 't']], on='date', how='inner')
# elif season == 'DJF':
#     data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'OD')&(MET_monthly['month'].isin([12,1,2]))], nve_monthly.loc[(nve_monthly['station_id'] == 'OV') & (nve_monthly['month'].isin([12,1,2])), ['date', 't']], on='date', how='inner')
# elif season == 'JJA':
#     data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'OD')&(MET_monthly['month'].isin([6,7,8]))], nve_monthly.loc[(nve_monthly['station_id'] == 'OV') & (nve_monthly['month'].isin([6,7,8])), ['date', 't']], on='date', how='inner')
# data = pd.merge(data, wind[['date', 'class']], on='date', how='inner')

# grid_i = WRF1000_ts.loc[WRF1000_ts['station_id']=='OD', 'grid_i']
# grid_j = WRF1000_ts.loc[WRF1000_ts['station_id']=='OD', 'grid_j']

# fig, ax = plt.subplots(figsize=(14,10))
# plt.rcParams.update({'font.size': 22})

# for key in regimes.keys():
#     ax.scatter(WRF_temp_exp1[key][grid_j, grid_i]-273.15, 
#                WRF_precip_exp1[key][grid_j, grid_i], 
#                c=wind_color(regimes[key]), s=60)
# for i, cl in enumerate(data['class']):
#     ax.scatter(data['t'][i], data['precip'][i], c='none', ec=wind_color(cl), marker='o', s=90)
# for cl in ['N', 'E', 'S', 'W']:
# #for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
#     ax.scatter((), (), c=wind_color(cl), label=cl)
# plt.legend()
# ax.set_xlabel('Mean monthly temperature (\u00b0C)')
# ax.set_ylabel('Mean monthly precipitation (mm)')
# if season == 'all':
#     ax.set_title(f'Mean values in Oldedalen (control experiment and observations)')
# else:
#     ax.set_title(f'Mean {season} values in Oldedalen (control experiment and observations)')
# plt.show()


#%%

# Fjærland temperature and precipitation for observations vs model and all wind regimes

# if season == 'all':
#     data = pd.merge(MET_monthly.loc[MET_monthly['station'] == 'FL'], wind[['date', 'class']], on='date', how='inner')
# elif season == 'DJF':
#     data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'FL')&(MET_monthly['month'].isin([12,1,2]))], wind[['date', 'class']], on='date', how='inner')
# elif season == 'JJA':
#     data = pd.merge(MET_monthly.loc[(MET_monthly['station'] == 'FL')&(MET_monthly['month'].isin([6,7,8]))], wind[['date', 'class']], on='date', how='inner')
    
# grid_i = WRF1000_ts.loc[WRF1000_ts['station_id']=='FL', 'grid_i']
# grid_j = WRF1000_ts.loc[WRF1000_ts['station_id']=='FL', 'grid_j']

# fig, ax = plt.subplots(figsize=(14,10))
# plt.rcParams.update({'font.size': 22})

# for key in regimes.keys():
#     ax.scatter(WRF_temp_exp1[key][grid_j, grid_i]-273.15, 
#                WRF_precip_exp1[key][grid_j, grid_i], 
#                c=wind_color(regimes[key]), s=60)
# for i, cl in enumerate(data['class']):
#     ax.scatter(data['temp'][i], data['precip'][i], c='none', ec=wind_color(cl), marker='o', s=90)
# for cl in ['N', 'E', 'S', 'W']:
# #for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
#     ax.scatter((), (), c=wind_color(cl), label=cl)
# plt.legend()
# ax.set_xlabel('Mean monthly temperature (\u00b0C)')
# ax.set_ylabel('Mean monthly precipitation (mm)')
# if season == 'all':
#     ax.set_title(f'Mean {season} values in Fjærland (control experiment and observations)')
# else:
#     ax.set_title(f'Mean {season} values in Fjærland (control experiment and observations)')
# plt.show()
