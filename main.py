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

# define experiments to compare
exp1='glac2019'; exp2='noice_BT' # will be changed later


if __name__ == "__main__":
    run_script('loadData.py') # load observational data and define functions for downloading model data
    run_script('weighted4pts.py') # define function for finding inverse distance weighted data


#%%

# define experiments to compare
exp1='glac2019'; exp2='glac2100_dem2100'#noice_dtm50'#noice_BT'#
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
# plotSnowAbs()
# plotPrecipDiff(vmin=-10, vmax=10)
# plotSnowDiff(vmin=-50, vmax=50)
# plotRainDiff(vmin=-50, vmax=50)

# plotTempAbs(temp_model = WRF_temp_exp1, vmin=0,vmax=14)
# plotTempDiff(vmin=-1, vmax=1)

# plotLUDiff()
# plotLUDiff3D()
# plotHGTDiff()
# plotHGTDiff3D()

#%%

# load model time series

t2m_1pt, t2m_1pt_monthly, t2m_monthly_obs = loadModelTemperatureTimeseries(experiments = [exp1, exp2])
precip_1pt, ws_1pt, wd_1pt = loadModelPrecipWindTimeseries(experiments = [exp1, exp2])


#%%

# validation plots

names, error_1pt = plotModelErrorTimeseries()
plotModelMonthlyError()

#%%

# prepare data

wind = classifyWind()

#plotAllYearsMonths()

regimes_daily = defineDailyRegimes()

#ice_low_mask, ice_high_mask, noice_low_mask, noice_high_mask = defineMasks()

t2m_daily, precip_daily, t2m_exp2_daily, precip_exp2_daily = defineDailyData(exp1=exp1, exp2=exp2)

#%%

# wind regime plots

x_values_N, x_values_E, x_values_S, x_values_W, y_values_N, y_values_E, y_values_S, y_values_W = plotTemperatureWindRegime()
testStatistics()

x_values_N, x_values_E, x_values_S, x_values_W, y_values_N, y_values_E, y_values_S, y_values_W = plotPrecipitationWindRegime()
testStatistics()

#%%

# wind rose plot

ws_1pt, wd_1pt, nve_hourly, MET_hourly, NB_hourly, SM = createSeasonSubset(summer = False, winter = False):
prepareDataWindRoses()
plotWindRoses()


