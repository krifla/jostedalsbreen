def shift_wind_direction(wind_directions):
    # Shift by 180 degrees and use modulo 360 to keep values in the proper range
    shifted_directions = (wind_directions + 180) % 360
    return shifted_directions

'''
load MET observations
'''

path_met = '/Users/kfha/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/ALT/JOSTICE/HPC, WRF and modelling/validation/MET_data'

MET_hourly = pd.read_csv(path_met+f"/obs_hourly.csv", sep='\t', decimal=',', na_values='-', skipfooter=1, engine='python')
MET_hourly = MET_hourly.rename({'Navn': 'station_name', 'Stasjon': 'station_id', 'Tid(norsk normaltid)': 'date', 'Lufttemperatur': 't', 'Relativ luftfuktighet': 'rh', 'Middelvind': 'ws', 'Vindretning': 'wd'}, axis=1)
MET_hourly['date'] = pd.to_datetime(MET_hourly['date'],format="%d.%m.%Y %H:%M")
MET_hourly['date'] -= pd.Timedelta(hours=1) # convert from utc+1 to utc

# add missing times for Fjærland and consider to fill variables with linearly interpolated values
df1 = (MET_hourly.loc[(MET_hourly['station_id']=='SN55430')])
df2 = (MET_hourly.loc[(MET_hourly['station_id']=='SN55820')])
df2.set_index('date', inplace=True)  # Set the date as the index
all_hours = pd.date_range(start=df2.index.min(), end=df2.index.max(), freq='h')
df2 = df2.reindex(all_hours)#[820:880]#[120:180]
df2['station_name'] = df2['station_name'].fillna(method='ffill')
df2['station_id'] = df2['station_id'].fillna(method='ffill')
df2.reset_index(inplace=True)
df2.rename(columns={'index': 'date'}, inplace=True)
MET_hourly = pd.concat([df1, df2])
MET_hourly.reset_index(inplace=False)
#MET_hourly.rename(columns={'index': 'date'}, inplace=True)

# add Spørteggbu

MET_add = pd.read_csv(path_met+f"/obs_hourly_SB.csv", sep=';', decimal=',', na_values='-', skipfooter=1, engine='python')
MET_add = MET_add.rename({'Navn': 'station_name', 'Stasjon': 'station_id', 'Tid(norsk normaltid)': 'date', 'Lufttemperatur': 't', 'Middelvind': 'ws', 'Vindretning': 'wd'}, axis=1)
MET_add['date'] = pd.to_datetime(MET_add['date'],format="%d.%m.%Y %H:%M")
MET_add['date'] -= pd.Timedelta(hours=1) # convert from utc+1 to utc

MET_hourly = pd.concat([MET_hourly,MET_add])

#MET_hourly['date'] = pd.to_datetime(MET_hourly['date'],format="%d.%m.%Y %H:%M")
MET_hourly['year'] = MET_hourly['date'].dt.year.convert_dtypes()
MET_hourly['month'] = MET_hourly['date'].dt.month.convert_dtypes()
MET_hourly['day'] = MET_hourly['date'].dt.day.convert_dtypes()
MET_hourly['hour'] = MET_hourly['date'].dt.hour.convert_dtypes()
MET_hourly = MET_hourly.sort_values(by=['station_name', 'date'],ascending=[False, True])

MET_hourly.loc[MET_hourly['ws']<1, 'wd'] = np.nan


MET_hourly_precip = pd.read_csv(path_met+f"/MET_hourly_precip_FL.csv", sep=';', decimal=',', na_values='-', skipfooter=1, engine='python')
MET_hourly_precip = MET_hourly_precip.rename({'Navn': 'station_name', 'Stasjon': 'station_id', 'Tid(norsk normaltid)': 'date', 'Nedbør (1 t)': 'precip'}, axis=1)
MET_hourly_precip['date'] = pd.to_datetime(MET_hourly_precip['date'],format="%d.%m.%Y %H:%M")
MET_hourly_precip['date'] -= pd.Timedelta(hours=1) # convert from utc+1 to utc


MET_add = pd.read_csv(path_met+f"/MET_hourly_precip_MG.csv", sep=';', decimal=',', na_values='-', skipfooter=1, engine='python')
MET_add = MET_add.rename({'Navn': 'station_name', 'Stasjon': 'station_id', 'Tid(norsk normaltid)': 'date', 'Nedbør (1 t)': 'precip'}, axis=1)
MET_add['date'] = pd.to_datetime(MET_add['date'],format="%d.%m.%Y %H:%M")
MET_add['date'] -= pd.Timedelta(hours=1) # convert from utc+1 to utc

MET_hourly_precip = pd.concat([MET_hourly_precip,MET_add])

MET_hourly_precip['year'] = MET_hourly_precip['date'].dt.year.convert_dtypes()
MET_hourly_precip['month'] = MET_hourly_precip['date'].dt.month.convert_dtypes()
MET_hourly_precip['day'] = MET_hourly_precip['date'].dt.day.convert_dtypes()
MET_hourly_precip['hour'] = MET_hourly_precip['date'].dt.hour.convert_dtypes()
MET_hourly_precip = MET_hourly_precip.sort_values(by=['station_name', 'date'],ascending=[False, True])



MET_monthly = pd.read_csv(path_met+f"/MET_monthly.csv", sep=';', decimal=',', na_values='-')
MET_monthly = MET_monthly.rename({'Navn': 'station_name', 'Stasjon': 'station_id', 'Tid(norsk normaltid)': 'date', 'Middeltemperatur (mnd)': 'temp', 'Nedbør (mnd)': 'precip'}, axis=1)
MET_monthly['date'] = pd.to_datetime(MET_monthly['date'],format="%m.%Y")
MET_monthly['year'] = MET_monthly['date'].dt.year.convert_dtypes()
MET_monthly['month'] = MET_monthly['date'].dt.month.convert_dtypes()
MET_monthly['day'] = MET_monthly['date'].dt.day.convert_dtypes()
MET_monthly['hour'] = MET_monthly['date'].dt.hour.convert_dtypes()

# add short station label
MET_monthly = MET_monthly.iloc[:-1]
MET_monthly.loc[MET_monthly['station_id']=='SN55430','station'] = 'MG'
MET_monthly.loc[MET_monthly['station_id']=='SN55820','station'] = 'FL'
MET_monthly.loc[MET_monthly['station_id']=='SN58480','station'] = 'OD'
#MET_monthly.loc[MET_monthly['station_id']=='SN58670','station'] = 'FN'
MET_monthly.loc[MET_monthly['station_id']=='SN55670','station'] = 'VS'
#MET_monthly.loc[MET_monthly['station_id']=='SN55840','station'] = 'FL_old'
MET_monthly.loc[MET_monthly['station_id']=='SN57390','station'] = 'SJ'
MET_monthly.loc[MET_monthly['station_id']=='SN55420','station'] = 'JD'
MET_monthly.loc[MET_monthly['station_id']=='SN55425','station'] = 'SB'


'''
load NVE observations
'''

path_nve = '/Users/kfha/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/ALT/JOSTICE/HPC, WRF and modelling/validation/NVE_data'

nve = pd.read_csv(path_nve+f"/NVE_OV.csv", sep=';', skiprows=1, decimal=',', usecols=[0,1])#, na_values='-')
nve = nve.rename({'Tidspunkt': 'date', 'Lufttemperatur (°C)': 't'}, axis=1)
nve['date'] = pd.to_datetime(nve['date']).dt.tz_localize(None)#['t']
nve['year'] = nve['date'].dt.year
nve['month'] = nve['date'].dt.month
nve['day'] = nve['date'].dt.day
nve['station_id'] = 'OV'
#print (nve['year'].unique())
#nve['date']

nve_add = pd.read_csv(path_nve+f"/NVE_LV.csv", sep=';', skiprows=1, usecols=[0,1], decimal=',')#, na_values='-')
nve_add = nve_add.rename({'Tidspunkt': 'date', 'Lufttemperatur (°C)': 't'}, axis=1)
nve_add['date'] = pd.to_datetime(nve_add['date']).dt.tz_localize(None)#['t']
nve_add['year'] = nve_add['date'].dt.year
nve_add['month'] = nve_add['date'].dt.month
nve_add['day'] = nve_add['date'].dt.day
nve_add['station_id'] = 'LV'
#print (nve_add['year'].unique())

nve = pd.concat([nve,nve_add])

nve_add = pd.read_csv(path_nve+f"/NVE_FB.csv", sep=';', skiprows=1, usecols=[0,1], decimal=',')#, na_values='-')
nve_add = nve_add.rename({'Tidspunkt': 'date', 'Lufttemperatur (°C)': 't'}, axis=1)
nve_add['date'] = pd.to_datetime(nve_add['date']).dt.tz_localize(None)#['t']
nve_add['year'] = nve_add['date'].dt.year
nve_add['month'] = nve_add['date'].dt.month
nve_add['day'] = nve_add['date'].dt.day
nve_add['station_id'] = 'FB'

# manually remove some periods with wrong data
nve_add.loc[(nve_add['date'] >= datetime.datetime(2013,6,1)) & (nve_add['date'] < datetime.datetime(2013,9,1)),'t'] = np.nan
nve_add.loc[(nve_add['date'] >= datetime.datetime(2013,12,1)) & (nve_add['date'] < datetime.datetime(2014,1,1)) & (nve_add['t'] < -20),'t'] = np.nan
nve_add.loc[(nve_add['date'] >= datetime.datetime(2013,12,1)) & (nve_add['date'] < datetime.datetime(2014,1,1)) & (nve_add['t'] > 20),'t'] = np.nan
nve_add.loc[(nve_add['date'] >= datetime.datetime(2014,1,1)) & (nve_add['date'] < datetime.datetime(2014,2,1)),'t'] = np.nan
#print (nve_add['year'].unique())

nve = pd.concat([nve,nve_add])

nve_add = pd.read_csv(path_nve+f"/NVE_AS.csv", sep=';', skiprows=1, usecols=[0,1], decimal=',')#, na_values='-')
nve_add = nve_add.rename({'Tidspunkt': 'date', 'Lufttemperatur (°C)': 't'}, axis=1)
nve_add['date'] = pd.to_datetime(nve_add['date']).dt.tz_localize(None)#['t']
nve_add['year'] = nve_add['date'].dt.year
nve_add['month'] = nve_add['date'].dt.month
nve_add['day'] = nve_add['date'].dt.day
nve_add['station_id'] = 'AS'
#print (nve_add['year'].unique())

nve = pd.concat([nve,nve_add])

# remove unrealistically low temperatures
nve.loc[nve['t']<-100, 't'] = np.nan

# remove short period with uncertain temperature measurements
nve = nve[~((nve['year']==2015)&(nve['station_id']=='FB')&(nve['month'].isin([8,9])))]


nve_hourly = pd.read_csv(path_nve+f"/FB_WS.csv", sep=';', skiprows=1, usecols=[0,1], decimal=',')#, na_values='-')
nve_hourly = nve_hourly.rename({'Tidspunkt': 'date', 'Vindhastighet (m/s)': 'ws'}, axis=1)
nve_hourly['date'] = pd.to_datetime(nve_hourly['date']).dt.tz_localize(None)#['t']
nve_hourly['year'] = nve_hourly['date'].dt.year
nve_hourly['month'] = nve_hourly['date'].dt.month
nve_hourly['day'] = nve_hourly['date'].dt.day
nve_hourly['station_id'] = 'FB'

nve_add = pd.read_csv(path_nve+f"/FB_WD.csv", sep='\t', skiprows=1, usecols=[0,1], decimal=',')
nve_add = nve_add.rename({'Tidspunkt': 'date', 'Vindretning (°)': 'wd'}, axis=1)
nve_add['date'] = pd.to_datetime(nve_add['date']).dt.tz_localize(None)
nve_hourly = pd.merge(nve_hourly, nve_add[['date','wd']], on='date', how='outer')


# manually remove some periods with wrong data
#nve_add.loc[(nve_add['date'] >= datetime.datetime(2013,6,1)) & (nve_add['date'] < datetime.datetime(2013,9,1)),'t'] = np.nan
#print (nve_add['year'].unique())

nve_add = pd.read_csv(path_nve+f"/AS_WS.csv", sep=';', skiprows=1, usecols=[0,1], decimal=',')
nve_add = nve_add.rename({'Tidspunkt': 'date', 'Vindhastighet (m/s)': 'ws'}, axis=1)
nve_add['date'] = pd.to_datetime(nve_add['date']).dt.tz_localize(None)#['t']
nve_add['year'] = nve_add['date'].dt.year
nve_add['month'] = nve_add['date'].dt.month
nve_add['day'] = nve_add['date'].dt.day
nve_add['station_id'] = 'AS'
#print (nve_add['year'].unique())

nve_add2 = pd.read_csv(path_nve+f"/AS_WD.csv", sep=';', skiprows=1, usecols=[0,1], decimal=',')
nve_add2 = nve_add2.rename({'Tidspunkt': 'date', 'Vindretning (°)': 'wd'}, axis=1)
nve_add2['date'] = pd.to_datetime(nve_add2['date']).dt.tz_localize(None)
nve_add = pd.merge(nve_add, nve_add2[['date','wd']], on='date', how='outer')

nve_hourly = pd.concat([nve_hourly,nve_add])

# remove unrealistic wind measurements
nve_hourly.loc[nve_hourly['wd']>360, ['ws','wd']] = np.nan
nve_hourly.loc[nve_hourly['ws']<1, 'wd'] = np.nan
#nve_hourly.loc[(nve_hourly['date'].dt.year.isin([2009,2010]))&(nve_hourly['station_id']=='FB'),'ws'] = np.nan
nve_hourly.loc[(nve_hourly['date']>datetime.datetime(2008,2,25))&(nve_hourly['date']<datetime.datetime(2011,5,1))&(nve_hourly['station_id']=='FB'),['ws','wd']] = np.nan

# remove short period with uncertain temperature measurements
nve_hourly = nve_hourly[~((nve_hourly['year']==2015)&(nve_hourly['station_id']=='FB')&(nve_hourly['month'].isin([8,9])))]


# create monthly mean temperature from nve
# FN station could be added
nve['date'] = pd.to_datetime(nve['date'])
nve_monthly = nve.groupby(['station_id', pd.Grouper(key='date', freq='ME')]).mean().reset_index()
nve_monthly['date'] -= pd.offsets.MonthBegin()



'''
load observations from Nigardsbreen
'''

path_hvl = '/Users/kfha/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/ALT/JOSTICE/HPC, WRF and modelling/validation/HVL_data'

NB_hourly = (pd.read_csv(path_hvl+f"/AWS_NB_hourly.csv", sep=','))
NB_hourly['date'] = pd.to_datetime(NB_hourly['time'])
#NB_hourly['date'] -= pd.Timedelta(minutes=30) # center hourly average
NB_hourly = NB_hourly.loc[np.where(NB_hourly['date'] >= pd.Timestamp('2021-07-01 00:00:00'))[0][0]:]
NB_hourly['station_id'] = 'NB'

NB_hourly = NB_hourly.rename(columns={'p_u': 'p', 't_u': 't', 'rh_u': 'rh_orig', 'rh_u_cor': 'rh', 'qh_u': 'qh', 'wspd_u': 'ws', 'wdir_u': 'wd'})
NB_hourly['wd'] = shift_wind_direction(NB_hourly['wd'])



# create monthly mean temperature from NB
NB_monthly = NB_hourly[['date', 't']]
NB_monthly.set_index('date', inplace=True)
NB_monthly = NB_monthly.resample('ME').mean()
NB_monthly.index = NB_monthly.index.to_period('M').to_timestamp()
NB_monthly.reset_index(inplace=True)


'''
load observations from Steinmannen
'''

path = '/Users/kfha/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/ALT/JOSTICE/Steinmannen'

#from datetime import datetime.strptime

SM = pd.read_excel(path+'/Steinmannen.xlsx')
#SM = (pd.read_csv(f"data_steinmann_2021-2022.csv", sep=','))
SM = (SM.replace(' ', np.nan, regex=True))
SM = SM.rename(columns={'Time': 'date', 'T0008A3R 0119 [W/m2]': 'rad', 'T0014A3K 0113 [deg]': 'wd', 'T0014A3K 0113 [deg] orig': 'wd orig', 'T0015A3K 0120 [m/s]': 'ws', 'T0015A3K 0120 [m/s] orig': 'ws orig', 'T0017A3K 0114 [deg C]': 't', 'T0017A3K 0114 [deg C] orig': 't orig'})
for i in range(len(SM['date'])):
    try:
        SM['date'][i] = datetime.strptime(SM['date'][i],'%d/%m/%Y/%H')
    except:
        if SM['date'][i][-2:] == '24':
            SM['date'][i] = SM['date'][i][:-2]+'00' # replacing 24 by 00 for correct time format
            SM['date'][i] = pd.to_datetime(SM['date'][i],format="%d/%m/%Y/%H")+pd.to_timedelta(1,'d') # adding one day
        else:
            SM['date'][i] = pd.to_datetime(SM['date'][i],format="%d/%m/%Y/%H")
SM['date'] = pd.to_datetime(SM['date'])
SM['station_id'] = 'SM'
#SM['date']

# remove unrealistic wind measurements
SM.loc[SM['wd']<0, ['ws','wd']] = np.nan
SM.loc[SM['ws']<1, 'wd'] = np.nan
SM.loc[SM['ws']==4.4, ['ws','wd']] = np.nan



'''
load mass balance measurements from NVE
'''

path = '/Users/kfha/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/ALT/JOSTICE/massebalanse_jostedalsbreen'

massbalance = pd.read_csv(path+'/2025-12-02_SWE_from_density_2006-2024.csv')

massbalance['date'] = pd.to_datetime(massbalance['dt_observation_date'])
massbalance['year'] = (massbalance['date'].dt.year)

from pyproj import Proj, transform
utm_proj = Proj(proj='utm', zone=32, ellps='WGS84')
wgs84_proj = Proj(proj='latlong', datum='WGS84')
massbalance['lon'], massbalance['lat'] = transform(utm_proj, wgs84_proj, massbalance['utm_east'].values, massbalance['utm_north'].values)

massbalance = massbalance[['stake_no', 'SWE(mm)', 'date', 'year', 'lon', 'lat']]


'''
load station location relative to model grid 
'''

WRF1000_ts = pd.read_csv('../../WRF_general_output/tslist/WRF1000_ts.csv')

# WRF1000_ts = pd.DataFrame()

# dirs = ['../../WRF_general_output/tslist/']
stations = ['MG','NB','SM','FL','OD','VS','FN','OV','LV','SV','FL_ol','JD','AS','NV','BH','FB','SJ','SB','PEAK']

# WRF_tmp = pd.DataFrame()

# for d in dirs:
#     for loc in stations[:-1]:
#         with open(f'{d}{loc}.d03.TS') as f:
#             firstline = f.readline().strip('\n')
        
#         WRF_tmp['date'] = [pd.to_datetime(firstline[-19:],format="%Y-%m-%d_%H:%M:%S")]###+pd.to_timedelta(WRF_tmp['ts_hour'],'h')
#         WRF_tmp['station_id'] = [loc]
#         WRF_tmp['station_lat'] = [float(firstline[firstline.find('(')+2:firstline.find('(')+8])]
#         WRF_tmp['station_lon'] = [float(firstline[firstline.find('(')+12:firstline.find('(')+17])]
#         WRF_tmp['grid_lat'] = [float(firstline[firstline.find('(')+32:firstline.find('(')+39])]
#         WRF_tmp['grid_lon'] = [float(firstline[firstline.find('(')+42:firstline.find('(')+48])]
#         WRF_tmp['grid_i'] = [int(firstline[firstline.find('(')+21:firstline.find('(')+24])-1] # -1 for right indexing starting with 0
#         WRF_tmp['grid_j'] = [int(firstline[firstline.find('(')+26:firstline.find('(')+29])-1] # -1 for right indexing starting with 0

#         if d == dirs[0]:
#             WRF1000_ts = pd.concat([WRF1000_ts,WRF_tmp])
            
# peak = pd.DataFrame([{'station_id': 'PEAK', 'station_lat': 61.6766, 'station_lon': 7.0188, 'grid_lat': 61.677, 'grid_lon': 7.019, 'grid_i': 76, 'grid_j': 55}])
# WRF1000_ts = pd.concat([WRF1000_ts, peak], ignore_index=True)

# del WRF_tmp


# # apply grid and station height to each station
# for loc, i, j in zip(WRF1000_ts['station_id'],WRF1000_ts['grid_i'],WRF1000_ts['grid_j']):
#     WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_hgt'] = WRF_hgt_exp1.sel(south_north=j, west_east=i).values
#     lat = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_lat'].values[0]; lon = WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_lon'].values[0]
#     WRF1000_ts.loc[WRF1000_ts['station_id']==loc,'grid_hgt_4pts'] = InverseDistanceWeighted(lat=lat, lon=lon, wind=False, precip=False, var='HGT', data=WRF)[0]
# WRF1000_ts.loc[WRF1000_ts['station_id']=='MG','station_hgt'] = 305
# WRF1000_ts.loc[WRF1000_ts['station_id']=='FL','station_hgt'] = 3
# WRF1000_ts.loc[WRF1000_ts['station_id']=='AS','station_hgt'] = 446
# WRF1000_ts.loc[WRF1000_ts['station_id']=='LV','station_hgt'] = 56
# WRF1000_ts.loc[WRF1000_ts['station_id']=='OV','station_hgt'] = 47
# WRF1000_ts.loc[WRF1000_ts['station_id']=='FB','station_hgt'] = 992
# WRF1000_ts.loc[WRF1000_ts['station_id']=='JD','station_hgt'] = 243
# WRF1000_ts.loc[WRF1000_ts['station_id']=='NB','station_hgt'] = 560 # approx. due to moving glacier

    
'''
define WRF and precipitation data to be analysed
'''
    

def defineStaticData(exp1='glac2006', exp2='noice_BT'):
       

    inpath = '../../WRF_general_output/static/'

    # assign static data

    WRF = xr.open_mfdataset(inpath+'wrfout_d03_static')
    WRF_hgt_exp1 = np.squeeze(WRF['HGT'])
    WRF_lu_exp1  = np.squeeze(WRF['LU_INDEX'])

    WRF_hgt_exp2 = np.squeeze(WRF['HGT'])
    WRF_lu_exp2  = np.squeeze(WRF['LU_INDEX'])

    if exp1 == 'glac2100_dtm50' or exp1 == 'glac2100_dem2100':
        WRF = xr.open_mfdataset(f'{inpath}/wrfout_d03_static_dem2100')
        WRF_lu_exp1  = np.squeeze(WRF['LU_INDEX'])
        if exp1 == 'glac2100_dem2100':
            WRF_hgt_exp1 = np.squeeze(WRF['HGT'])
    elif exp1 == 'noice_dtm50' or (exp1 == 'noice_BT'):# and exp2 == 'modlakes_noice_BT'):
        WRF = xr.open_mfdataset(f'{inpath}/wrf_static_d03_nomodlakes_noice_BT')
        WRF_lu_exp1 = np.squeeze(WRF['LU_INDEX'])
        if exp1 == 'noice_BT' or exp1 == 'modlakes_noice_BT':
            WRF_hgt_exp1 = np.squeeze(WRF['HGT'])

    if exp2 == 'glac2100_dtm50' or exp2 == 'glac2100_dem2100':
        WRF = xr.open_mfdataset(f'{inpath}/wrfout_d03_static_dem2100')
        WRF_lu_exp2  = np.squeeze(WRF['LU_INDEX'])
        if exp2 == 'glac2100_dem2100':
            WRF_hgt_exp2 = np.squeeze(WRF['HGT'])
    elif exp2 == 'noice_dtm50' or exp2 == 'noice_BT':
        WRF = xr.open_mfdataset(f'{inpath}/wrf_static_d03_nomodlakes_noice_BT')
        WRF_lu_exp2  = np.squeeze(WRF['LU_INDEX'])
        if exp2 == 'noice_BT':
            WRF_hgt_exp2 = np.squeeze(WRF['HGT'])
    elif exp2 == 'modlakes_noice_BT':
        WRF = xr.open_mfdataset(f'{inpath}/wrfout_d03_static_modlakes_noice')
        WRF_lu_exp2  = np.squeeze(WRF['LU_INDEX'])
        if exp2 == 'noice_BT' or exp1 == 'modlakes_noice_BT':
            WRF_hgt_exp2 = np.squeeze(WRF['HGT'])

    #WRF = xr.open_mfdataset(f'{inpath}/{variable1}_{y}-{m}_{exp1}')
    WRF_lon = np.squeeze(WRF['XLONG'])
    WRF_lat = np.squeeze(WRF['XLAT'])    
    
    return (WRF_hgt_exp1, WRF_hgt_exp2, WRF_lu_exp1, WRF_lu_exp2, WRF_lon, WRF_lat)
    
    
'''
define WRF and precipitation data to be analysed
'''
    
years = np.arange(2007, 2023)

def definePrecipData(years=years, season='SONDJFMAM', #months=months, start=start, end=end, 
               exp1='glac2006', exp2='noice_BT', variable1='RAINNC', variable2='SNOWNC'):

    print (years) 
    
    if season == 'SONDJFMAM':
        months = [1,2,3,4,5, 9,10,11,12] # np.arange(1,13)
        start = datetime.datetime(years[0],9,1)
        end = datetime.datetime(years[-1],5,1)
    elif season == 'DJF':
        months = [1,2,12]
        start = datetime.datetime(years[0],1,1)
        end = datetime.datetime(years[-1],12,1)
    elif season == 'MAM':
        months = np.arange(3,6)
        start = datetime.datetime(years[0],3,1)
        end = datetime.datetime(years[-1],5,1)
    elif season == 'JJA':
        months = np.arange(6,9)
        start = datetime.datetime(years[0],6,1)
        end = datetime.datetime(years[-1],8,1)
    elif season == 'SON':
        months = np.arange(9,12)
        start = datetime.datetime(years[0],9,1)
        end = datetime.datetime(years[-1],11,1)
    elif season == 'MAMJJA':
        months = np.arange(3,9)
        start = datetime.datetime(years[0],3,1)
        end = datetime.datetime(years[-1],8,1)
    elif season == 'JFMAMJJA':
        months = np.arange(1,9)
        start = datetime.datetime(years[0],1,1)
        end = datetime.datetime(years[-1],8,1)
    elif season == 'all':
        months = np.arange(1,13)
        start = datetime.datetime(years[0],1,1)
        end = datetime.datetime(years[-1],12,1)
        

    WRF_precip_exp1 = None
    WRF_precip_exp2 = None
    WRF_precip_var2_exp1 = None
    WRF_precip_var2_exp2 = None
    obs_precip = np.zeros((8))

    inpath = '../../output_subsets/maps/precip'

    for y in years:
        for m in months:
            date = datetime.datetime(y,m,1)
            if date < start or date > end:
                continue

            m = "{:02d}".format(m)
            #print (f'{y}-{m}')

            # for experiment 1
            
            WRF = xr.open_mfdataset(f'{inpath}/{variable1}_{y}-{m}_{exp1}')
            #WRF_lon = np.squeeze(WRF['XLONG'])
            #WRF_lat = np.squeeze(WRF['XLAT'])
            #WRF_hgt = np.squeeze(WRF['HGT_M'])

            WRF_precip_tmp = np.squeeze(WRF[variable1])
            if WRF_precip_exp1 is None:
                WRF_precip_exp1 = {} #np.zeros_like(WRF_precip_tmp)
            #WRF_precip_exp1 += WRF_precip_tmp.values
            WRF_precip_exp1[f'{y}_{m}'] = WRF_precip_tmp.values

            
            # for experiment 2

            WRF = xr.open_mfdataset(f'{inpath}/{variable1}_{y}-{m}_{exp2}')

            WRF_precip_tmp = np.squeeze(WRF[variable1])
            if WRF_precip_exp2 is None:
                WRF_precip_exp2 = {} #np.zeros_like(WRF_precip_tmp)
            #WRF_precip_exp2 += WRF_precip_tmp.values
            WRF_precip_exp2[f'{y}_{m}'] = WRF_precip_tmp.values

            # for experiment 1 and variable 2

            WRF = xr.open_mfdataset(f'{inpath}/{variable2}_{y}-{m}_{exp1}')

            WRF_precip_tmp = np.squeeze(WRF[variable2])
            if WRF_precip_var2_exp1 is None:
                WRF_precip_var2_exp1 = {} #np.zeros_like(WRF_precip_tmp)
            #WRF_precip_var2_exp1 += WRF_precip_tmp.values
            WRF_precip_var2_exp1[f'{y}_{m}'] = WRF_precip_tmp.values

            # for experiment 2 and variable 2

            WRF = xr.open_mfdataset(f'{inpath}/{variable2}_{y}-{m}_{exp2}')

            WRF_precip_tmp = np.squeeze(WRF[variable2])
            if WRF_precip_var2_exp2 is None:
                WRF_precip_var2_exp2 = {} #np.zeros_like(WRF_precip_tmp)
            #WRF_precip_var2_exp2 += WRF_precip_tmp.values
            WRF_precip_var2_exp2[f'{y}_{m}'] = WRF_precip_tmp.values

        # adding winter mass balance observations

        if y > years[0]:
            obs_precip[6] += massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('A')),'SWE(mm)']
            if y == 2018: # several option in 2018
                obs_precip[7] += massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N9')),'SWE(mm)']
            else:
                obs_precip[7] += massbalance.loc[(massbalance['year']==y)&(massbalance['stake_no'].str.startswith('N')),'SWE(mm)']

    # adding MET observations

    for s, st in enumerate(['MG', 'JD', 'FL', 'VS', 'OD', 'SJ']):
        if st not in [' ']:
            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'].values[0])
            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'].values[0])
            for y in years:
                for m in months:
                    date = datetime.datetime(y,m,1)
                    if date < start or date > end:
                        continue
                    mask = (MET_monthly['station'] == st) & (MET_monthly['year'] == y) & (MET_monthly['month'] == m) # | ((MET_monthly['year'] == y-1)&(MET_monthly['month'].isin(months_prevyear))))
                    if st not in ['MG', 'JD'] or (st == 'MG' and y == 2021) or (st == 'JD' and y in list(np.arange(2016,2020))): # MG only good precip data for 2021, JD for 2016-2019
                        obs_precip[s] += MET_monthly[mask]['precip'].sum()

                    
    return (WRF_precip_exp1, WRF_precip_exp2, WRF_precip_var2_exp1, WRF_precip_var2_exp2, obs_precip)    
    #return (WRF_hgt_exp1, WRF_hgt_exp2, WRF_lu_exp1, WRF_lu_exp2, WRF_lon, WRF_lat, WRF_precip_exp1, WRF_precip_exp2, WRF_precip_var2_exp1, obs_precip)


'''
define temperature data to be analysed
'''
    
years = np.arange(2007, 2023)

def defineTempData(years=years, season='SONDJFMAM', #months=months, start=start, end=end, 
               exp1='glac2006', exp2='noice_BT', variable3='T2'):

    if season == 'SONDJFMAM':
        months = [1,2,3,4,5, 9,10,11,12] # np.arange(1,13)
        start = datetime.datetime(years[0],9,1)
        end = datetime.datetime(years[-1],5,1)
    elif season == 'DJF':
        months = [1,2,12]
        start = datetime.datetime(years[0],1,1)
        end = datetime.datetime(years[-1],12,1)
    elif season == 'MAM':
        months = np.arange(3,6)
        start = datetime.datetime(years[0],3,1)
        end = datetime.datetime(years[-1],5,1)
    elif season == 'JJA':
        months = np.arange(6,9)
        start = datetime.datetime(years[0],6,1)
        end = datetime.datetime(years[-1],8,1)
    elif season == 'SON':
        months = np.arange(9,12)
        start = datetime.datetime(years[0],9,1)
        end = datetime.datetime(years[-1],11,1)
    elif season == 'MAMJJA':
        months = np.arange(3,9)
        start = datetime.datetime(years[0],3,1)
        end = datetime.datetime(years[-1],8,1)
    elif season == 'JFMAMJJA':
        months = np.arange(1,9)
        start = datetime.datetime(years[0],1,1)
        end = datetime.datetime(years[-1],8,1)
    elif season == 'all':
        months = np.arange(1,13)
        start = datetime.datetime(years[0],1,1)
        end = datetime.datetime(years[-1],12,1)
        

    WRF_temp_exp1 = {}
    WRF_temp_exp2 = {}
    obs_temp = {}

    inpath = '../../output_subsets/maps/temp'

    for y in years:
        for m in months:
            date = datetime.datetime(y,m,1)
            if date < start or date > end:
                continue

            m = "{:02d}".format(m)
            #print (f'{y}-{m}')

            # for experiment 1
            
            WRF = xr.open_mfdataset(f'{inpath}/{variable3}_{y}-{m}_{exp1}')
            WRF_temp_tmp = np.squeeze(WRF[variable3])
            if WRF_temp_exp1 is None:
                WRF_temp_exp1 = {}
            WRF_temp_exp1[f'{y}_{m}'] = WRF_temp_tmp.values

            # for experiment 2
            
            WRF = xr.open_mfdataset(f'{inpath}/{variable3}_{y}-{m}_{exp2}')
            WRF_temp_tmp = np.squeeze(WRF[variable3])
            if WRF_temp_exp2 is None:
                WRF_temp_exp2 = {}
            WRF_temp_exp2[f'{y}_{m}'] = WRF_temp_tmp.values

    # adding MET observations

    for s, st in enumerate(stations):#['MG', 'FL', 'VS', 'OD', 'SJ']):
        if st not in [' ']:
            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'])#[0])
            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'])#[0])
            for y in years:
                for m in months:
                    m2 = "{:02d}".format(m)
                    #obs_temp[f'{y}_{m}'] = np.nan*np.ones((len(stations))) # set up df
                    date = datetime.datetime(y,m,1)
                    if date < start or date > end:
                        continue
                    if f'{y}_{m2}' not in obs_temp:
                        obs_temp[f'{y}_{m2}'] = np.nan*np.ones((len(stations)))#[np.nan] * len(stations)
                        
                    # check nve_monthly
                    mask = (nve_monthly['station_id'] == st) & (nve_monthly['year'] == y) & (nve_monthly['month'] == m)
                    if not nve_monthly[mask]['t'].empty:
                        #print (st, y, m2, nve_monthly[mask]['t'].values[0])
                        obs_temp[f'{y}_{m2}'][s] = nve_monthly[mask]['t'].values[0]
                        
                    # check MET_monthly
                    mask = (MET_monthly['station'] == st) & (MET_monthly['year'] == y) & (MET_monthly['month'] == m)
                    if not MET_monthly[mask]['temp'].empty:
                        #print (st, y, m2, MET_monthly[mask]['temp'].values[0])
                        obs_temp[f'{y}_{m2}'][s] = MET_monthly[mask]['temp'].values[0]

                    
    return (WRF_temp_exp1, WRF_temp_exp2, obs_temp)







