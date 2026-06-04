'''
load station location relative to model grid 
'''

# WRF1000_ts = pd.DataFrame()

# dirs = ['../../WRF_general_output/tslist/']
# stations = ['MG','NB','SM','FL','OD','VS','FN','OV','LV','SV','FL_ol','JD','AS','NV','BH','FB','SJ','SB','PEAK']

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

# WRF1000_ts = WRF1000_ts.reset_index(drop=True)
            
# peak = pd.DataFrame([{'station_id': 'PEAK', 'station_lat': 61.6766, 'station_lon': 7.0188, 'grid_lat': 61.677, 'grid_lon': 7.019, 'grid_i': 76, 'grid_j': 55}])
# WRF1000_ts = pd.concat([WRF1000_ts, peak], ignore_index=True)

# #stations = stations.append('PEAK')

# del WRF_tmp


# --------------------------------------------------------------

def findClosestGridPoints(lat=61.659, lon=7.276, data=WRF_3D):
    
    #la,lo = wrf.ll_to_xy(data,lat,lon)[::-1].values
    for lat_idx in np.argwhere(np.round(XLAT, 3) == lat):
        for lon_idx in np.argwhere(np.round(XLONG, 3) == lon):
            if lat_idx[0] == lon_idx[0] and lat_idx[1] == lon_idx[1]:
                la, lo = lat_idx[0], lat_idx[1]
    
    if   ((data['XLAT'][0][la,lo] < lat) & (data['XLONG'][0][la,lo] < lon)):
        #print ('<<', la, lo, la+1, lo+1)
        las = [la,la,la+1,la+1]; los = [lo,lo+1,lo+1,lo]
    elif ((data['XLAT'][0][la,lo] > lat) & (data['XLONG'][0][la,lo] < lon)):
        #print ('><', la, lo, la+1, lo-1)
        las = [la,la,la-1,la-1]; los = [lo,lo+1,lo+1,lo]
    elif ((data['XLAT'][0][la,lo] > lat) & (data['XLONG'][0][la,lo] > lon)):
        #print ('>>', la, lo, la-1, lo-1)
        las = [la,la,la-1,la-1]; los = [lo,lo-1,lo-1,lo]
    elif ((data['XLAT'][0][la,lo] < lat) & (data['XLONG'][0][la,lo] > lon)):
        #print ('<>', la, lo, la-1, lo+1)
        las = [la,la,la+1,la+1]; los = [lo,lo-1,lo-1,lo]
        
    #print ('coordinates of station', (lat,lon))
    distances = []
    for la,lo in zip(las,los):
        distances.append(geopy.distance.geodesic((lat,lon),(data['XLAT'][0][la,lo],data['XLONG'][0][la,lo])).m)
        
    return (las,los,distances)

# --------------------------------------------------------------

def InverseDistanceWeighted(lat, lon, wind, precip, var='T2', data=WRF):
    
    las,los,dis = (findClosestGridPoints(lat=lat, lon=lon, data=WRF_3D))
    
    if wind == True and precip == False:
        weighted_u = []
        weighted_v = []
        var4pts_u = []
        var4pts_v = []
        for i,j in zip(los,las):
            moddata = data['U10'].sel(south_north=j, west_east=i).values # adjusted to xarray instead of df
            var4pts_u.append(moddata)
            moddata = data['V10'].sel(south_north=j, west_east=i).values
            var4pts_v.append(moddata)
        weighted_u.append( (var4pts_u[0]/dis[0] + var4pts_u[1]/dis[1] + var4pts_u[2]/dis[2] + var4pts_u[3]/dis[3]) / (1/dis[0] + 1/dis[1] + 1/dis[2] + 1/dis[3]))
        weighted_v.append( (var4pts_v[0]/dis[0] + var4pts_v[1]/dis[1] + var4pts_v[2]/dis[2] + var4pts_v[3]/dis[3]) / (1/dis[0] + 1/dis[1] + 1/dis[2] + 1/dis[3]))
        weighted = [weighted_u, weighted_v]
        
    if var in ['T2', 'RAINNC', 'SNOWNC'] and wind == False: # var == 'T2'
        weighted = []
        var4pts = []
        for i,j in zip(los,las):
            moddata = data[var].sel(south_north=j, west_east=i).values # adjusted to xarray instead of df
            var4pts.append(moddata)
        weighted.append( (var4pts[0]/dis[0] + var4pts[1]/dis[1] + var4pts[2]/dis[2] + var4pts[3]/dis[3]) / (1/dis[0] + 1/dis[1] + 1/dis[2] + 1/dis[3]))
        
    if var == 'HGT' and wind == False:
        weighted = []
        var4pts = []
        for i,j in zip(los,las):
            moddata = data[var].sel(south_north=j, west_east=i).values # adjusted to xarray instead of df
            var4pts.append(moddata)
        weighted.append( (var4pts[0]/dis[0] + var4pts[1]/dis[1] + var4pts[2]/dis[2] + var4pts[3]/dis[3]) / (1/dis[0] + 1/dis[1] + 1/dis[2] + 1/dis[3]))
        
    return weighted