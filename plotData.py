''' 
calculate accumulated precipitation        
'''

def calculateAccPrecip(dic = WRF_precip_exp1):

    AccPrecip = np.zeros_like(dic[list(dic.keys())[0]])
    
    # Iterate through the dictionary
    for key, data in dic.items():
        
        AccPrecip += data

    return AccPrecip 



'''
compare precipitation from model and observations
'''

title = f'Precipitation for {season} months ({years[0]}-{years[-1]+1} average for model/obs)'
title = ' '#f'Mean annual precipitation (model/observations)'
if season == 'SONDJFMAM':
    title = f'Precipitation for extended winter ({years[0]}-{years[-1]+1} average for model/obs)'

def plotPrecipAbs(precip_model = WRF_precip_exp1, snow_model = WRF_precip_var2_exp1, precip_obs = obs_precip, massbalance=massbalance, title=title, vmin=0, vmax=3000):

    if season == 'all':
        vmax = 4000
        
    
    precip_model_subset = {key: precip_model[key] for key in list(precip_model.keys())[-24:-12]} # subset for MG (only 2021, i.e., second last year)
    precip_model_subset = calculateAccPrecip(precip_model_subset)
    precip_model_subset2 = {key: precip_model[key] for key in list(precip_model.keys())[-84:-36]} # subset for JD (only 2016-2019)
    precip_model_subset2 = calculateAccPrecip(precip_model_subset2)
        
    precip_model = calculateAccPrecip(precip_model)
    #print (precip_model)
    scale=1.#.8
#    fig, ax = plt.subplots(figsize=(scale*12.5,scale*10))#15,10))
    fig, ax = plt.subplots(figsize=(scale*15,scale*8.5))
    plt.rcParams.update({'font.size': 16}) 

    # plot model data

    cm = ax.pcolormesh(WRF_lon, WRF_lat, precip_model/len(years),
                  vmin=vmin, vmax=vmax,  
                  cmap='viridis')

    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, levels=levels2, cmap='terrain')
    #if exp2 == 'noice_BT':
    #    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp2, linestyles='--', cmap='terrain')
#    ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000) # Hodlekve

    # add dots where there is ice or snow
    
    indices = np.argwhere(WRF_lu_exp1.values == 24)
    for (i, j) in indices:
        ax.scatter(WRF_lon[i,j], WRF_lat[i,j], marker='|', color='blue', s=13)


    # plot MET observations

    for s, st in enumerate(['MG', 'JD', 'FL', 'VS', 'OD', 'SJ']):
        if st not in [' '] and ~np.isnan(precip_obs[s]):
            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'].values[0])
            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'].values[0])
            if precip_obs[s] > 0:
                grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'].values[0])
                grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'].values[0])
                wrf_precip = precip_model[grid_j, grid_i]
                if st == 'MG':
                    ax.scatter(lon, lat, c=(precip_obs[s]+34.4)/1, ec='k', lw=.6, s=2*50, # +34.4 due to missing data for Aug 2021 in monthly sum, but existing as a minimum in daily data
                               vmin=vmin, vmax=vmax,  
                               cmap='viridis', zorder=1000)
                    ax.text(WRF_lon[grid_j, grid_i]+.05, WRF_lat[grid_j, grid_i]+.01, 
                            f'{st}: {precip_model_subset[grid_j, grid_i]/1:.0f} / {(precip_obs[s]+34.4)/1:.0f}', 
                            fontsize=14, c='dimgrey', va='center',
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')) 
                elif st == 'JD':
                    ax.scatter(lon, lat, c=(precip_obs[s])/4, ec='k', lw=.6, s=2*50, # +34.4 due to missing data for Aug 2021 in monthly sum, but existing as a minimum in daily data
                               vmin=vmin, vmax=vmax,  
                               cmap='viridis', zorder=1000)
                    ax.text(WRF_lon[grid_j, grid_i]+.05, WRF_lat[grid_j, grid_i]-.019, 
                            f'{st}: {precip_model_subset2[grid_j, grid_i]/4:.0f} / {(precip_obs[s])/4:.0f}', 
                            fontsize=14, c='dimgrey', va='center',
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')) 
                else:
                    ax.scatter(lon, lat, c=precip_obs[s]/len(years), ec='k', lw=.6, s=2*50,
                               vmin=vmin, vmax=vmax,  
                               cmap='viridis', zorder=1000)
                    if st == 'OD' or st == 'SJ':
                        ax.text(WRF_lon[grid_j, grid_i]-.06, WRF_lat[grid_j, grid_i]+.01, 
                                f'{st}: {wrf_precip/len(years):.0f} / {precip_obs[s]/len(years):.0f}', 
                                fontsize=14, va='center', ha='right',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))            
                    else:
                        ax.text(WRF_lon[grid_j, grid_i]+.05, WRF_lat[grid_j, grid_i]-.01, 
                                f'{st}: {wrf_precip/len(years):.0f} / {precip_obs[s]/len(years):.0f}', 
                                fontsize=14, va='center',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))            

    # plot mass balance

    if season == 'all':#SONDJFMAM':
        snow_model = calculateAccPrecip({key: snow_model[key] for key in list(WRF_precip_var2_exp1.keys())[8:-7]}) # use full extended winter seasons only
        for s in range(6,8):
            if s == 6: # Austdalsbreen    
                lon = massbalance.loc[(massbalance['year']==years[0])&(massbalance['stake_no'].str.startswith('A')),'lon']
                lat = massbalance.loc[(massbalance['year']==years[0])&(massbalance['stake_no'].str.startswith('A')),'lat']
                grid_i = 93; grid_j = 71 

            elif s == 7: # Nigardsbreen
                lon = massbalance.loc[(massbalance['year']==years[0])&(massbalance['stake_no'].str.startswith('N')),'lon']
                lat = massbalance.loc[(massbalance['year']==years[0])&(massbalance['stake_no'].str.startswith('N')),'lat']
                grid_i = 79; grid_j = 59

            ax.scatter(lon, lat, c='white', ec='k', lw=.6, marker='*', s=6*50, #precip_obs[s]/len(years)
                               vmin=vmin, vmax=vmax,  
                               cmap='viridis', zorder=1000)

            wrf_snow = ((snow_model)[grid_j, grid_i])
            ax.text(WRF_lon[grid_j, grid_i]+.06, WRF_lat[grid_j, grid_i]+.01, 
                    f'{wrf_snow/(len(years)-1):.0f} / {precip_obs[s]/(len(years)-1):.0f}', # len(years)-1 because we are only using full extended winter seasons
                    fontsize=14, va='center',
                    bbox=dict(facecolor='w', alpha=0.7, edgecolor='none'))
        ax.scatter((),(), c='grey', ec='k', s=2*50, label='rain+snow')
        ax.scatter((),(), c='w', ec='k', s=6*50, marker='*', label='snow')
        
        ax.legend(loc=4)

    # add distance scale
    startlat = 61.8974; startlon = 5.9761; endlat = 61.8974; endlon = 6.1671
    dlo = -.16; dla = -.535; dlon = endlon-startlon; dlat = 0.017
    
    rect = patches.Rectangle((startlon+dlo-0.01, startlat+dla-dlat/4), dlon+.02, 1.5*dlat, linewidth=1, edgecolor='w', facecolor='w', alpha=.8, zorder=1000)
    ax.add_patch(rect)
    ax.plot([startlon+dlo,endlon+dlo],[startlat+dla,endlat+dla],'k', zorder=1000)
    ax.plot([startlon+dlo,startlon+dlo],[startlat+dla,startlat+dla+dlat],'k', zorder=1000)
    ax.plot([endlon+dlo,endlon+dlo],[endlat+dla,endlat+dla+dlat],'k', zorder=1000)
    ax.text(startlon+dlo, startlat+dla+dlat/4, '  10 km', fontsize=13.5, zorder=1000)


    # plot configurations

    cb = plt.colorbar(cm, extend='max', pad=.02)
    #cbar_ax = fig.add_axes([0.135, -.05, .75, .03])  # [left, bottom, width, height]
    #cb = plt.colorbar(cm, extend='max', cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=13)
    cb.set_label('Mean annual precipitation (mm)', fontsize=15, rotation=270, labelpad=28)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')
    ax.set_title(f'{title}')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin+.07, ymax-.07)

    ax.set_yticks(np.arange(61.4,61.9,.2))

    plt.savefig('figures/precip.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()   
    
    
'''
plot difference in precipitation between two experiments
'''

def plotPrecipDiff(precip_model_exp1 = WRF_precip_exp1, precip_model_exp2 = WRF_precip_exp2, 
                   title=title, vmin=-20, vmax=20):

    plt.rcParams.update({'font.size': 22})
    
    title = f'{exp2lab} - {exp1lab}'
    
    precip_model_exp1 = calculateAccPrecip(precip_model_exp1)
    precip_model_exp2 = calculateAccPrecip(precip_model_exp2)
    
    # Define the discrete levels and use coolwarm_r colormap
    levels = np.arange(vmin, vmax+1, 1)
    cmap = plt.get_cmap("coolwarm_r")

    # Create a normalization using BoundaryNorm
    norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)


    condition_hgt = WRF_hgt_exp1.values <= 1000
    condition_lu = WRF_lu_exp1.values == 24
    
    combined_condition = condition_hgt & condition_lu

    print (np.mean(((precip_model_exp2-precip_model_exp1)/precip_model_exp1*100)[combined_condition]))

    
    fig, ax = plt.subplots(figsize=(15,10))

    # plot model data

    cm = ax.pcolormesh(WRF_lon, WRF_lat, (precip_model_exp2-precip_model_exp1)/precip_model_exp1*100,#len(years),
                       #vmin=vmin, vmax=vmax,  
                       cmap=cmap, norm=norm)

    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, levels=levels2, cmap='terrain')
    #if exp2 == 'noice_BT':
    #    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp2, linestyles='--', cmap='terrain')
#    ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000) # Hodlekve

    # add dots where there is ice or snow
    
    if exp1 == 'glac2019':
        indices = np.argwhere((WRF_lu_exp1.values == 24) & (WRF_lu_exp2.values != 24))
        mc = 'k'
        for (i, j) in indices:
            ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
    if exp1 == 'noice_BT' and exp2 == 'modlakes_noice_BT':
        indices = np.argwhere((WRF_lu_exp1.values != 16) & (WRF_lu_exp2.values == 16))
        mc = 'k'

        for (i, j) in indices:
            #ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, marker='.', s=22.2)
            #ax.pcolormesh(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
            
            # Create a rectangle patch for the selected cell
            rect = patches.Rectangle(
                ((WRF_lon[i,j].values + WRF_lon[i,j-1].values)/2, (WRF_lat[i,j].values+WRF_lat[i-1,j].values)/2),  # Bottom left corner
                WRF_lon[i,j].values-WRF_lon[i,j-1].values,  # Width of the rectangle
                WRF_lat[i,j].values-WRF_lat[i-1,j].values,  # Height of the rectangle
                linewidth=.8,  # Thickness of the frame
                edgecolor=mc,  # Color of the frame
                facecolor='none'  # No fill color
            )
            ax.add_patch(rect)
        

    if exp1 == 'glac2019' and exp2 == 'glac2100_dem2100':
        # add distance scale
        startlat = 61.8974; startlon = 6.0761; endlat = 61.8974; endlon = 6.3626#6.4581#6.2671
        dlo = -.16; dla = -.58; dlon = endlon-startlon; dlat = 0.022#17
        
        rect = patches.Rectangle((startlon+dlo-0.01, startlat+dla-dlat/4), dlon+.02, 1.5*dlat, linewidth=1, edgecolor='w', facecolor='w', alpha=.8, zorder=1000)
        ax.add_patch(rect)
        ax.plot([startlon+dlo,endlon+dlo],[startlat+dla,endlat+dla],'k', zorder=1000)
        ax.plot([startlon+dlo,startlon+dlo],[startlat+dla,startlat+dla+dlat],'k', zorder=1000)
        ax.plot([endlon+dlo,endlon+dlo],[endlat+dla,endlat+dla+dlat],'k', zorder=1000)
        ax.text(startlon+dlo, startlat+dla+dlat/4, '  15 km', fontsize=18.5, zorder=1000)

    cb = plt.colorbar(cm, extend='both', ticks=levels[1::3])
    cb.set_label('Relative difference in precipitation (%)', rotation=270, labelpad=30)
    #cb.ax.tick_params(labelsize=14)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    ax.set_yticks(np.arange(61.4,61.9,.2))

    ax.set_title(title)
    if season == 'all':
        plt.savefig(f'figures/precipdiff_{exp2}-{exp1}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    
'''
plot snow
'''

title = f'Snow for {season} months ({years[0]}-{years[-1]+1} average for model/obs)'
title = ' '#f'Mean annual precipitation (model/observations)'
if season == 'SONDJFMAM':
    title = f'Precipitation for extended winter ({years[0]}-{years[-1]+1} average for model/obs)'

def plotSnowAbs(snow_model = WRF_precip_var2_exp1, precip_obs = obs_precip, massbalance=massbalance, title=title, vmin=0, vmax=3000):

    plt.rcParams.update({'font.size': 26}) 

    if season == 'all':
        vmax = 3000
        
    precip_model = calculateAccPrecip(snow_model)
    scale=1.#.8
    
    fig, ax = plt.subplots(figsize=(scale*15,scale*10))#15, 8.5))

    # plot model data

    cm = ax.pcolormesh(WRF_lon, WRF_lat, precip_model/len(years),
                  vmin=vmin, vmax=vmax,  
                  cmap='viridis')

    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, levels=levels2, cmap='terrain')
    #if exp2 == 'noice_BT':
    #    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp2, linestyles='--', cmap='terrain')
#    ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000) # Hodlekve

    # add dots where there is ice or snow
    
    indices = np.argwhere(WRF_lu_exp1.values == 24)
    for (i, j) in indices:
        ax.scatter(WRF_lon[i,j], WRF_lat[i,j], marker='|', color='blue', s=13)


    # plot MET observations

    # for s, st in enumerate(['MG', 'JD', 'FL', 'VS', 'OD', 'SJ']):
    #     if st not in [' '] and ~np.isnan(precip_obs[s]):
    #         lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'].values[0])
    #         lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'].values[0])
    #         if precip_obs[s] > 0:
    #             grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'].values[0])
    #             grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'].values[0])
    #             wrf_precip = precip_model[grid_j, grid_i]
    #             if st == 'MG':
    #                 ax.scatter(lon, lat, c=(precip_obs[s]+34.4)/1, ec='k', lw=.6, s=2*50, # +34.4 due to missing data for Aug 2021 in monthly sum, but existing as a minimum in daily data
    #                            vmin=vmin, vmax=vmax,  
    #                            cmap='viridis', zorder=1000)
    #                 ax.text(WRF_lon[grid_j, grid_i]+.05, WRF_lat[grid_j, grid_i]+.01, 
    #                         f'{st}: {precip_model_subset[grid_j, grid_i]/1:.0f} / {(precip_obs[s]+34.4)/1:.0f}', 
    #                         fontsize=14, c='dimgrey', va='center',
    #                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')) 
    #             elif st == 'JD':
    #                 ax.scatter(lon, lat, c=(precip_obs[s])/4, ec='k', lw=.6, s=2*50, # +34.4 due to missing data for Aug 2021 in monthly sum, but existing as a minimum in daily data
    #                            vmin=vmin, vmax=vmax,  
    #                            cmap='viridis', zorder=1000)
    #                 ax.text(WRF_lon[grid_j, grid_i]+.05, WRF_lat[grid_j, grid_i]-.019, 
    #                         f'{st}: {precip_model_subset2[grid_j, grid_i]/4:.0f} / {(precip_obs[s])/4:.0f}', 
    #                         fontsize=14, c='dimgrey', va='center',
    #                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')) 
    #             else:
    #                 ax.scatter(lon, lat, c=precip_obs[s]/len(years), ec='k', lw=.6, s=2*50,
    #                            vmin=vmin, vmax=vmax,  
    #                            cmap='viridis', zorder=1000)
    #                 if st == 'OD' or st == 'SJ':
    #                     ax.text(WRF_lon[grid_j, grid_i]-.06, WRF_lat[grid_j, grid_i]+.01, 
    #                             f'{st}: {wrf_precip/len(years):.0f} / {precip_obs[s]/len(years):.0f}', 
    #                             fontsize=14, va='center', ha='right',
    #                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))            
    #                 else:
    #                     ax.text(WRF_lon[grid_j, grid_i]+.05, WRF_lat[grid_j, grid_i]-.01, 
    #                             f'{st}: {wrf_precip/len(years):.0f} / {precip_obs[s]/len(years):.0f}', 
    #                             fontsize=14, va='center',
    #                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))            

    # plot mass balance

    if season == 'all':#SONDJFMAM':
        snow_model = calculateAccPrecip({key: snow_model[key] for key in list(WRF_precip_var2_exp1.keys())[8:-7]}) # use full extended winter seasons only
        for s in range(6,8):
            if s == 6: # Austdalsbreen    
                lon = massbalance.loc[(massbalance['year']==years[0])&(massbalance['stake_no'].str.startswith('A')),'lon']
                lat = massbalance.loc[(massbalance['year']==years[0])&(massbalance['stake_no'].str.startswith('A')),'lat']
                grid_i = 93; grid_j = 71 

            elif s == 7: # Nigardsbreen
                lon = massbalance.loc[(massbalance['year']==years[0])&(massbalance['stake_no'].str.startswith('N')),'lon']
                lat = massbalance.loc[(massbalance['year']==years[0])&(massbalance['stake_no'].str.startswith('N')),'lat']
                grid_i = 79; grid_j = 59
                
            wrf_snow = ((snow_model)[grid_j, grid_i])

            ax.scatter(lon, lat, c=wrf_snow/(len(years)-1), ec='k', lw=1.8*.6, marker='*', s=1.8*6*50, #precip_obs[s]/len(years)
                               vmin=vmin, vmax=vmax,  
                               cmap='viridis', zorder=1000)

            ax.text(WRF_lon[grid_j, grid_i]+.06, WRF_lat[grid_j, grid_i]+.01, 
                    f'{wrf_snow/(len(years)-1):.0f} / {precip_obs[s]/(len(years)-1):.0f}', # len(years)-1 because we are only using full extended winter seasons
                    fontsize=20, va='center',
                    bbox=dict(facecolor='w', alpha=0.7, edgecolor='none'))
        ax.scatter((),(), c='grey', ec='k', s=2*50, label='rain+snow')
        ax.scatter((),(), c='w', ec='k', s=6*50, marker='*', label='snow')
        
#        ax.legend(loc=4)

    # add distance scale
    startlat = 61.8974; startlon = 5.9761; endlat = 61.8974; endlon = 6.1671+0.0955
    dlo = -.16; dla = -.535; dlon = endlon-startlon; dlat = 0.017
    
    rect = patches.Rectangle((startlon+dlo-0.01, startlat+dla-dlat/4), dlon+.02, 1.5*dlat, linewidth=1, edgecolor='w', facecolor='w', alpha=.8, zorder=1000)
    ax.add_patch(rect)
    ax.plot([startlon+dlo,endlon+dlo],[startlat+dla,endlat+dla],'k', zorder=1000)
    ax.plot([startlon+dlo,startlon+dlo],[startlat+dla,startlat+dla+dlat],'k', zorder=1000)
    ax.plot([endlon+dlo,endlon+dlo],[endlat+dla,endlat+dla+dlat],'k', zorder=1000)
    ax.text(startlon+dlo, startlat+dla+dlat/4, '  15 km', fontsize=20, zorder=1000)

    # plot configurations

    cb = plt.colorbar(cm, extend='max', pad=.02)
    #cbar_ax = fig.add_axes([0.135, -.05, .75, .03])  # [left, bottom, width, height]
    #cb = plt.colorbar(cm, extend='max', cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=20)
    cb.set_label('Mean annual snow (mm)', fontsize=20, rotation=270, labelpad=28)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')
    ax.set_title(f'{title}')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin+.07, ymax-.07)

    ax.set_yticks(np.arange(61.4,61.9,.2))

    plt.savefig('figures/snow.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()   
    
    
'''
plot difference in snow between two experiments
'''

def plotSnowDiff(snow_model_exp1 = WRF_precip_var2_exp1, snow_model_exp2 = WRF_precip_var2_exp2, 
                   title=title, vmin=-20, vmax=20):
    
    title = f'Difference in snow ({exp2} - {exp1})\nfor {season} months ({years[0]}-{years[-1]+1} average)'
    
    snow_model_exp1 = calculateAccPrecip(snow_model_exp1)
    snow_model_exp2 = calculateAccPrecip(snow_model_exp2)
    
    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams.update({'font.size': 16}) 

    # plot model data

    cm = ax.pcolormesh(WRF_lon, WRF_lat, (snow_model_exp2-snow_model_exp1)/len(years), #/snow_model_exp1*100,#len(years),
                       vmin=vmin, vmax=vmax,  
                       cmap='coolwarm_r')

    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, levels=levels2, cmap='terrain')
    #if exp2 == 'noice_BT':
    #    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp2, linestyles='--', cmap='terrain')
    ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000) # Hodlekve
    ax.scatter(7.217269140081988, 61.33605680043416, c='k', marker='*', s=150, zorder=1000) # Heggis

    # add dots where there is ice or snow
    
    if exp1 == 'glac2019':
        indices = np.argwhere((WRF_lu_exp1.values == 24) & (WRF_lu_exp2.values != 24))
        mc = 'k'
        for (i, j) in indices:
            ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
    if exp1 == 'noice_BT' and exp2 == 'modlakes_noice_BT':
        indices = np.argwhere((WRF_lu_exp1.values != 16) & (WRF_lu_exp2.values == 16))
        mc = 'b'

        for (i, j) in indices:
            #ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
            #ax.pcolormesh(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
            
            # Create a rectangle patch for the selected cell
            rect = patches.Rectangle(
                ((WRF_lon[i,j].values + WRF_lon[i,j-1].values)/2, (WRF_lat[i,j].values+WRF_lat[i-1,j].values)/2),  # Bottom left corner
                WRF_lon[i,j].values-WRF_lon[i,j-1].values,  # Width of the rectangle
                WRF_lat[i,j].values-WRF_lat[i-1,j].values,  # Height of the rectangle
                linewidth=.5,  # Thickness of the frame
                edgecolor=mc,  # Color of the frame
                facecolor='none'  # No fill color
            )
            ax.add_patch(rect)


    cb = plt.colorbar(cm)
#    cb.set_label('Relative difference in snow (%)', rotation=270, labelpad=30)
    cb.set_label('Difference in annual snow (mm)', rotation=270, labelpad=30)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_title(title)
    plt.show()
 
   
'''
plot the difference between the difference in snow and the difference in precipitation between two experiments
'''

def plotRainDiff(precip_model_exp1 = WRF_precip_exp1, precip_model_exp2 = WRF_precip_exp2,
                 snow_model_exp1 = WRF_precip_var2_exp1, snow_model_exp2 = WRF_precip_var2_exp2,
                 title=title, vmin=-20, vmax=20):
    
    title = f'Difference in rain ({exp2} - {exp1})\nfor {season} months ({years[0]}-{years[-1]+1} average)'
    
    precip_model_exp1 = calculateAccPrecip(precip_model_exp1)
    precip_model_exp2 = calculateAccPrecip(precip_model_exp2)
    snow_model_exp1 = calculateAccPrecip(snow_model_exp1)
    snow_model_exp2 = calculateAccPrecip(snow_model_exp2)
    
    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams.update({'font.size': 16}) 

    # plot model data
    
    # print ((precip_model_exp2-precip_model_exp1).max())
    # print ((snow_model_exp2-snow_model_exp1).max())
    # print ((precip_model_exp2-precip_model_exp1).min())
    # print ((snow_model_exp2-snow_model_exp1).min())

    cm = ax.pcolormesh(WRF_lon, WRF_lat, 
                       ((precip_model_exp2-precip_model_exp1)-(snow_model_exp2-snow_model_exp1))/len(years), #/snow_model_exp1*100,
                       vmin=vmin, vmax=vmax,  
                       cmap='coolwarm_r')

    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, levels=levels2, cmap='terrain')
    #if exp2 == 'noice_BT':
    #    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp2, linestyles='--', cmap='terrain')
#    ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000) # Hodlekve

    # add dots where there is ice or snow
    
    if exp1 == 'glac2019':
        indices = np.argwhere((WRF_lu_exp1.values == 24) & (WRF_lu_exp2.values != 24))
        mc = 'k'
        for (i, j) in indices:
            ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
    if exp1 == 'noice_BT' and exp2 == 'modlakes_noice_BT':
        indices = np.argwhere((WRF_lu_exp1.values != 16) & (WRF_lu_exp2.values == 16))
        mc = 'b'

        for (i, j) in indices:
            #ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
            #ax.pcolormesh(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
            
            # Create a rectangle patch for the selected cell
            rect = patches.Rectangle(
                ((WRF_lon[i,j].values + WRF_lon[i,j-1].values)/2, (WRF_lat[i,j].values+WRF_lat[i-1,j].values)/2),  # Bottom left corner
                WRF_lon[i,j].values-WRF_lon[i,j-1].values,  # Width of the rectangle
                WRF_lat[i,j].values-WRF_lat[i-1,j].values,  # Height of the rectangle
                linewidth=.5,  # Thickness of the frame
                edgecolor=mc,  # Color of the frame
                facecolor='none'  # No fill color
            )
            ax.add_patch(rect)
        

    cb = plt.colorbar(cm)
    cb.set_label('Difference in annual rain (mm)', rotation=270, labelpad=30)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_title(title)
    plt.show()
    
''' 
calculate mean from monthly mean        
'''

def calculateMeanTemp(dic = WRF_temp_exp1):

    # Days in months, accounting for leap years
    days_in_months = {
        '01': 31,
        '02': 28,  # default; we'll adjust for leap years later
        '03': 31,
        '04': 30,
        '05': 31,
        '06': 30,
        '07': 31,
        '08': 31,
        '09': 30,
        '10': 31,
        '11': 30,
        '12': 31,
    }

    # Dictionary to adjust February for leap years
    leap_years = {year: 29 for year in range(2000, 2030) if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)}

    # Initialize variables to accumulate weighted sums and total days
    weighted_sum = None
    total_days = 0

    # Iterate through the dictionary
    for key, data in dic.items():
        year, month = key.split('_')

        # Determine the number of days for this month
        num_days = days_in_months[month]

        # If it's February and a leap year, override the days
        if month == '02' and int(year) in leap_years:
            num_days = leap_years[int(year)]

        # Calculate the weighted data
        weighted_data = data * num_days

        # Accumulate the weighted sum
        if weighted_sum is None:
            weighted_sum = weighted_data
            total_days = np.where(np.isnan(weighted_data), np.nan, num_days)
            #print ('assigning weighted data')#, weighted_sum, total_days)
        else:
            if month == '01': # start adding new data only at the start of the year
                weighted_sum = (np.nan_to_num(weighted_sum) + np.nan_to_num(weighted_data))
                total_days = (np.nan_to_num(total_days) + np.nan_to_num(num_days))
            else:
                weighted_sum += weighted_data # this does not add new data from stations on months after January
                total_days += num_days
                        
    # Calculate the average across all months
    if np.nansum(total_days) > 0:
        average_temp = weighted_sum / total_days
        #print (weighted_sum, total_days, average_temp)
    else:
        average_temp = None  # Or handle the case where total_days is 0

    return average_temp
 
    
'''
compare temperature from model and observations
'''

title = f'Mean temperature for {season} months ({years[0]}-{years[-1]} average)'

def plotTempAbs(temp_model = WRF_temp_exp1, temp_obs = obs_temp, 
                  title=title, vmin=-10, vmax=20):

    if season == 'all':
        vmin = -8; vmax = 8
    elif season == 'JJA':
        vmin = 0; vmax = 15
    #elif season == 'DJF':
    #    vmin = -10; vmax = 20
        
    temp_model = calculateMeanTemp(temp_model)-273.15
    
    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams.update({'font.size': 16}) 

    # plot model data

    cm = ax.pcolormesh(WRF_lon, WRF_lat, temp_model,
                  vmin=vmin, vmax=vmax,  
                  cmap='coolwarm')

    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, cmap='terrain')
    #if exp2 == 'noice_BT':
    #    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp2, linestyles='--', cmap='terrain')
#    ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000) # Hodlekve

    # add dots where there is ice or snow
    
    indices = np.argwhere(WRF_lu_exp1.values == 24)
    for (i, j) in indices:
        ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color='k', s=2)


    # plot MET observations

    temp_obs = calculateMeanTemp(temp_obs)
    
    for s, st in enumerate(stations):#['MG', 'FL', 'VS', 'OD', 'SJ']):
        if ~np.isnan(temp_obs[s]) and st in ['FL', 'OV', 'LV']: #st not in [' ']:
            print (st, WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'].values[0])
            alt_corr = 5*10**(-3)*(WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_hgt']-WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_hgt']).values[0]
#            print (st, alt_corr)
            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'].values[0])
            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'].values[0])
            if temp_obs[s] > 0:
                ax.scatter(lon, lat, c=temp_obs[s], ec='k', lw=.6, s=50,
                           vmin=vmin, vmax=vmax,
                           cmap='coolwarm', zorder=1000)
                grid_i = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_i'].values[0])
                grid_j = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'grid_j'].values[0])
                wrf_temp = ((temp_model)[grid_j, grid_i])
                #print (s, wrf_temp, alt_corr)
                ax.text(WRF_lon[grid_j, grid_i], WRF_lat[grid_j, grid_i]-.01, 
                        f'  {wrf_temp+alt_corr:.1f} / {temp_obs[s]:.1f}', 
                        fontsize=14, va='center',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))            


    # plot configurations

    cb = plt.colorbar(cm)
    cb.set_label('Mean temperature ($\u00b0$C)', rotation=270, labelpad=30)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')
    ax.set_title(f'{title}')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.show()
    
    
    
'''
plot difference in temperature between two experiments
'''

def plotTempDiff(temp_model_exp1 = WRF_temp_exp1, temp_model_exp2 = WRF_temp_exp2, 
                   title=title, vmin=-20, vmax=20):
    
    plt.rcParams.update({'font.size': 22})
    
    title = f'Difference in average temperature ({exp2} - {exp1})\nfor {season} months ({years[0]}-{years[-1]+1} average)'
    title = f'{exp2lab} - {exp1lab}'#' ({years[0]}-{years[-1]} average)'
    
    temp_model_exp1 = calculateMeanTemp(temp_model_exp1)
    temp_model_exp2 = calculateMeanTemp(temp_model_exp2)
    
    # Define the discrete levels and use coolwarm_r colormap
    levels = np.arange(vmin, vmax+.1, .1)
    cmap = plt.get_cmap("coolwarm")

    # Create a normalization using BoundaryNorm
    norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)
    
    #if exp2 == 'noice_BT' and season == 'all':
    #    vmin = -30; vmax = 30
    
    fig, ax = plt.subplots(figsize=(15,10))

    # plot model data

    cm = ax.pcolormesh(WRF_lon, WRF_lat, (temp_model_exp2-temp_model_exp1),
                       #vmin=vmin, vmax=vmax,  
                       cmap=cmap, norm=norm)#'coolwarm')

    cm2 = ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, levels=levels2, cmap='terrain')#colors='darkgrey')#
    cm3 = ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1*np.nan, levels=levels2, linewidths=5, cmap='terrain')#colors='darkgrey')#

    #if exp2 == 'noice_BT':
    #    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp2, linestyles='--', cmap='terrain')
#    ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000) # Hodlekve

    # add dots where there is ice or snow
    
    if exp1 == 'glac2019':
        indices = np.argwhere((WRF_lu_exp1.values == 24) & (WRF_lu_exp2.values != 24))
        mc = 'k'
        for (i, j) in indices:
            ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
    if exp1 == 'noice_BT' and exp2 == 'modlakes_noice_BT':
        indices = np.argwhere((WRF_lu_exp1.values != 16) & (WRF_lu_exp2.values == 16))
        mc = 'k'
        
        for (i, j) in indices:
            #ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, marker='.', s=22.2)
            #ax.pcolormesh(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
            
            # Create a rectangle patch for the selected cell
            rect = patches.Rectangle(
                ((WRF_lon[i,j].values + WRF_lon[i,j-1].values)/2, (WRF_lat[i,j].values+WRF_lat[i-1,j].values)/2),  # Bottom left corner
                WRF_lon[i,j].values-WRF_lon[i,j-1].values,  # Width of the rectangle
                WRF_lat[i,j].values-WRF_lat[i-1,j].values,  # Height of the rectangle
                linewidth=.8,  # Thickness of the frame
                edgecolor=mc,  # Color of the frame
                facecolor='none' # No fill color
            )
            ax.add_patch(rect)
        
    ice_mask = ((WRF_lu_exp1.values == 24))

    if exp1 == 'glac2019' and exp2 == 'glac2100_dem2100':
        # add distance scale
        startlat = 61.8974; startlon = 6.0761; endlat = 61.8974; endlon = 6.3626#6.4581#6.2671
        dlo = -.16; dla = -.58; dlon = endlon-startlon; dlat = 0.022#17
        
        rect = patches.Rectangle((startlon+dlo-0.01, startlat+dla-dlat/4), dlon+.02, 1.5*dlat, linewidth=1, edgecolor='w', facecolor='w', alpha=.8, zorder=1000)
        ax.add_patch(rect)
        ax.plot([startlon+dlo,endlon+dlo],[startlat+dla,endlat+dla],'k', zorder=1000)
        ax.plot([startlon+dlo,startlon+dlo],[startlat+dla,startlat+dla+dlat],'k', zorder=1000)
        ax.plot([endlon+dlo,endlon+dlo],[endlat+dla,endlat+dla+dlat],'k', zorder=1000)
        ax.text(startlon+dlo, startlat+dla+dlat/4, '  15 km', fontsize=18.5, zorder=1000)

    cb = plt.colorbar(cm, extend='both', ticks=levels[1::3])
    cb.set_label('Difference in annual temperature (K)', rotation=270, labelpad=30)


    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_yticks(np.arange(61.4,61.9,.2))

    ax.set_title(title)
    if season == 'all':
        plt.savefig(f'figures/tempdiff_{exp2}-{exp1}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    
'''
plot difference in landuse between two experiments
'''

def createIceClasses(exp1, exp2):
    classes = np.empty(exp1.shape, dtype=int)

    classes[(exp1.values != 24)] = 0
    classes[(exp1.values == 24) & (exp2.values != 24)] = 1
    classes[(exp2.values == 24)] = 2

    return classes

def create_colormap():
    from matplotlib.colors import ListedColormap
    colors = ['tan', 'turquoise', 'white'] #(0, 1, 0, 0) transparent
    return ListedColormap(colors)


def plotLUDiff(title=title, vmin=-20, vmax=20):
    
    title = f'Difference in landuse'
    
    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams.update({'font.size': 16}) 

    classes = createIceClasses(WRF_lu_exp1, WRF_lu_exp2)
    cmap = create_colormap()    
    
    # plot model data

    #ax.contourf(WRF_lon, WRF_lat, WRF_hgt_exp1, cmap='terrain')

    cm = ax.pcolormesh(WRF_lon, WRF_lat, classes,
                       #vmin=vmin, vmax=vmax,  
                       cmap=cmap, shading='auto')  # Use shading='auto' for better visuals
    
    #ax.scatter((), (), marker='s', color='snow', label='no ice')
    ax.scatter((), (), marker='s', s=100, color='turquoise', ec='k', label='ice in 2019')
    ax.scatter((), (), marker='s', s=100, color='white', ec='k', label='ice in 2100')
    
    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, levels=levels2, cmap='terrain')
#    ax.scatter(6.971929730189916, 61.28864257057141, c='k', marker='*', s=150, zorder=1000) # Hodlekve
#    ax.scatter(7.198, 61.686, c='k', marker='*', s=150, zorder=1000) # NB

    for lon, lat in zip(WRF1000_ts['station_lon'], WRF1000_ts['station_lat']):
        ax.scatter(lon, lat, marker='o', c='k', zorder=100)

    ax.legend()

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    #ax.set_title(title)
    plt.show()

def plotLUDiff3D(z_scale_factor=2, elevation_angle=40, azimuth_angle=-105, vmin=-300, vmax=300):
    
    fig = plt.figure(figsize = (15*1.3, 10*1.3))
    ax = fig.add_subplot(111, projection='3d')
    
    sind = 16
    eind = -10
    
    classes = createIceClasses(WRF_lu_exp1[sind:eind, sind:eind], WRF_lu_exp2[sind:eind, sind:eind])
    cmap = create_colormap() 

    # Calculate the difference in elevation
    elevation_diff = WRF_hgt_exp1 - WRF_hgt_exp2

    from matplotlib.colors import LightSource

    # Create a LightSource object
    ls = LightSource(azdeg=azimuth_angle, altdeg=elevation_angle)

    # Shade the surface, adjusting the intensity to reduce shading effect
    shaded_surface = ls.shade(classes, cmap=cmap, vert_exag=0.1, blend_mode='soft')#soft')
    
    # Plot the terrain surface colored by the elevation difference
    terrain_surface = ax.plot_surface(WRF_lon[sind:eind, sind:eind], WRF_lat[sind:eind, sind:eind], 
                                      WRF_hgt_exp1[sind:eind, sind:eind], 
                                      facecolors=shaded_surface, #cmap(classes),
                                      edgecolor='none')#, shade=False)#, alpha=0.5)

    ax.scatter((), (), marker='s', s=100, color='turquoise', ec='k', label='ice in 2019')
    ax.scatter((), (), marker='s', s=100, color='white', ec='k', label='ice in 2100')
    ax.legend(loc='upper left', bbox_to_anchor=(0.08, 0.9))

    ax.set_xlabel('Longitude ($\u00b0$)', labelpad=10)
    ax.set_ylabel('Latitude ($\u00b0$)', labelpad=25)
    ax.set_zlabel('               Elevation (m)', labelpad=25)

    # Increase tick label padding
    ax.tick_params(axis='x', which='major', pad=0)  # Increase tick label padding for x-axis
    ax.tick_params(axis='y', which='major', pad=10)  # Increase tick label padding for y-axis
    ax.tick_params(axis='z', which='major', pad=12)  # Increase tick label padding for z-axis

    ax.set_xticks(np.arange(6,8.1,.5))
    ax.set_yticks(np.arange(61.3,62,.2))
    ax.set_zticks(np.arange(0,2100,1000))
    
    # Set the z-axis limits to squeeze the height
    z_min = 0 # np.min(WRF_hgt_exp1) * z_scale_factor
    z_max = 2500 # np.max(WRF_hgt_exp1) # * z_scale_factor
    ax.set_zlim(z_min, z_max)

    ax.set_xlim(xmin+.2, xmax-.2)
    ax.set_ylim(ymin+.1, ymax-.05)

    # Set the aspect ratio, compressing the z-axis visually
    ax.set_box_aspect([1, 1, 0.4])  # Keep x and y aspect ratio 1, compress z-axis to 0.3    
    
    # Set the viewing angle
    ax.view_init(elev=elevation_angle, azim=azimuth_angle)
    
    #ax.grid(False)
    
    plt.show()    

    
def plotHGTDiff(vmin=-330, vmax=330):
        
    scale=1.
    vmin = vmin; vmax = vmax
    levels = np.arange(vmin, vmax+1, 60)

    cmap = plt.get_cmap("coolwarm_r")

    # Create a normalization using BoundaryNorm
    norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(scale*15,scale*8.5))
    plt.rcParams.update({'font.size': 16}) 

    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, levels=levels2, cmap='terrain')
    cm = ax.pcolormesh(WRF_lon, WRF_lat, WRF_hgt_exp2-WRF_hgt_exp1, cmap='seismic', norm=norm)#, vmin=vmin, vmax=vmax)

    cb = plt.colorbar(cm, extend='both', ticks=levels[1::3])
    cb.set_label('Difference in elevation (m)', rotation=270, labelpad=30)
    #cb.ax.tick_params(labelsize=14)

    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_yticks(np.arange(61.4,61.9,.2))

    ax.set_title('no-ice-volume vs. control')
    
def plotHGTDiff3D(z_scale_factor=2, elevation_angle=40, azimuth_angle=-105, vmin=-300, vmax=300):
    
    fig = plt.figure(figsize = (15*1.3, 10*1.3))
    ax = fig.add_subplot(111, projection='3d')
    
    sind = 16
    eind = -10
    
    # Calculate the difference in elevation
    elevation_diff = WRF_hgt_exp1[sind:eind, sind:eind] - WRF_hgt_exp2[sind:eind, sind:eind]
    print ('MAX ELEVATION DIFF: ', np.max(elevation_diff.values))
    
    # Plot the terrain surface colored by the elevation difference
    terrain_surface = ax.plot_surface(WRF_lon[sind:eind, sind:eind], WRF_lat[sind:eind, sind:eind], 
                                      WRF_hgt_exp1[sind:eind, sind:eind], 
                                      facecolors=plt.cm.coolwarm((elevation_diff - vmin) / (vmax - vmin)),
                                      edgecolor='none')

    # Adding color bar for the elevation differences
    cbar_diff = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=ax, pad=0.01, aspect=18, shrink=.7)
    cbar_diff.set_label('Difference in elevation (m)', rotation=270, labelpad=20)

    # Set the range for the color bar
    cbar_diff.set_ticks(np.arange(.5-1/12, 1.01, 1/12))  # Set ticks based on the elevation difference
    cbar_diff.set_ticklabels(np.arange(vmin/6,vmax+1,50).astype(int))
    cbar_diff.ax.set_ylim(.5-1/12,1)

    ax.set_xlabel('Longitude ($\u00b0$)', labelpad=10)
    ax.set_ylabel('Latitude ($\u00b0$)', labelpad=25)
    ax.set_zlabel('               Elevation (m)', labelpad=25)

    # Increase tick label padding
    ax.tick_params(axis='x', which='major', pad=0)  # Increase tick label padding for x-axis
    ax.tick_params(axis='y', which='major', pad=10)  # Increase tick label padding for y-axis
    ax.tick_params(axis='z', which='major', pad=12)  # Increase tick label padding for z-axis

    ax.set_xticks(np.arange(6,8.1,.5))
    ax.set_yticks(np.arange(61.3,62,.2))
    ax.set_zticks(np.arange(0,2100,1000))
    
    # Set the z-axis limits to squeeze the height
    z_min = 0 # np.min(WRF_hgt_exp1) * z_scale_factor
    z_max = 2500 # np.max(WRF_hgt_exp1) # * z_scale_factor
    ax.set_zlim(z_min, z_max)

    ax.set_xlim(xmin+.2, xmax-.2)
    ax.set_ylim(ymin+.1, ymax-.05)

    # Set the aspect ratio, compressing the z-axis visually
    ax.set_box_aspect([1, 1, 0.4])  # Keep x and y aspect ratio 1, compress z-axis to 0.3    
    
    # Set the viewing angle
    ax.view_init(elev=elevation_angle, azim=azimuth_angle)
    
    #ax.grid(False)
    
    plt.show()
    

#c = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf', 'k', 'navy', 'aquamarine']
colors = [u'#1f77b4', u'#2ca02c', u'#ff7f0e', u'#7f7f7f', u'#bcbd22', 'navy', u'#17becf']
ordered_indices = [6, 3, 4, 5, 0, 1, 2]

def plotModelErrorTimeseries(experiments = [exp1, exp2]):

    plt.rcParams.update({'font.size': 18})
    
    # 10 days of missing data at AS in june 2012
    corr_1pt = np.nan*np.ones((len(t2m_monthly_obs.columns[1:])))
    corr_4pts = np.nan*np.ones((len(t2m_monthly_obs.columns[1:])))
    error_1pt = np.nan*np.ones((len(t2m_monthly_obs.columns[1:]), len(t2m_monthly_obs['date'])))
    error_4pts = np.nan*np.ones((len(t2m_monthly_obs.columns[1:]), len(t2m_monthly_obs['date'])))
    
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
            line, = ax1.plot(t2m_monthly_obs['date'], t2m_1pt_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15, c=colors[i], alpha=alp, ls='-', label=f'{loc}: {abserror:.1f} / {biaserror:.1f}')#_{pts}: {abserror:.1f}')
            handles.append(line)  # Store the handle
            names.append(f'{loc}: {abserror:.1f} / {biaserror:.1f}')
        
            if exp == 'glac2019':
                error_1pt[i] = t2m_1pt_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15
            
            # pts = '4pts'
            # if corrlabel == 'altitude adjusted':
            #     corr = corr_4pts[i]
            # abserror = abs(t2m_4pts_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15).mean()
            # ax1.plot(t2m_monthly_obs['date'], t2m_4pts_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15, c=colors[i], ls='--', label=f'{loc}_{pts}: {abserror:.1f}')
            # if exp == 'glac2019':
            #     error_4pts[i] = t2m_4pts_monthly[exp][loc]+corr-t2m_monthly_obs[loc]-273.15
    
    #ax1.plot((), (), c='w', label=' ')
    #ax1.plot((), (), c='grey', ls='-', label='1 pt')
    #ax1.plot((), (), c='grey', ls='--', label='4 pts')
    #ax1.legend(loc=3, ncols=3)
    ax1.legend([handles[i] for i in ordered_indices], [names[i] for i in ordered_indices], loc=3, ncols=3)
    #ax1.set_ylim(-7,3.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Error in mean monthly temperature (K)')
    ax1.set_title(f'Modelled minus observed temperature ({corrlabel})')
    
    ax1.grid()
    
    plt.savefig(f'figures/temp_error.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    return (names, error_1pt)

def plotModelMonthlyError():

    handles = []
    
    fig, ax = plt.subplots(figsize=(10,7))
    
    for i,loc in enumerate(t2m_monthly_obs.columns[1:]):
        for m in range(12):
            ax.bar(m+i/8, np.nanmean(error_1pt[i][m::12]), width=.125, color=colors[i])
        line, = ax.bar(m+i/8, np.nanmean(error_1pt[i][m::12]), width=.125, color=colors[i], label=loc)
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
    
    
    
def classifyWind():
    
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
        
    # def classify_wd_v2(wd):
    #     if (0 <= wd < 90):
    #         return 'NE'
    #     elif 90 <= wd < 180:
    #         return 'SE'
    #     elif 180 <= wd < 270:
    #         return 'SW'
    #     elif 270 <= wd <= 360:
    #         return 'NW'
    #     else:
    #         return 'Invalid'
        
    # def classify_wd_v3(wd):
    #     if 360*15/16 <= wd or wd < 360*1/16:
    #         return 'N'
    #     elif 360*1/16 <= wd < 360*3/16:
    #         return 'NE'
    #     elif 360*3/16 <= wd < 360*5/16:
    #         return 'E'
    #     elif 360*5/16 <= wd < 360*7/16:
    #         return 'SE'
    #     elif 360*7/16 <= wd < 360*9/16:
    #         return 'S'
    #     elif 360*9/16 <= wd < 360*11/16:
    #         return 'SW'
    #     elif 360*11/16 <= wd < 360*13/16:
    #         return 'W'
    #     elif 360*13/16 <= wd < 360*15/16:
    #         return 'NW'
    #     else:
    #         return 'Invalid'
        
    

    wind = wd_1pt[exp1][['date', 'PEAK']]
    wind.rename(columns={'PEAK': 'wd'}, inplace=True)
    wind = pd.merge(wind, ws_1pt[exp1][['date', 'PEAK']], on='date', how='inner')
    wind.rename(columns={'PEAK': 'ws'}, inplace=True)
    wind = pd.merge(wind, precip_1pt[exp1], on='date', how='inner')

    # apply wd classes
    wind['class'] = wind['wd'].apply(classify_wd)

    for loc in precip_1pt[exp1].columns[1:]:
        wind[loc][np.where(wind[loc]<wind[loc][0])[0][0]] = np.nan
        wind[loc] = wind[loc].diff()
        wind.loc[wind[loc]<0, loc] = np.nan # remove negative values (that arise due to restart?)
        wind.loc[wind[loc]>100, loc] = np.nan # remove unrealistically high values (that arise due to restart?)
        
    return (wind)


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



def wind_color(dominant_regime): 

    cmap=cmocean.cm.phase
    offset=.1 # should be less than 0.125 / 0.25
    
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


def plotAllYearsMonths():
    
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
            mask = (ws_1pt[exp1]['year'] == y) & (ws_1pt[exp1]['month'] == month)
            
            dominant_regime = estimate_wind_regime(wd_1pt[exp1][mask]['PEAK'])
            regimes[f'{y}_{month:02}'] = dominant_regime
    
            sm = wind_rose(ws_1pt[exp1][mask]['PEAK'], wd_1pt[exp1][mask]['PEAK'], ax[(y-years[0])*num_months+m], speed_bins, ec=wind_color(dominant_regime))
    
    for y, year in enumerate(years):
        ax[y*num_months].set_ylabel(f'{year}', fontsize=12, labelpad=20)
    
    for m, month_name in enumerate(months_names):
        ax[(len(years)-1)*num_months+m].set_xlabel(month_name, fontsize=12)
        
    plt.show()
    
    
def defineDailyRegimes():

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
                
    return (regimes_daily)


def defineMasks():

    if exp1 == 'glac2019':
        ice_mask = ((WRF_lu_exp1.values == 24))
        
        thr_ice = np.median(WRF_hgt_exp1.values[ice_mask])
        ice_low_mask = (WRF_hgt_exp1.values < thr_ice)
        ice_high_mask = (WRF_hgt_exp1.values >= thr_ice)
        
        thr_noice = np.median(WRF_hgt_exp1.values[~ice_mask])
        noice_low_mask = (WRF_hgt_exp1.values < thr_noice)
        noice_high_mask = (WRF_hgt_exp1.values >= thr_noice)
    
    return (ice_low_mask, ice_high_mask, noice_low_mask, noice_high_mask)


def plotTemperatureWindRegime():
    
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
                
            ax.scatter(x+os, y, color=wind_color(cl), s=10)
        
        
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
    #print (np.mean(y_values_N), np.mean(y_values_E), np.mean(y_values_S), np.mean(y_values_W))
    
    ax.axvline(x=np.mean(x_values_N)+0*xrange, c=wind_color('N'), ls='--')
    ax.axvline(x=np.mean(x_values_E)+1*xrange, c=wind_color('E'), ls='--')
    ax.axvline(x=np.mean(x_values_S)+2*xrange, c=wind_color('S'), ls='--')
    ax.axvline(x=np.mean(x_values_W)+3*xrange, c=wind_color('W'), ls='--')
    
    for cl in ['N', 'E', 'S', 'W']:
    #for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
        ax.scatter((), (), color=wind_color(cl), label=cl)
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
    
    return (x_values_N, x_values_E, x_values_S, x_values_W, y_values_N, y_values_E, y_values_S, y_values_W)


    
def plotPrecipitationWindRegime():
        
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
                    
                ax.scatter(x+os, y, color=wind_color(cl), s=10)
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
    #print (np.nanmean(y_values_N), np.nanmean(y_values_E), np.nanmean(y_values_S), np.nanmean(y_values_W))
    
    ax.axvline(x=np.nanmean(x_values_N)+0*xrange, c=wind_color('N'), ls='--')
    ax.axvline(x=np.nanmean(x_values_E)+1*xrange, c=wind_color('E'), ls='--')
    ax.axvline(x=np.nanmean(x_values_S)+2*xrange, c=wind_color('S'), ls='--')
    ax.axvline(x=np.nanmean(x_values_W)+3*xrange, c=wind_color('W'), ls='--')
    
    for cl in ['N', 'E', 'S', 'W']:
    #for cl in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
        ax.scatter((), (), s=50, color=wind_color(cl), label=cl)
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
    
    return (x_values_N, x_values_E, x_values_S, x_values_W, y_values_N, y_values_E, y_values_S, y_values_W)

    

def testStatistics():#y_values_N=y_values_N, y_values_E=y_values_E, y_values_S=y_values_S, y_values_W=y_values_W):
    
    # testing significance

    for x in [y_values_N, y_values_E, y_values_S, y_values_W]:

        #np.nanmean(y_values_N)
        x = np.array(x)
        x = x[~np.isnan(x)]
        
        n = len(x)
        xbar = x.mean()
        s = x.std(ddof=1)
        
        tstat, pval = stats.ttest_1samp(x, 0.0)
        
        print ("n, xbar, s, tstat, pval:  ", n, xbar, s, tstat, pval)


    # checking slope

    for x,y in zip([x_values_N, x_values_E, x_values_S, x_values_W], [y_values_N, y_values_E, y_values_S, y_values_W]):

        res = stats.linregress(x, y)
        slope = res.slope
        
        print ("slope:  ", slope)

    
def plotWindRoses():
    
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

# plots below can be considered added...

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
