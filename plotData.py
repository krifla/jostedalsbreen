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
        ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color='k', s=2)


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
    
    title = f'Difference in precipitation ({exp2} - {exp1})\nfor {season} months ({years[0]}-{years[-1]+1} average)'
    title = f'{exp2lab} - {exp1lab}'#' ({years[0]}-{years[-1]} average)'
    
#    if season == 'SONDJFMAM' and exp2 == 'noice_BT':
#        vmin = -2000/len(years); vmax = 2000/len(years)
#    elif season == 'SONDJFMAM' and exp2 == 'noice_dtm50':
#        vmin = -300/len(years); vmax = 300/len(years)
    #if exp2 == 'noice_BT':# and season == 'all':
    #    vmin = -20; vmax = 20
    
    precip_model_exp1 = calculateAccPrecip(precip_model_exp1)
    precip_model_exp2 = calculateAccPrecip(precip_model_exp2)
    
    # Define the discrete levels and use coolwarm_r colormap
    levels = np.arange(vmin, vmax+1, 1)
    cmap = plt.get_cmap("coolwarm_r")

    # Create a normalization using BoundaryNorm
    norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)

    
    fig, ax = plt.subplots(figsize=(15,10))
    #plt.rcParams.update({'font.size': 16}) 

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
        mc = 'b'

        for (i, j) in indices:
            #ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)
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
        
#    for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ']):
#        if st not in [' '] and ~np.isnan(obs_precip[s]):
#            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'].values[0])
#            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'].values[0])
#            if obs_precip[s] > 0:
#                ax.scatter(lon, lat, c='k', zorder=1000)
#                ax.text(lon, lat, 
#                        f'   {obs_precip[s]/len(years):.0f}', fontsize=14, va='center')

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

    if season == 'all':
        vmax = 3000
        
    precip_model = calculateAccPrecip(snow_model)
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
        ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color='k', s=2)


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

    # plot configurations

    cb = plt.colorbar(cm, extend='max', pad=.02)
    #cbar_ax = fig.add_axes([0.135, -.05, .75, .03])  # [left, bottom, width, height]
    #cb = plt.colorbar(cm, extend='max', cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=13)
    cb.set_label('Mean annual snow (mm)', fontsize=15, rotation=270, labelpad=28)

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
    
#    if season == 'SONDJFMAM' and exp2 == 'noice_BT':
#        vmin = -2000/len(years); vmax = 2000/len(years)
#    elif season == 'SONDJFMAM' and exp2 == 'noice_dtm50':
#        vmin = -300/len(years); vmax = 300/len(years)
#    if exp2 == 'noice_BT':# and season == 'all':
#        vmin = -20; vmax = 20
    
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

        
#    for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ']):
#        if st not in [' '] and ~np.isnan(obs_precip[s]):
#            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'].values[0])
#            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'].values[0])
#            if obs_precip[s] > 0:
#                ax.scatter(lon, lat, c='k', zorder=1000)
#                ax.text(lon, lat, 
#                        f'   {obs_precip[s]/len(years):.0f}', fontsize=14, va='center')

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
    
#    if season == 'SONDJFMAM' and exp2 == 'noice_BT':
#        vmin = -2000/len(years); vmax = 2000/len(years)
#    elif season == 'SONDJFMAM' and exp2 == 'noice_dtm50':
#        vmin = -300/len(years); vmax = 300/len(years)
#    if exp2 == 'noice_BT':# and season == 'all':
#        vmin = -20; vmax = 20
    
    precip_model_exp1 = calculateAccPrecip(precip_model_exp1)
    precip_model_exp2 = calculateAccPrecip(precip_model_exp2)
    snow_model_exp1 = calculateAccPrecip(snow_model_exp1)
    snow_model_exp2 = calculateAccPrecip(snow_model_exp2)
    
    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams.update({'font.size': 16}) 

    # plot model data
    
    print ((precip_model_exp2-precip_model_exp1).max())
    print ((snow_model_exp2-snow_model_exp1).max())
    print ((precip_model_exp2-precip_model_exp1).min())
    print ((snow_model_exp2-snow_model_exp1).min())

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
        
#    for s, st in enumerate(['MG', 'FL', 'VS', 'OD', 'SJ']):
#        if st not in [' '] and ~np.isnan(obs_precip[s]):
#            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'].values[0])
#            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'].values[0])
#            if obs_precip[s] > 0:
#                ax.scatter(lon, lat, c='k', zorder=1000)
#                ax.text(lon, lat, 
#                        f'   {obs_precip[s]/len(years):.0f}', fontsize=14, va='center')

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
                        
    #print (key, np.nansum(total_days), total_days, weighted_sum)

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

    cm2 = ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, levels=levels2, cmap='terrain')
    cm3 = ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1*np.nan, levels=levels2, linewidths=5, cmap='terrain')

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
                linewidth=.8,  # Thickness of the frame
                edgecolor=mc,  # Color of the frame
                facecolor='none'  # No fill color
            )
            ax.add_patch(rect)
        
    ice_mask = ((WRF_lu_exp1.values == 24))
    #print ('min, mean, max over ice: ', (temp_model_exp2-temp_model_exp1)[ice_mask].min(), (temp_model_exp2-temp_model_exp1)[ice_mask].mean(), (temp_model_exp2-temp_model_exp1)[ice_mask].max())
    #print ('min, mean, max outside of ice: ', (temp_model_exp2-temp_model_exp1)[~ice_mask].min(), (temp_model_exp2-temp_model_exp1)[~ice_mask].mean(), (temp_model_exp2-temp_model_exp1)[~ice_mask].max())

#    obs_temp = calculateMeanTemp(obs_temp)
#        
#    for s, st in enumerate(stations):#['MG', 'FL', 'VS', 'OD', 'SJ']):
#        if st not in [' '] and ~np.isnan(obs_temp[s]):
#            lon = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lon'][0])
#            lat = (WRF1000_ts.loc[WRF1000_ts['station_id']==st,'station_lat'][0])
#            if obs_temp[s] > 0:
#                ax.scatter(lon, lat, c='k', zorder=1000)
#                ax.text(lon, lat, 
#                        f'   {obs_temp[s]:.0f}', fontsize=14, va='center')

    #cb = plt.colorbar(cm3, extend='both')#, ticks=levels[1::3])
    #cb.set_label('Elevation (m)', rotation=270, labelpad=30)
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
    
    ## add dots where there is ice or snow
    #
    #indices = np.argwhere((WRF_lu_exp1.values == 24) & (WRF_lu_exp2.values != 24))
    #mc = 'k'
    #if exp1 == 'noice_BT' and exp2 == 'modlakes_noice_BT':
    #    indices = np.argwhere((WRF_lu_exp1.values != 16) & (WRF_lu_exp2.values == 16))
    #    mc = 'b'
    #for (i, j) in indices:
    #    ax.scatter(WRF_lon[i,j], WRF_lat[i,j], color=mc, s=2)

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

    
def plotHGTDiff(title=title, vmin=-20, vmax=20):
        
    fig, ax = plt.subplots(figsize=(15,10))
    plt.rcParams.update({'font.size': 16}) 
    
    # plot model data

    cm = ax.pcolormesh(WRF_lon, WRF_lat, WRF_hgt_exp1-WRF_hgt_exp2,
                       vmin=-300, vmax=300,
                       cmap='coolwarm', 
                       shading='auto')  # Use shading='auto' for better visuals
    cbar = plt.colorbar(cm)
    cbar.ax.set_ylim(-50, 300)
    cbar.ax.set_ylabel('Difference in elevation (m)', rotation=270, labelpad=20)
    
    ax.contour(WRF_lon, WRF_lat, WRF_hgt_exp1, cmap='terrain')
    
    ax.set_xlabel('Longitude ($\u00b0$)')
    ax.set_ylabel('Latitude ($\u00b0$)')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.show()
    
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
