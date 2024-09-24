def make_geo_plot_index(ref_varname,metrics):
    from matplotlib import colors
    key=ref_varname
    for metric in metrics:
        print(f'plotting metric: {metric}')
        if metric in ['bias', 'mae', 'ubRMSE', 'apb', 'RMSE', 'L','pc_bias','apb']:
            vmin = -100.0
            vmax = 100.0
        elif metric in ['KGE', 'NSE', 'correlation']:
            vmin = -1
            vmax = 1
        elif metric in ['correlation_R2', 'index_agreement']:
            vmin = 0
            vmax = 1
        else:
            vmin = -1
            vmax = 1
        bnd = np.linspace(vmin, vmax, 11)
        cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
        cmap = colors.ListedColormap(cpool)
        norm = colors.BoundaryNorm(bnd, cmap.N)
        plot_geo_map(casedir,cmap, norm, key, bnd,metric)
    
    for score in self.scores:
        print(f'plotting score: {score}')
        if score in ['KGESS']:
            vmin = -1
            vmax = 1
        elif score in ['nBiasScore','nRMSEScore']:
            vmin = 0
            vmax = 1
        else:
            vmin = -1
            vmax = 1
        bnd = np.linspace(vmin, vmax, 11)
        cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
        cmap = colors.ListedColormap(cpool)
        norm = colors.BoundaryNorm(bnd, cmap.N)
        plot_geo_map(casedir,cmap, norm, key, bnd,score)

def plot_geo_map(casedir, colormap, normalize, key, levels, xitem, **kwargs):
    # Plot settings
    # Set the region of the map based on self.Max_lat, self.Min_lat, self.Max_lon, self.Min_lon
    ds=xr.open_dataset(f'{casedir}/output/{key}_{xitem}.nc')
    # Extract variables
    lat = ds.lat.values
    lon = ds.lon.values
    lat, lon = np.meshgrid(lat[::-1], lon)

    var = ds[xitem].transpose("lon", "lat")[:, ::-1].values

    fig = plt.figure()
    M = Basemap(projection='cyl', llcrnrlat=self.min_lat, urcrnrlat=self.max_lat,
                llcrnrlon=self.min_lon, urcrnrlon=self.max_lon, resolution='l')

    M.drawmapboundary(fill_color='white', zorder=-1)
    M.fillcontinents(color='0.8', lake_color='white', zorder=0)
    M.drawcoastlines(color='0.6', linewidth=0.1)
    loc_lon, loc_lat = M(lon, lat)
    cs = M.contourf(loc_lon, loc_lat, var, cmap=colormap, norm=normalize, levels=levels, extend='both')
    cbaxes = fig.add_axes([0.26, 0.31, 0.5, 0.015])
    cb = fig.colorbar(cs, cax=cbaxes, ticks=levels, orientation='horizontal', spacing='uniform')
    cb.solids.set_edgecolor("face")
    cb.set_label('%s' % (xitem), position=(0.5, 1.5), labelpad=-35)
    plt.savefig(f'{casedir}/output/{key}_{xitem}.png', format='png', dpi=300)
    plt.close()