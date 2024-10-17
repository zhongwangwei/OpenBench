import pandas as pd
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import colors
# from   mpl_toolkits.basemap import Basemap
from matplotlib import rcParams

font = {'family': 'DejaVu Sans'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
            'axes.labelsize': 10,
            'grid.linewidth': 0.2,
            'font.size': 12,
            'legend.fontsize': 12,
            'legend.frameon': False,
            'xtick.labelsize': 12,
            'xtick.direction': 'out',
            'ytick.labelsize': 12,
            'ytick.direction': 'out',
            'savefig.bbox': 'tight',
            'axes.unicode_minus': False,
            'text.usetex': False}
rcParams.update(params)

def plot_stn_map(stn_lon, stn_lat, metric, cmap, norm, ticks,key,varname,min_lon,max_lon,min_lat,max_lat,casedir):
    from pylab import rcParams
    # from mpl_toolkits.basemap import Basemap
    import matplotlib
    import matplotlib.pyplot as plt
    ### Plot settings
    font = {'family' : 'DejaVu Sans'}
    #font = {'family' : 'Myriad Pro'}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
        'axes.labelsize': 12,
        'grid.linewidth': 0.2,
        'font.size': 15,
        'legend.fontsize': 12,
        'legend.frameon': False,
        'xtick.labelsize': 12,
        'xtick.direction': 'out',
        'ytick.labelsize': 12,
        'ytick.direction': 'out',
        'savefig.bbox': 'tight',
        'axes.unicode_minus': False,
        'text.usetex': False}
    rcParams.update(params)
    fig = plt.figure()
    M = Basemap(projection='cyl',llcrnrlat=min_lat,urcrnrlat=max_lat,\
                llcrnrlon=min_lon,urcrnrlon=max_lon,resolution='l')

    #fig.set_tight_layout(True)
    #M = Basemap(projection='robin', resolution='l', lat_0=15, lon_0=0)
    M.drawmapboundary(fill_color='white', zorder=-1)
    M.fillcontinents(color='0.8', lake_color='white', zorder=0)
    M.drawcoastlines(color='0.6', linewidth=0.1)
    #M.drawcountries(color='0.6', linewidth=0.1)
    # M.drawparallels(np.arange(-60.,60.,30.), dashes=[1,1], linewidth=0.25, color='0.5')
    #M.drawmeridians(np.arange(0., 360., 60.), dashes=[1,1], linewidth=0.25, color='0.5')
    loc_lon, loc_lat = M(stn_lon, stn_lat)
    cs = M.scatter(loc_lon, loc_lat, 15, metric, cmap=cmap, norm=norm, marker='.', edgecolors='none', alpha=0.9)
    cbaxes = fig.add_axes([0.26, 0.31, 0.5, 0.015])
    cb = fig.colorbar(cs, cax=cbaxes, ticks=ticks, orientation='horizontal', spacing='uniform')
    cb.solids.set_edgecolor("face")
    cb.set_label('%s'%(varname), position=(0.5, 1.5), labelpad=-35)
    plt.savefig(f'{casedir}/output/{key}_{varname}_validation.png',  format='png',dpi=400)
    plt.close()

def make_stn_plot_index(casedir,ref_varname,metrics):
    # read the data
    df = pd.read_csv(f'{casedir}/output/{ref_varname[0]}_metric.csv', header=0)
    for metric in metrics:
        min_metric  = -999.0
        max_metric  = 100000.0
        #print(df['%s'%(metric)])
        ind0 = df[df['%s'%(metric)]>min_metric].index
        data_select0 = df.loc[ind0]
        #print(data_select0[data_select0['%s'%(metric)] < max_metric])
        ind1 = data_select0[data_select0['%s'%(metric)] < max_metric].index
        data_select = data_select0.loc[ind1]

        try:
            lon_select = data_select['ref_lon'].values
            lat_select = data_select['ref_lat'].values
        except:
            lon_select = data_select['sim_lon'].values
            lat_select = data_select['sim_lat'].values 
        plotvar=data_select['%s'%(metric)].values
        if metric == 'pc_bias':
            vmin=-100.0
            vmax= 100.0
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd, ref_varname[0],metric)
        elif metric == 'KGE':
            vmin=-1
            vmax= 1
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd, ref_varname[0],metric)
        elif metric == 'KGESS':
            vmin=-1
            vmax= 1
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd, ref_varname[0],metric)
        elif metric == 'NSE':
            vmin=-1
            vmax= 1
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd,  ref_varname[0],metric)
        elif metric == 'correlation':
            vmin=-1
            vmax= 1
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd,  ref_varname[0],metric) 
        elif metric == 'index_agreement':
            vmin=-1
            vmax= 1
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd,  ref_varname[0],metric)
