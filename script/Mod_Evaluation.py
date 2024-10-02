import numpy as np
import os, sys
import xarray as xr
import shutil 
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import colors
# Check the platform
from Mod_Metrics import metrics
from Mod_Scores import scores
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
from matplotlib import colors

class Evaluation_grid(metrics,scores):
    def __init__(self, info):
        self.name = 'Evaluation_grid'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.__dict__.update(info)
        os.makedirs(self.casedir+'/output/', exist_ok=True)

        print(" ")
        print("\033[1;32m╔═══════════════════════════════════════════════════════════════╗\033[0m")
        print("\033[1;32m║                Evaluation processes starting!                 ║\033[0m")
        print("\033[1;32m╚═══════════════════════════════════════════════════════════════╝\033[0m")
        print("\n")

    def process_metric(self, metric, s, o, vkey=''):
        pb = getattr(self, metric)(s, o)
        pb=pb.squeeze()
        pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=metric)
        pb_da.to_netcdf(f'{self.casedir}/output/metrics/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{metric}{vkey}.nc')

    def process_score(self, score, s, o, vkey=''):
        pb = getattr(self, score)(s, o)
        pb=pb.squeeze()
        pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=score)
        pb_da.to_netcdf(f'{self.casedir}/output/scores/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{score}{vkey}.nc')

    def make_Evaluation(self, **kwargs):
        o = xr.open_dataset(f'{self.casedir}/output/data/{self.item}_ref_{self.ref_source}_{self.ref_varname}.nc')[f'{self.ref_varname}'] 
        s = xr.open_dataset(f'{self.casedir}/output/data/{self.item}_sim_{self.sim_source}_{self.sim_varname}.nc')[f'{self.sim_varname}'] 

        s['time'] = o['time']

        mask1 = np.isnan(s) | np.isnan(o)
        s.values[mask1] = np.nan
        o.values[mask1] = np.nan
        print("\033[1;32m" + "=" * 80 + "\033[0m")
        for metric in self.metrics:
            if hasattr(self, metric):
                print(f'calculating metric: {metric}')
                self.process_metric(metric, s, o)
            else:
                print('No such metric')
                sys.exit(1)

        for score in self.scores:
            if hasattr(self, score):
                print(f'calculating score: {score}')
                self.process_score(score, s, o)
            else:
                print('No such score')
                sys.exit(1)

        print("\033[1;32m" + "=" * 80 + "\033[0m")

        return

    def make_plot_index(self):
        key=self.ref_varname
        for metric in self.metrics:
            print(f'plotting metric: {metric}')
            if metric in ['bias', 'mean_absolute_error', 'ubRMSE', 'absolute_percent_bias', 'RMSE', 'L','percent_bias','absolute_percent_bias']:
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
            self.plot_map(cmap, norm, key, bnd,metric,'metrics')
     
        #print("\033[1;32m" + "=" * 80 + "\033[0m")
        for score in self.scores:
            print(f'plotting score: {score}')
            if score in ['KGESS']:
                vmin = -1
                vmax = 1
            elif score in ['nBiasScore','nRMSEScore']:
                vmin = 0
                vmax = 1
            else:
                vmin = 0
                vmax = 1
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            self.plot_map(cmap, norm, key, bnd,score,'scores')
        print("\033[1;32m" + "=" * 80 + "\033[0m")

    def plot_map(self, colormap, normalize, key, levels, xitem, k, **kwargs):
        # Plot settings
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        import xarray as xr
        from mpl_toolkits.basemap import Basemap
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

        # Set the region of the map based on self.Max_lat, self.Min_lat, self.Max_lon, self.Min_lon
        ds=xr.open_dataset(f'{self.casedir}/output/{k}/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{xitem}.nc')

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
        plt.savefig(f'{self.casedir}/output/{k}/{self.item}_ref_{self.ref_source}_sim_{self.sim_source}_{xitem}.png', format='png', dpi=300)
        plt.close()

class Evaluation_stn(metrics,scores):
    def __init__(self,info):
        self.name = 'Evaluation_point'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"

        self.__dict__.update(info)
        if isinstance(self.sim_varname, str): self.sim_varname = [self.sim_varname]
        if isinstance(self.ref_varname, str): self.ref_varname = [self.ref_varname]

        print ('Evaluation processes starting!')
        print("=======================================")
        print(" ")
        print(" ")  

    def make_evaluation(self):
        #read station information
        stnlist  =f"{self.casedir}/stn_list.txt"
        station_list = pd.read_csv(stnlist,header=0)

        # loop the keys in self.variables to get the metric output
        for metric in self.metrics:
            station_list[f'{metric}']=[-9999.0] * len(station_list['ID'])
        for score in self.scores:
            station_list[f'{score}']=[-9999.0] * len(station_list['ID'])
        for iik in range(len(station_list['ID'])):
            s=xr.open_dataset(f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/sim_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[self.sim_varname]
            o=xr.open_dataset(f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/ref_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[self.ref_varname]
            s['time'] = o['time']
            mask1 = np.isnan(s) | np.isnan(o)
            s.values[mask1] = np.nan
            o.values[mask1] = np.nan
            
            for metric in self.metrics:
                if hasattr(self, metric):
                    pb = getattr(self, metric)(s, o)
                    station_list.loc[iik, f'{metric}']=pb.values
                  #  self.plot_stn(s.squeeze(),o.squeeze(),station_list['ID'][iik],self.ref_varname, float(station_list['RMSE'][iik]), float(station_list['KGE'][iik]),float(station_list['correlation'][iik]))
                else:
                    print('No such metric')
                    sys.exit(1)

            for score in self.scores:
                if hasattr(self, score):
                    pb = getattr(self, score)(s, o)
                
                else:
                    print('No such score')
                    sys.exit(1)

            print("=======================================")
            print(" ")
            print(" ")

        
        print ('Comparison dataset prepared!')
        print("=======================================")
        print(" ")
        print(" ")  
        print(f"send {self.ref_varname} evaluation to {self.ref_varname}_{self.sim_varname}_metric.csv'")
        station_list.to_csv(f'{self.casedir}/output/metrics/stn_{self.ref_source}_{self.sim_source}/{self.ref_varname}_{self.sim_varname}_metric.csv',index=False)
        station_list.to_csv(f'{self.casedir}/output/scores/stn_{self.ref_source}_{self.sim_source}/{self.ref_varname}_{self.sim_varname}_scores.csv',index=False)

    def plot_stn(self,sim,obs,ID,key,RMSE,KGESS,correlation):
        from pylab import rcParams
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
        
        legs =['obs','sim']
        lines=[1.5, 1.5]
        alphas=[1.,1.]
        linestyles=['solid','dotted']
        colors=['g',"purple"]
        fig, ax = plt.subplots(1,1,figsize=(10,5))

        obs.plot.line (x='time', label='obs', linewidth=lines[0], linestyle=linestyles[0], alpha=alphas[0],color=colors[0]                ) 
        sim.plot.line (x='time', label='sim', linewidth=lines[0], linestyle=linestyles[1], alpha=alphas[1],color=colors[1],add_legend=True) 
        #set ylabel to be the same as the variable name
        ax.set_ylabel(key, fontsize=18)        
        #ax.set_ylabel(f'{obs}', fontsize=18)
        ax.set_xlabel('Date', fontsize=18)
        ax.tick_params(axis='both', top='off', labelsize=16)
        ax.legend(loc='upper right', shadow=False, fontsize=14)
        #add RMSE,KGE,correlation in two digital to the legend in left top
        ax.text(0.01, 0.95, f'RMSE: {RMSE:.2f} \n R: {correlation:.2f} \n KGESS: {KGESS:.2f} ', transform=ax.transAxes, fontsize=14,verticalalignment='top')
        plt.tight_layout()
        plt.savefig(f'{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/{key[0]}_{ID}_timeseries.png')
        plt.close(fig)

    def plot_stn_map(self, stn_lon, stn_lat, metric, cmap, norm, ticks,key,varname,s_m):
        from pylab import rcParams
        from mpl_toolkits.basemap import Basemap
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
        #set the region of the map based on self.Max_lat, self.Min_lat, self.Max_lon, self.Min_lon

        M = Basemap(projection='cyl',llcrnrlat=self.min_lat,urcrnrlat=self.max_lat,\
                    llcrnrlon=self.min_lon,urcrnrlon=self.max_lon,resolution='l')

        
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
        plt.savefig(f'{self.casedir}/output/{s_m}/{self.item}_stn_{self.ref_source}_{self.sim_source}_{varname}.png',  format='png',dpi=400)
        plt.close()

    def make_plot_index(self):
        # read the data
        df = pd.read_csv(f'{self.casedir}/output/scores/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv', header=0)
        # loop the keys in self.variables to get the metric output
        for metric in self.metrics:
            print(f'plotting metric: {metric}')
            min_metric  = -999.0
            max_metric  = 100000.0
            #print(df['%s'%(metric)])
            ind0 = df[df['%s'%(metric)]>min_metric].index
            data_select0 = df.loc[ind0]
            #print(data_select0[data_select0['%s'%(metric)] < max_metric])
            ind1 = data_select0[data_select0['%s'%(metric)] < max_metric].index
            data_select = data_select0.loc[ind1]
            #if key=='discharge':
            #    #ind2 = data_select[abs(data_select['err']) < 0.001].index
            #    #data_select = data_select.loc[ind2]
            #    ind3 = data_select[abs(data_select['area1']) > 1000.].index
            #    data_select = data_select.loc[ind3]
            try:
                lon_select = data_select['ref_lon'].values
                lat_select = data_select['ref_lat'].values
            except:
                lon_select = data_select['sim_lon'].values
                lat_select = data_select['sim_lat'].values 
            plotvar=data_select['%s'%(metric)].values
            if metric == 'percent_bias':
                vmin=-100.0
                vmax= 100.0
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd, self.ref_varname[0],metric,'metrics')
            elif metric == 'KGE':
                vmin=-1
                vmax= 1
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd, self.ref_varname[0],metric,'metrics')
            elif metric == 'KGESS':
                vmin=-1
                vmax= 1
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd, self.ref_varname[0],metric,'metrics')
            elif metric == 'NSE':
                vmin=-1
                vmax= 1
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd,  self.ref_varname[0],metric,'metrics')
            elif metric == 'correlation':
                vmin=-1
                vmax= 1
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd,  self.ref_varname[0],metric,'metrics') 
            elif metric == 'index_agreement':
                vmin=-1
                vmax= 1
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd,  self.ref_varname[0],metric,'metrics')


        for score in self.scores:
            print(f'plotting score: {score}')
            min_score  = -999.0
            max_score  = 100000.0
            #print(df['%s'%(score)])
            ind0 = df[df['%s'%(score)]>min_score].index
            data_select0 = df.loc[ind0]
            #print(data_select0[data_select0['%s'%(score)] < max_score])
            ind1 = data_select0[data_select0['%s'%(score)] < max_score].index
            data_select = data_select0.loc[ind1]
            #if key=='discharge':
            #    #ind2 = data_select[abs(data_select['err']) < 0.001].index
            #    #data_select = data_select.loc[ind2]
            #    ind3 = data_select[abs(data_select['area1']) > 1000.].index
            #    data_select = data_select.loc[ind3]
            try:
                lon_select = data_select['ref_lon'].values
                lat_select = data_select['ref_lat'].values
            except:
                lon_select = data_select['sim_lon'].values
                lat_select = data_select['sim_lat'].values 
            plotvar=data_select['%s'%(score)].values
            vmin=0
            vmax= 1
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd,  self.ref_varname[0],score,'scores')             

    def make_evaluation_parallel(self,station_list,iik):
        s=xr.open_dataset(f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/{self.item}_sim_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[self.sim_varname].to_array().squeeze()
        o=xr.open_dataset(f"{self.casedir}/output/data/stn_{self.ref_source}_{self.sim_source}/{self.item}_ref_{station_list['ID'][iik]}" + f"_{station_list['use_syear'][iik]}" + f"_{station_list['use_eyear'][iik]}.nc")[self.ref_varname].to_array().squeeze()

        s['time'] = o['time']
        mask1 = np.isnan(s) | np.isnan(o)
        s.values[mask1] = np.nan
        o.values[mask1] = np.nan
        #remove the nan values
        #s=s.dropna(dim='time').astype(np.float32)
        #o=o.dropna(dim='time').astype(np.float32)
        row={}
        # for based plot
        try:
            row['KGESS']=self.KGESS(s, o).values
        except:
            row['KGESS']=-9999.0
        try:
            row['RMSE']=self.rmse(s, o).values
        except:
            row['RMSE']=-9999.0 
        try:
            row['correlation']=self.correlation(s, o).values
        except:
            row['correlation']=-9999.0   
        
        for metric in self.metrics:
            if hasattr(self, metric):
                pb = getattr(self, metric)(s, o)
                if pb.values is not None:
                    row[f'{metric}']=pb.values
                else:
                    row[f'{metric}']=-9999.0
                    self.plot_stn(s.squeeze(),o.squeeze(),station_list['ID'][iik],self.ref_varname, float(station_list['RMSE'][iik]), float(station_list['KGE'][iik]),float(station_list['correlation'][iik]))
            else:
                print(f'No such metric: {metric}')
                sys.exit(1)

        for score in self.scores:
            if hasattr(self, score):
                pb2 = getattr(self, score)(s, o)
                #if pb2.values is not None:
                if pb2.values is not None:
                    row[f'{score}']=pb2.values
                else:
                    row[f'{score}']=-9999.0
            else:
                print('No such score')
                sys.exit(1)
        
        self.plot_stn(s,o,station_list['ID'][iik],self.ref_varname, float(row['RMSE']),float(row['KGESS']),float(row['correlation']))
        return row
        # return station_list
  
    def make_evaluation_P(self):
        stnlist  =f"{self.casedir}/stn_list.txt"
        station_list = pd.read_csv(stnlist,header=0)
        num_cores = os.cpu_count()  ##用来计算现在可以获得多少cpu核心。 也可以用multipocessing.cpu_count(),或者随意设定<=cpu核心数的数值
        #shutil.rmtree(f'{self.casedir}/output',ignore_errors=True)
        #creat tmp directory
        #os.makedirs(f'{self.casedir}/output', exist_ok=True)

        # loop the keys in self.variables
        # loop the keys in self.variables to get the metric output
        #for metric in self.metrics.keys():
        #    station_list[f'{metric}']=[-9999.0] * len(station_list['ID'])
        #if self.ref_source.lower() == 'grdc':

        results=Parallel(n_jobs=-1)(delayed(self.make_evaluation_parallel)(station_list,iik) for iik in range(len(station_list['ID'])))
        station_list = pd.concat([station_list, pd.DataFrame(results)], axis=1)

        print ('Evaluation finished')
        print("=======================================")
        print(" ")
        print(" ")  
        print(f"send evaluation to {self.casedir}/output/scores/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations")
        print(f"send evaluation to {self.casedir}/output/metrics/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations")

        station_list.to_csv(f'{self.casedir}/output/scores/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv',index=False)
        station_list.to_csv(f'{self.casedir}/output/metrics/{self.item}_stn_{self.ref_source}_{self.sim_source}_evaluations.csv',index=False)
