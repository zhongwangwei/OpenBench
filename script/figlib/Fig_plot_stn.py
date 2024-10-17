import matplotlib
import matplotlib.pyplot as plt
# from   mpl_toolkits.basemap import Basemap
from   matplotlib import rcParams

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

def plot_stn(casedir,sim,obs,ID,key,RMSE,KGESS,correlation):        
    legs =['obs','sim']
    lines=[1.5, 1.5]
    alphas=[1.,1.]
    linestyles=['solid','dotted']
    colors=['g',"purple"]
    fig, ax = plt.subplots(1,1,figsize=(10,5))

    obs.plot.line (x='time', label='obs', linewidth=lines[0], linestyle=linestyles[0], alpha=alphas[0],color=colors[0]                ) 
    sim.plot.line (x='time', label='sim', linewidth=lines[0], linestyle=linestyles[1], alpha=alphas[1],color=colors[1],_add_legend=True) 
    #set ylabel to be the same as the variable name
    ax.set_ylabel(key, fontsize=18)        
    #ax.set_ylabel(f'{obs}', fontsize=18)
    ax.set_xlabel('Date', fontsize=18)
    ax.tick_params(axis='both', top='off', labelsize=16)
    ax.legend(loc='upper right', shadow=False, fontsize=14)
    #add RMSE,KGE,correlation in two digital to the legend in left top
    ax.text(0.01, 0.95, f'RMSE: {RMSE:.2f} \n R: {correlation:.2f} \n KGESS: {KGESS:.2f} ', transform=ax.transAxes, fontsize=14,verticalalignment='top')
    plt.tight_layout()
    plt.savefig(f'{casedir}/tmp/plt/{key[0]}_{ID}_timeseries.png')
    plt.close(fig)