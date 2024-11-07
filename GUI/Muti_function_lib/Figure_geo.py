import math
from matplotlib import cm
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
# from mpl_toolkits.basemap import Basemap
from pylab import rcParams
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from io import BytesIO
import streamlit as st


def geo_single_average(option: dict, selected_item, refselect, simselect, ref, sim, var, filename):
    font = {'family': 'Times New Roman'}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.labelsize': option['fontsize'],
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option['fontsize'],
              'legend.fontsize': option['fontsize'],
              'legend.frameon': False,
              'xtick.labelsize': option['xtick'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['ytick'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    try:
        ds = xr.open_dataset(filename)
        lat = ds.lat.values
        lon = ds.lon.values
        lat, lon = np.meshgrid(lat[::-1], lon)
        ds = ds[var].mean('time', skipna=True).transpose("lon", "lat")[:, ::-1].values

        fig, ax = plt.subplots(1, figsize=(option['x_wise'], option['y_wise']), subplot_kw={'projection': ccrs.PlateCarree()})

        ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])
        mticks = ticks.tick_values(vmin=option['vmin'], vmax=option['vmax'])
        mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in
                  mticks]
        if mticks[0] < option['vmin'] and mticks[-1] < option['vmax']:
            mticks = mticks[1:]
        elif mticks[0] > option['vmin'] and mticks[-1] > option['vmax']:
            mticks = mticks[:-1]
        elif mticks[0] < option['vmin'] and mticks[-1] > option['vmax']:
            mticks = mticks[1:-1]
        option['vmax'], option['vmin'] = mticks[-1], mticks[0]

        if option['cpool'] is not None:
            cmap = cm.get_cmap(option['cpool'])
            bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
            norm = colors.BoundaryNorm(bnd, cmap.N)
        else:
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            bnd = np.arange(mticks[0], mticks[-1] + option['colorbar_ticks'] / 2, option['colorbar_ticks'] / 2)
            norm = colors.BoundaryNorm(bnd, cmap.N)

        cs = ax.contourf(lon, lat, ds, levels=bnd, alpha=1, cmap=cmap, norm=norm, extend=option["extend"])

        pos = ax.get_position()
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height

        if option['colorbar_position'] == 'horizontal':
            if len(option['xticklabel']) == 0:
                cbaxes = fig.add_axes([left + width / 6, bottom - 0.12, width / 3 * 2, 0.04])
            else:
                cbaxes = fig.add_axes([left + width / 6, bottom - 0.17, width / 3 * 2, 0.04])
        else:
            cbaxes = fig.add_axes([right + 0.05, bottom, 0.03, height])

        cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, orientation='horizontal', spacing='uniform')
        cb.solids.set_edgecolor("face")
        cb.set_label('%s' % (var), position=(0.5, bottom - 1.5))  # , labelpad=-60

        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '50m', edgecolor='0.6', facecolor='none')
        rivers = cfeature.NaturalEarthFeature(
            'physical', 'rivers_lake_centerlines', '110m', edgecolor='0.6', facecolor='none')
        ax.add_feature(cfeature.LAND, facecolor='0.8')
        ax.add_feature(coastline, linewidth=0.6)
        ax.add_feature(cfeature.LAKES, alpha=1, facecolor='white', edgecolor='white')
        ax.add_feature(rivers, linewidth=0.5)

        if option['grid']:
            ax.gridlines(draw_labels=False, linestyle=option['grid_style'], linewidth=option['grid_linewidth'], color='k',
                         alpha=0.8)
        ax.set_extent([option['min_lon'], option['max_lat'], option['min_lat'], option['max_lat']])

        ax.set_title(option['title'], fontsize=option['title_size'])
        ax.set_xlabel(option['xticklabel'], fontsize=option['xtick'] + 1)  # labelpad=xoffsets
        ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1)

        ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -60)[::-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -30)[::-1], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        st.pyplot(fig)

        # 将图像保存到 BytesIO 对象
        buffer = BytesIO()
        fig.savefig(buffer, format=option['saving_format'], dpi=300)
        buffer.seek(0)
        st.download_button('Download image', buffer, file_name=f'{option["plot_type"]}.{option["saving_format"]}',
                           mime=f"image/{option['saving_format']}",
                           type="secondary", disabled=False, use_container_width=False)

    except FileNotFoundError:
        st.error(f"File {filename} not found. Please check the file path.")
    except KeyError as e:
        st.error(f"Key error: {e}. Please check the keys in the option dictionary.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


def geo_average_diff(option: dict, selected_item, refselect, simselect, ref, sim, refvar, simvar):
    from matplotlib import colors
    font = {'family': 'Times New Roman'}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.labelsize': option['fontsize'],
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option['fontsize'],
              'legend.fontsize': option['fontsize'],
              'legend.frameon': False,
              'xtick.labelsize': option['xtick'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['ytick'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    try:
        simfile = f'{option["data_path"]}/{selected_item}_sim_{simselect}_{simvar}.nc'
        ds_sim = xr.open_dataset(simfile)

    except FileNotFoundError:
        st.error(f"File {simfile} not found. Please check the file path.")
    except KeyError as e:
        st.error(f"Key error: {e}. Please check the keys in the option dictionary.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    try:
        reffile = f'{option["data_path"]}/{selected_item}_ref_{refselect}_{refvar}.nc'
        ds_ref = xr.open_dataset(reffile)
    except FileNotFoundError:
        st.error(f"File {reffile} not found. Please check the file path.")
    except KeyError as e:
        st.error(f"Key error: {e}. Please check the keys in the option dictionary.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    lat = ds_sim.lat.values
    lon = ds_sim.lon.values
    lat, lon = np.meshgrid(lat[::-1], lon)
    ds_ref = ds_ref[refvar].mean('time', skipna=True).transpose("lon", "lat")[:, ::-1].values
    ds_sim = ds_sim[simvar].where(ds_sim[simvar] > -999, np.nan).mean('time', skipna=True).transpose("lon", "lat")[:, ::-1].values
    fig, axes = plt.subplots(2, 2, figsize=(option['x_wise'], option['y_wise']), subplot_kw={'projection': ccrs.PlateCarree()})

    ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])

    option['vmin'] = math.floor(np.nanmin(ds_sim))
    option['vmax'] = math.ceil(np.nanmax(ds_sim))
    mticks, cmap, bnd, norm = get_color_info(option['vmin'], option['vmax'], ticks, option['cpool'], option['colorbar_ticks'])
    option['vmax'], option['vmin'] = mticks[-1], mticks[0]

    cs1 = axes[0][0].contourf(lon, lat, ds_sim, levels=bnd, alpha=1, cmap=cmap, norm=norm, extend='both')  # , extend='both'
    pos = axes[0][0].get_position()
    left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
    cbaxes = fig.add_axes([right + 0.008, bottom, 0.01, height])
    cb = fig.colorbar(cs1, cax=cbaxes, ticks=bnd[::2], orientation='vertical', spacing='uniform')
    cb.solids.set_edgecolor("face")
    axes[0][0].set_title('Simulation', fontsize=option['title_size'])

    option['vmin'] = math.floor(np.nanmin(ds_ref))
    option['vmax'] = math.ceil(np.nanmin(ds_ref))
    mticks, cmap, bnd, norm = get_color_info(option['vmin'], option['vmax'], ticks, option['cpool'], option['colorbar_ticks'])
    option['vmax'], option['vmin'] = mticks[-1], mticks[0]
    cs2 = axes[0][1].contourf(lon, lat, ds_ref, levels=bnd, alpha=1, cmap=cmap, norm=norm, extend='both')  # , extend='both'
    pos = axes[0][1].get_position()
    left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
    cbaxes = fig.add_axes([right + 0.008, bottom, 0.01, height])
    cb = fig.colorbar(cs2, cax=cbaxes, ticks=bnd[::2], orientation='vertical', spacing='uniform')
    cb.solids.set_edgecolor("face")
    axes[0][1].set_title('Reference', fontsize=option['title_size'])

    diff = ds_ref - ds_sim
    option['vmin'] = math.floor(np.nanmin(diff))
    option['vmax'] = math.ceil(np.nanmax(diff))
    mticks, cmap, levels, normalize = get_color_info(option['vmin'], option['vmax'], ticks, option['cpool'], option['colorbar_ticks'])

    cs3 = axes[1][0].contourf(lon, lat, diff, levels=levels, alpha=1, cmap=cmap, norm=normalize, extend='both')  # , extend='both'
    pos = axes[1][0].get_position()
    left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
    cbaxes = fig.add_axes([right + 0.01, bottom, 0.01, height])
    cb = fig.colorbar(cs3, cax=cbaxes, ticks=levels, orientation='vertical', spacing='uniform')
    cb.solids.set_edgecolor("face")
    axes[1][0].set_title('Difference between ref and sim', fontsize=option['title_size'])

    for ax in axes.flat:
        coastline = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '50m', edgecolor='0.6', facecolor='none')
        rivers = cfeature.NaturalEarthFeature(
            'physical', 'rivers_lake_centerlines', '110m', edgecolor='0.6', facecolor='none')
        ax.add_feature(cfeature.LAND, facecolor='0.8')
        ax.add_feature(coastline, linewidth=0.6)
        ax.add_feature(cfeature.LAKES, alpha=1, facecolor='white', edgecolor='white')
        ax.add_feature(rivers, linewidth=0.5)


        if option['grid']:
            ax.gridlines(draw_labels=False, linestyle=option['grid_style'], linewidth=option['grid_linewidth'], color='k',
                         alpha=0.8)
        ax.set_extent([option['min_lon'], option['max_lat'], option['min_lat'], option['max_lat']])

        ax.set_xlabel(option['xticklabel'], fontsize=option['xtick'] + 1)  # labelpad=xoffsets
        ax.set_ylabel(option['yticklabel'], fontsize=option['ytick'] + 1)
        ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -60)[::-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -30)[::-1], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    fig.delaxes(axes[-1][-1])
    fig.subplots_adjust(hspace=option['hspace'], wspace=option['wspace'])
    st.pyplot(fig)

    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=300)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{option["plot_type"]}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)


def get_color_info(vmin, vmax, ticks, cpool, colorbar_ticks):
    # if 2 >= vmax - vmin > 1:
    #     colorbar_ticks = 0.2
    # elif 10 >= vmax - vmin > 5:
    #     colorbar_ticks = 1
    # elif 100 >= vmax - vmin > 10:
    #     colorbar_ticks = 5
    # elif 200 >= vmax - vmin > 100:
    #     colorbar_ticks = 20
    # elif 500 >= vmax - vmin > 200:
    #     colorbar_ticks = 50
    # elif 1000 >= vmax - vmin > 500:
    #     colorbar_ticks = 100
    # elif 2000 >= vmax - vmin > 1000:
    #     colorbar_ticks = 200
    # elif 10000 >= vmax - vmin > 2000:
    #     colorbar_ticks = 10 ** math.floor(math.log10(vmax - vmin)) / 2
    # else:
    #     colorbar_ticks = 0.1
    mticks = ticks.tick_values(vmin=vmin, vmax=vmax)
    # st.write(mticks)
    mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in
              mticks]
    # if mticks[0] < vmin and mticks[-1] < vmax:
    #     mticks = mticks[1:]
    # elif mticks[0] > vmin and mticks[-1] > vmax:
    #     mticks = mticks[:-1]
    # elif mticks[0] < vmin and mticks[-1] > vmax:
    #     mticks = mticks[1:-1]
    # if mticks[0] == mticks[-1]:
    #     mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in
    #               mticks]
    if cpool is not None:
        cmap = cm.get_cmap(cpool)
        bnd = np.arange(mticks[0], mticks[-1] + colorbar_ticks / 2, colorbar_ticks / 2)
        norm = colors.BoundaryNorm(bnd, cmap.N)
    else:
        cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
        cmap = colors.ListedColormap(cpool)
        bnd = np.arange(mticks[0], mticks[-1] + colorbar_ticks / 2, colorbar_ticks / 2)
        norm = colors.BoundaryNorm(bnd, cmap.N)
    return mticks, cmap, bnd, norm
