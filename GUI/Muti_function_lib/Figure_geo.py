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


def get_color_info(vmin, vmax, ticks, cpool, colorbar_ticks):
    mticks = ticks.tick_values(vmin=vmin, vmax=vmax)

    mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in
              mticks]

    if mticks[0] < vmin and mticks[-1] < vmax:
        mticks = mticks[1:]
    elif mticks[0] > vmin and mticks[-1] > vmax:
        mticks = mticks[:-1]
    elif mticks[0] < vmin and mticks[-1] > vmax:
        mticks = mticks[1:-1]

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


def geo_single_average(option: dict, selected_item, refselect, simselect, ref, sim, var, filename):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.labelsize': option['fontsize'],
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option['fontsize'],
              'legend.fontsize': option['fontsize'],
              'legend.frameon': False,
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    try:
        ds = xr.open_dataset(filename)
        ilat = ds.lat.values
        ilon = ds.lon.values
        ds = ds[var].mean('time', skipna=True)  # .transpose("lon", "lat")[:, ::-1].values

        fig, ax = plt.subplots(1, figsize=(option['x_wise'], option['y_wise']), subplot_kw={'projection': ccrs.PlateCarree()})

        ticks = matplotlib.ticker.MultipleLocator(base=option['colorbar_ticks'])
        mticks, cmap, bnd, norm = get_color_info(option['vmin'], option['vmax'], ticks, option['cpool'],
                                                 option['colorbar_ticks'])

        if ilat[0] - ilat[-1] < 0:
            option['origin'] = 'lower'
        else:
            option['origin'] = 'upper'

        if option["map"] == 'imshow':
            extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])
            cs = ax.imshow(ds, cmap=cmap, vmin=option['vmin'], vmax=option['vmax'], extent=extent,
                           origin=option['origin'])
        elif option['map'] == 'contourf':
            lat, lon = np.meshgrid(ilat[::-1], ilon)
            cs = ax.contourf(lon, lat, ds.transpose("lon", "lat")[:, ::-1].values, levels=bnd, alpha=1, cmap=cmap, norm=norm,
                             extend=option["extend"])

        pos = ax.get_position()
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height

        if option['colorbar_position'] == 'horizontal':
            if len(option['xticklabel']) == 0:
                cbaxes = fig.add_axes([left + width / 6, bottom - 0.12, width / 3 * 2, 0.04])
            else:
                cbaxes = fig.add_axes([left + width / 6, bottom - 0.17, width / 3 * 2, 0.04])
        else:
            cbaxes = fig.add_axes([right + 0.05, bottom, 0.03, height])

        cb = fig.colorbar(cs, cax=cbaxes, orientation='horizontal', spacing='uniform')
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
        ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']])

        ax.set_title(option['title'], fontsize=option['title_size'])
        ax.set_xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)  # labelpad=xoffsets
        ax.set_ylabel(option['yticklabel'], fontsize=option['yticksize'] + 1)

        ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -1 * option["xtick"])[::-1], crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -1 * option["ytick"])[::-1], crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        st.pyplot(fig)

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
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.labelsize': option['fontsize'],
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option['fontsize'],
              'legend.fontsize': option['fontsize'],
              'legend.frameon': False,
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
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

    # Simulation --------------------------------------------------------------------------
    ilat = ds_sim.lat.values
    ilon = ds_sim.lon.values
    lat, lon = np.meshgrid(ilat[::-1], ilon)
    ds_ref = ds_ref[refvar].mean('time', skipna=True)
    ds_sim = ds_sim[simvar].where(ds_sim[simvar] > -999, np.nan).mean('time', skipna=True)
    fig, axes = plt.subplots(2, 2, figsize=(option['x_wise'], option['y_wise']), subplot_kw={'projection': ccrs.PlateCarree()})

    ticks = matplotlib.ticker.MultipleLocator(base=option['sim_colorbar_ticks'])
    mticks, cmap, bnd, norm = get_color_info(option['sim_vmin'], option['sim_vmax'], ticks, option['cpool'],
                                             option['sim_colorbar_ticks'])
    option['sim_vmin'], option['sim_vmax'] = mticks[-1], mticks[0]

    if ilat[0] - ilat[-1] < 0:
        option['origin'] = 'lower'
    else:
        option['origin'] = 'upper'

    if option["map"] == 'imshow':
        extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])
        cs1 = axes[0][0].imshow(ds_sim, cmap=cmap, vmin=option['sim_vmin'], vmax=option['sim_vmax'], extent=extent,
                                origin=option['origin'])
    elif option['map'] == 'contourf':
        lat, lon = np.meshgrid(ilat[::-1], ilon)
        cs1 = axes[0][0].contourf(lon, lat, ds_sim.transpose("lon", "lat")[:, ::-1].values, levels=bnd, alpha=1, cmap=cmap,
                                  norm=norm, extend=option["extend"])  # , extend='both'

    pos = axes[0][0].get_position()
    left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
    cbaxes = fig.add_axes([right + 0.001, bottom + 0.05, 0.01, height - 0.05])
    cb = fig.colorbar(cs1, cax=cbaxes, ticks=bnd[::4], orientation='vertical', spacing='uniform')
    cb.solids.set_edgecolor("face")
    axes[0][0].set_title('Simulation', fontsize=option['title_size'])

    # Reference =======================================================================
    ticks = matplotlib.ticker.MultipleLocator(base=option['ref_colorbar_ticks'])
    mticks, cmap, bnd, norm = get_color_info(option['ref_vmin'], option['ref_vmax'], ticks, option['cpool'],
                                             option['ref_colorbar_ticks'])
    option['ref_vmin'], option['ref_vmax'] = mticks[-1], mticks[0]

    if ilat[0] - ilat[-1] < 0:
        option['origin'] = 'lower'
    else:
        option['origin'] = 'upper'

    if option["map"] == 'imshow':
        extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])
        cs2 = axes[0][1].imshow(ds_ref, cmap=cmap, vmin=option['ref_vmin'], vmax=option['ref_vmax'], extent=extent,
                                origin=option['origin'])
    elif option['map'] == 'contourf':
        lat, lon = np.meshgrid(ilat[::-1], ilon)
        cs2 = axes[0][1].contourf(lon, lat, ds_ref.transpose("lon", "lat")[:, ::-1].values, levels=bnd, alpha=1, cmap=cmap,
                                  norm=norm, extend=option["extend"])  # , extend='both'
    pos = axes[0][1].get_position()
    left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
    cbaxes = fig.add_axes([right + 0.001, bottom + 0.05, 0.01, height - 0.05])
    cb = fig.colorbar(cs2, cax=cbaxes, ticks=bnd[::4], orientation='vertical', spacing='uniform')
    cb.solids.set_edgecolor("face")
    axes[0][1].set_title('Reference', fontsize=option['title_size'])

    # difference --------------------------------------------------------------------------------
    diff = ds_ref - ds_sim
    ticks = matplotlib.ticker.MultipleLocator(base=option['diff_colorbar_ticks'])
    mticks, cmap, levels, normalize = get_color_info(option['diff_vmin'], option['diff_vmax'], ticks, option['cpool'],
                                                     option['diff_colorbar_ticks'])

    if ilat[0] - ilat[-1] < 0:
        option['origin'] = 'lower'
    else:
        option['origin'] = 'upper'

    if option["map"] == 'imshow':
        extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])
        cs3 = axes[1][0].imshow(diff, cmap='coolwarm', vmin=option['diff_vmin'], vmax=option['diff_vmax'], extent=extent,
                                origin=option['origin'])
    elif option['map'] == 'contourf':
        lat, lon = np.meshgrid(ilat[::-1], ilon)
        cs3 = axes[1][0].contourf(lon, lat, diff.transpose("lon", "lat")[:, ::-1].values, levels=bnd, alpha=1, cmap='coolwarm',
                                  norm=normalize, extend=option["extend"])  # , extend='both'

    pos = axes[1][0].get_position()
    left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
    cbaxes = fig.add_axes([right + 0.0005, bottom, 0.01, height - 0.05])
    cb = fig.colorbar(cs3, cax=cbaxes, ticks=levels[::2], orientation='vertical', spacing='uniform', extend=option['extend'])
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
        ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']])

        ax.set_xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)  # labelpad=xoffsets
        ax.set_ylabel(option['yticklabel'], fontsize=option['yticksize'] + 1)
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


def make_geo_time_average(selected_item, refselect, simselect, path, ref, sim, nl):
    option = {}
    fkey = 'geo_time_average_'
    with st.container(border=True):

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            option['title'] = st.text_input('Title', label_visibility="visible", key=f"{fkey}title")
            option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{fkey}title_size")
        with col2:
            option['xticklabel'] = st.text_input('X tick labels', value='Longitude', label_visibility="visible",
                                                 key=f"{fkey}xticklabel")
            option['xticksize'] = st.number_input("xtick label size", min_value=0, value=17, key=f"{fkey}xticksize")
        with col3:
            option['yticklabel'] = st.text_input('Y tick labels', value='Latitude', label_visibility="visible",
                                                 key=f"{fkey}yticklabel")
            option['yticksize'] = st.number_input("ytick label size", min_value=0, value=17, key=f"{fkey}yticksize")
        with col4:
            option['fontsize'] = st.number_input("Font size", min_value=0, value=17, key=f"{fkey}fontsize")
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1, key=f"{fkey}axes_linewidth")

        st.divider()

        color = '#9DA79A'
        st.markdown(f"""
        <div style="font-size:18px; font-weight:bold; color:{color}; border-bottom:3px solid {color}; padding: 5px;">
             Please choose Average type...
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns((2, 1))
        with col1:
            option['plot_type'] = st.radio('Please choose your type',
                                           ['Simulation', 'Reference', 'Differentiate'],
                                           index=None, label_visibility="collapsed", horizontal=True, key=f"{fkey}plot_type")

        with st.expander('More info', expanded=True):
            col1, col2, col3 = st.columns((1.5, 1, 1))
            option['grid'] = col1.toggle("Showing grid?", value=False, label_visibility="visible", key=f"{fkey}grid")
            if option['grid']:
                option['grid_style'] = col2.selectbox('Grid Line Style', ['solid', 'dotted', 'dashed', 'dashdot'],
                                                      index=2, placeholder="Choose an option", label_visibility="visible",
                                                      key=f"{fkey}grid_style")
                option['grid_linewidth'] = col3.number_input("grid linewidth", min_value=0, value=1, key=f"{fkey}grid_linewidth")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                option['max_lat'] = col1.number_input("Max latitude: ", value=float(st.session_state['generals']["max_lat"]),
                                                      key="geo_time_average_max_lat",
                                                      min_value=-90.0, max_value=90.0)
                option['min_lat'] = col2.number_input("Min latitude: ", value=float(st.session_state['generals']["min_lat"]),
                                                      key="geo_time_average_min_lat",
                                                      min_value=-90.0, max_value=90.0)
                option['max_lon'] = col3.number_input("Max Longitude: ", value=float(st.session_state['generals']["max_lon"]),
                                                      key="geo_time_average_max_lon",
                                                      min_value=-180.0, max_value=180.0)
                option['min_lon'] = col4.number_input("Min Longitude: ", value=float(st.session_state['generals']["min_lon"]),
                                                      key="geo_time_average_min_lon",
                                                      min_value=-180.0, max_value=180.0)

            def get_ticks(vmin, vmax):
                if 2 >= vmax - vmin > 1:
                    colorbar_ticks = 0.2
                elif 5 >= vmax - vmin > 2:
                    colorbar_ticks = 0.5
                elif 10 >= vmax - vmin > 5:
                    colorbar_ticks = 1
                elif 100 >= vmax - vmin > 10:
                    colorbar_ticks = 5
                elif 100 >= vmax - vmin > 50:
                    colorbar_ticks = 20
                elif 200 >= vmax - vmin > 100:
                    colorbar_ticks = 20
                elif 500 >= vmax - vmin > 200:
                    colorbar_ticks = 50
                elif 1000 >= vmax - vmin > 500:
                    colorbar_ticks = 100
                elif 2000 >= vmax - vmin > 1000:
                    colorbar_ticks = 200
                elif 10000 >= vmax - vmin > 2000:
                    colorbar_ticks = 10 ** math.floor(math.log10(vmax - vmin)) / 2
                else:
                    colorbar_ticks = 0.10
                return colorbar_ticks

            ref_type, sim_type = ('grid', 'grid')
            ref_var = ref[refselect][selected_item][f"varname"]
            try:
                sim_var = sim[simselect][selected_item][f"varname"]
            except:
                nml = nl.read_namelist(sim[simselect]['general']['model_namelist'])
                sim_var = nml[selected_item][f"varname"]

            option['data_path'] = path + f'/data/'

            import math
            if option['plot_type'] == 'Differentiate':
                sim_vmin_max_on = False
                sim_error = False
                try:
                    var = sim_var
                    filename = f'{option["data_path"]}/{selected_item}_sim_{simselect}_{sim_var}.nc'
                    if len(option['title']) == 0:
                        option['title'] = 'Simulation'
                    ds = xr.open_dataset(filename)
                    ds_sim = ds[var].mean('time', skipna=True)
                    sim_vmin = math.floor(np.nanmin(ds_sim))
                    sim_vmax = math.floor(np.nanmax(ds_sim))
                except Exception as e:
                    st.error(f"Error: {e}")
                    sim_error = True

                if not sim_error:
                    col1, col2, col3 = st.columns((4, 2, 2))
                    option["sim_vmin_max_on"] = col1.toggle('Setting Simulation max min', value=sim_vmin_max_on,
                                                            key=f"{fkey}sim_vmin_max_on")
                    if option["sim_vmin_max_on"]:
                        try:
                            option["sim_vmin"] = col2.number_input(f"colorbar min", value=sim_vmin, key=f"{fkey}sim_vmin")
                            option["sim_vmax"] = col3.number_input(f"colorbar max", value=sim_vmax, key=f"{fkey}sim_vmax")
                        except ValueError:
                            st.error(f"Max value must larger than min value.")
                    else:
                        option["sim_vmin"] = sim_vmin
                        option["sim_vmax"] = sim_vmax
                    sim_colorbar_ticks = get_ticks(option["sim_vmin"], option["sim_vmax"])
                else:
                    sim_colorbar_ticks = 0.5

                ref_vmin_max_on = False
                ref_error = False
                try:
                    var = ref_var
                    filename = f'{option["data_path"]}/{selected_item}_ref_{refselect}_{ref_var}.nc'
                    if len(option['title']) == 0:
                        option['title'] = 'Reference'
                    ds = xr.open_dataset(filename)

                    ds_ref = ds[var].mean('time', skipna=True)
                    ref_vmin = math.floor(np.nanmin(ds_ref))
                    ref_vmax = math.floor(np.nanmax(ds_ref))
                except Exception as e:
                    st.error(f"Error: {e}")
                    ref_error = True

                if not ref_error:
                    col1, col2, col3 = st.columns((4, 2, 2))
                    option["ref_vmin_max_on"] = col1.toggle('Setting Reference max min', value=ref_vmin_max_on,
                                                            key=f"{fkey}ref_vmin_max_on")
                    if option["ref_vmin_max_on"]:
                        try:
                            option["ref_vmin"] = col2.number_input(f"colorbar min", value=ref_vmin, key=f"{fkey}ref_vmin")
                            option["ref_vmax"] = col3.number_input(f"colorbar max", value=ref_vmax, key=f"{fkey}ref_vmax")
                        except ValueError:
                            st.error(f"Max value must larger than min value.")
                    else:
                        option["ref_vmin"] = ref_vmin
                        option["ref_vmax"] = ref_vmax
                    ref_colorbar_ticks = get_ticks(option["ref_vmin"], option["ref_vmax"])
                else:
                    ref_colorbar_ticks = 0.5

                diff_vmin_max_on = False
                diff_error = False
                try:
                    diff = ds_ref - ds_sim
                    diff_vmin = math.floor(np.nanmin(diff))
                    diff_vmax = math.floor(np.nanmax(diff))
                except Exception as e:
                    st.error(f"Error: {e}")
                    ref_error = True
                except Exception as e:
                    st.error(f"Error: {e}")
                    diff_error = True

                if not diff_error:
                    col1, col2, col3 = st.columns((4, 2, 2))
                    option["diff_vmin_max_on"] = col1.toggle('Setting Difference max min', value=ref_vmin_max_on,
                                                             key=f"{fkey}diff_vmin_max_on")
                    if option["diff_vmin_max_on"]:
                        try:
                            option["diff_vmin"] = col2.number_input(f"colorbar min", value=diff_vmin, key=f"{fkey}diff_vmin")
                            option["diff_vmax"] = col3.number_input(f"colorbar max", value=diff_vmax, key=f"{fkey}diff_vmax")
                        except ValueError:
                            st.error(f"Max value must larger than min value.")
                    else:
                        option["diff_vmin"] = diff_vmin
                        option["diff_vmax"] = diff_vmax
                    diff_colorbar_ticks = get_ticks(option["diff_vmin"], option["diff_vmax"])
                else:
                    diff_colorbar_ticks = 0.5

                st.write('##### :blue[Colorbar Ticks locater]')
                col1, col2, col3 = st.columns((3, 3, 3))
                option["sim_colorbar_ticks"] = col1.number_input(f"Simulation", value=float(sim_colorbar_ticks), step=0.1,
                                                                 key=f"{fkey}sim_colorbar_ticks")
                option["ref_colorbar_ticks"] = col2.number_input(f"Reference", value=float(ref_colorbar_ticks), step=0.1,
                                                                 key=f"{fkey}ref_colorbar_ticks")
                option["diff_colorbar_ticks"] = col3.number_input(f"Difference", value=float(diff_colorbar_ticks), step=0.1,
                                                                  key=f"{fkey}diff_colorbar_ticks")
            elif option['plot_type'] == 'Simulation' or option['plot_type'] == 'Reference':
                vmin_max_on = False
                error = False
                try:
                    if option['plot_type'] == 'Simulation':
                        var = sim_var
                        filename = f'{option["data_path"]}/{selected_item}_sim_{simselect}_{sim_var}.nc'
                        if len(option['title']) == 0:
                            option['title'] = 'Simulation'

                        ds = xr.open_dataset(filename)
                        ds = ds[var].mean('time', skipna=True)
                        vmin = math.floor(np.nanmin(ds))
                        vmax = math.floor(np.nanmax(ds))
                    elif option['plot_type'] == 'Reference':
                        var = ref_var
                        filename = f'{option["data_path"]}/{selected_item}_ref_{refselect}_{ref_var}.nc'
                        if len(option['title']) == 0:
                            option['title'] = 'Reference'
                        ds = xr.open_dataset(filename)

                        ds = ds[var].mean('time', skipna=True)
                        vmin = math.floor(np.nanmin(ds))
                        vmax = math.floor(np.nanmax(ds))
                except Exception as e:
                    st.error(f"Error: {e}")
                    error = True

                if not error:
                    col1, col2, col3 = st.columns(3)
                    option["vmin_max_on"] = col1.toggle('Setting max min', value=vmin_max_on, key=f"{fkey}vmin_max_on")
                    if option["vmin_max_on"]:
                        try:
                            option["vmin"] = col2.number_input(f"colorbar min", value=vmin, key=f"{fkey}vmin")
                            option["vmax"] = col3.number_input(f"colorbar max", value=vmax, key=f"{fkey}vmax")
                        except ValueError:
                            st.error(f"Max value must larger than min value.")
                    else:
                        option["vmin"] = vmin
                        option["vmax"] = vmax
                    colorbar_ticks = get_ticks(option["vmin"], option["vmax"])
                else:
                    colorbar_ticks = 0.5
            st.divider()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                option['cpool'] = st.selectbox('Colorbar',
                                               ['RdYlGn', 'RdYlGn_r', 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG',
                                                'BrBG_r', 'BuGn', 'BuGn_r',
                                                'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
                                                'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
                                                'Oranges',
                                                'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
                                                'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                                                'PuBu_r',
                                                'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
                                                'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
                                                'RdYlGn_r',
                                                'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                                                'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
                                                'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
                                                'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r',
                                                'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm',
                                                'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag',
                                                'flag_r',
                                                'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                                'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                                'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
                                                'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray',
                                                'gray_r',
                                                'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet',
                                                'jet_r',
                                                'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
                                                'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                                'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                                'summer_r',
                                                'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
                                                'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight',
                                                'twilight_r',
                                                'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter',
                                                'winter_r'], index=0, placeholder="Choose an option",
                                               label_visibility="visible", key=f"{fkey}cpool")
            with col2:
                option["colorbar_position"] = st.selectbox('colorbar position', ['horizontal', 'vertical'],  # 'Season',
                                                           index=0, placeholder="Choose an option",
                                                           label_visibility="visible", key=f"{fkey}colorbar_position")

            with col3:
                option["extend"] = st.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                                index=0, placeholder="Choose an option", label_visibility="visible",
                                                key=f"geo_time_average_extend")
            if option['plot_type'] == 'Simulation' or option['plot_type'] == 'Reference':
                with col4:
                    option["colorbar_ticks"] = st.number_input(f"Colorbar Ticks locater", value=float(colorbar_ticks), step=0.1,
                                                               key=f"{fkey}colorbar_ticks")
            option["map"] = col1.selectbox(f"Draw map", ['imshow', 'contourf'],
                                           index=0, placeholder="Choose an option", label_visibility="visible",
                                           key=f"{fkey}map")
            option["xtick"] = col2.number_input(f"Set x tick scale", value=60, min_value=0, max_value=360, step=10,
                                                key=f"{fkey}xtick")
            option["ytick"] = col3.number_input(f"Set y tick scale", value=30, min_value=0, max_value=180, step=10,
                                                key=f"{fkey}ytick")

        col1, col2, col3 = st.columns(3)
        option["x_wise"] = col1.number_input(f"X Length", min_value=0, value=14, key=f"{fkey}x_wise")
        option["y_wise"] = col2.number_input(f"y Length", min_value=0, value=7, key=f"{fkey}y_wise")
        option['saving_format'] = col3.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                 index=1, placeholder="Choose an option", label_visibility="visible",
                                                 key=f"{fkey}saving_format")
        option['font'] = col1.selectbox('Image saving format',
                                        ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                         'Helvetica',
                                         'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                        index=0, placeholder="Choose an option", label_visibility="visible",
                                        key=f"{fkey}font")
        if option['plot_type'] == 'Differentiate':
            option["hspace"] = col2.number_input(f"hspace", min_value=0., max_value=1.0, value=0.5, step=0.1, key=f"{fkey}hspace")
            option["wspace"] = col3.number_input(f"wspace", min_value=0., max_value=1.0, value=0.25, step=0.1,
                                                 key=f"{fkey}wspace")

    if option['plot_type'] == 'Simulation' or option['plot_type'] == 'Reference':
        geo_single_average(option, selected_item, refselect, simselect, ref, sim, var, filename)
    elif option['plot_type'] == 'Differentiate':
        geo_average_diff(option, selected_item, refselect, simselect, ref, sim, ref_var, sim_var)
    else:
        st.error('please choose first!')
