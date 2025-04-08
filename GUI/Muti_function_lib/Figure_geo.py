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


def get_ticks(vmin, vmax):
    if 2 >= vmax - vmin > 1:
        colorbar_ticks = 0.2
    elif 5 >= vmax - vmin > 2:
        colorbar_ticks = 0.5
    elif 10 >= vmax - vmin > 5:
        colorbar_ticks = 1.
    elif 100 >= vmax - vmin > 10:
        colorbar_ticks = 5.
    elif 100 >= vmax - vmin > 50:
        colorbar_ticks = 20.
    elif 200 >= vmax - vmin > 100:
        colorbar_ticks = 20.
    elif 500 >= vmax - vmin > 200:
        colorbar_ticks = 50.
    elif 1000 >= vmax - vmin > 500:
        colorbar_ticks = 100.
    elif 2000 >= vmax - vmin > 1000:
        colorbar_ticks = 200.
    elif 10000 >= vmax - vmin > 2000:
        colorbar_ticks = 10 ** math.floor(math.log10(vmax - vmin)) / 2
    else:
        colorbar_ticks = 0.10
    return colorbar_ticks


def get_index(vmin, vmax, colormap, option):
    if not option['vmin_max_on']:
        colorbar_ticks = get_ticks(vmin, vmax)
    else:
        colorbar_ticks = option['colorbar_ticks']
    ticks = matplotlib.ticker.MultipleLocator(base=colorbar_ticks)
    mticks = ticks.tick_values(vmin=vmin, vmax=vmax)
    mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in
              mticks]
    if mticks[0] < vmin and mticks[-1] < vmax:
        mticks = mticks[1:]
    elif mticks[0] > vmin and mticks[-1] > vmax:
        mticks = mticks[:-1]
    elif mticks[0] < vmin and mticks[-1] > vmax:
        mticks = mticks[1:-1]

    cmap = cm.get_cmap(colormap)
    bnd = np.arange(vmin, vmax + colorbar_ticks / 2, colorbar_ticks / 2)

    try:
        norm = colors.BoundaryNorm(bnd, cmap.N)
    except:
        if cmap.N < len(bnd):
            bnd = np.linspace(vmin, vmax + colorbar_ticks / 2, 15)
            norm = colors.BoundaryNorm(bnd, cmap.N)
    return mticks, norm, bnd, cmap


def prepare(selected_item, ref_var, sim_var, refselect, simselect, path, plot_type, function):
    ds_sim = xr.open_dataset(f'{path}/{selected_item}_sim_{simselect}_{sim_var}.nc')[sim_var]
    ds_ref = xr.open_dataset(f'{path}/{selected_item}_ref_{refselect}_{ref_var}.nc')[ref_var]
    data_sim = ds_sim.mean('time', skipna=True)
    data_ref = ds_ref.mean('time', skipna=True)
    diff = data_ref - data_sim
    ilat = ds_ref.lat.values
    ilon = ds_ref.lon.values
    if function == 'min_max':
        if plot_type == 'Differentiate':
            diff_vmin, diff_vmax = math.floor(np.nanmin(diff)), math.floor(np.nanmax(diff))
            return get_ticks(diff_vmin, diff_vmax), diff_vmin, diff_vmax
        elif plot_type == 'Simulation':
            sim_vmin, sim_vmax = math.floor(np.nanmin(ds_sim)), math.floor(np.nanmax(ds_sim))
            return get_ticks(sim_vmin, sim_vmax), sim_vmin, sim_vmax
        elif plot_type == 'Reference':
            ref_vmin, ref_vmax = math.floor(np.nanmin(ds_ref)), math.floor(np.nanmax(ds_ref))
            return get_ticks(ref_vmin, ref_vmax), ref_vmin, ref_vmax
    else:
        if plot_type == 'Differentiate':
            return ilat, ilon, diff
        elif plot_type == 'Simulation':
            return ilat, ilon, data_sim
        elif plot_type == 'Reference':
            return ilat, ilon, data_ref


def geo_Showing(Figure_show, option: dict, selected_item, refselect, simselect, vars, path):
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

    ilat, ilon, data = prepare(selected_item, vars[0], vars[1], refselect, simselect, path, option['plot_type'], 'values')

    fig, ax = plt.subplots(1, figsize=(option['x_wise'], option['y_wise']), subplot_kw={'projection': ccrs.PlateCarree()})

    mticks, norm, bnd, cmap = get_index(option['vmin'], option['vmax'], option['cpool'], option)

    if ilat[0] - ilat[-1] < 0:
        option['origin'] = 'lower'
    else:
        option['origin'] = 'upper'

    if option['map'] == 'interpolate':
        lat, lon = np.meshgrid(ilat[::-1], ilon)
        cs = ax.contourf(lon, lat, data.transpose("lon", "lat")[:, ::-1].values, levels=bnd, alpha=1, cmap=cmap, norm=norm,
                         extend=option["extend"])
    else:
        extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])
        cs = ax.imshow(data, cmap=cmap, vmin=option['vmin'], vmax=option['vmax'], extent=extent,
                       origin=option['origin'])

    ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']], crs=ccrs.PlateCarree())
    ax.set_adjustable('datalim')  # 固定数据范围，不自动扩展
    ax.set_aspect('equal', adjustable='box')  # 保持数据比例

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

    ax.set_title(option['title'], fontsize=option['title_size'])
    ax.set_xlabel(option['xticklabel'], fontsize=option['xticksize'] + 1)  # labelpad=xoffsets
    ax.set_ylabel(option['yticklabel'], fontsize=option['yticksize'] + 1)

    ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -1 * option["xtick"])[::-1], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -1 * option["ytick"])[::-1], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

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

    with Figure_show:
        st.pyplot(fig)

    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=300)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{option["plot_type"]}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)


def make_geo_time_average(selected_item, refselect, simselect, path, ref, sim):

    option = {}
    fkey = 'geo_time_average_'

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
    Figure_show = st.container()
    Labels_tab, Scale_tab, Map_tab, Save_tab = st.tabs(['Labels', 'Scale', 'Map', 'Save'])

    with Labels_tab:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if option['plot_type'] == 'Differentiate':
                title = f"{simselect} vs {refselect}"
            elif option['plot_type']  == 'Simulation':
                title = simselect
            elif option['plot_type']  == 'Reference':
                title = refselect
            else:
                title=""
            option['title'] = st.text_input('Title', value=title,label_visibility="visible", key=f"{fkey}title")
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

    with Map_tab:
        col1, col2, col3, col4 = st.columns(4)
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
        option["map"] = col1.selectbox(f"Draw map", ['None', 'interpolate'],
                                       index=0, placeholder="Choose an option", label_visibility="visible",
                                       key=f"{fkey}map")
        option["xtick"] = col2.number_input(f"Set x tick scale", value=60, min_value=0, max_value=360, step=10,
                                            key=f"{fkey}xtick")
        option["ytick"] = col3.number_input(f"Set y tick scale", value=30, min_value=0, max_value=180, step=10,
                                            key=f"{fkey}ytick")

    with Scale_tab:
        col1, col2, col3 = st.columns((1.5, 1, 1))
        option['grid'] = col1.toggle("Showing grid?", value=False, label_visibility="visible", key=f"{fkey}grid")
        if option['grid']:
            option['grid_style'] = col2.selectbox('Grid Line Style', ['solid', 'dotted', 'dashed', 'dashdot'],
                                                  index=2, placeholder="Choose an option", label_visibility="visible",
                                                  key=f"{fkey}grid_style")
            option['grid_linewidth'] = col3.number_input("grid linewidth", min_value=0, value=1, key=f"{fkey}grid_linewidth")

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

        ref_var = ref[refselect][selected_item][f"varname"]
        sim_var = sim[simselect][selected_item][f"varname"]

        if option['plot_type'] is not None:
            colorbar_ticks, vmin, vmax = prepare(selected_item, ref_var, sim_var, refselect, simselect, path, option['plot_type'], 'min_max')
            col1, col2, col3, col4 = st.columns(4)
            option["vmin_max_on"] = col1.toggle('Fit to Data', value=False, key=f"{fkey}vmin_max_on")
            if option["vmin_max_on"]:
                option["colorbar_ticks"] = col2.number_input(f"Colorbar Ticks locater", value=float(colorbar_ticks), step=0.1,
                                                             key=f"{fkey}colorbar_ticks")
                option["vmin"] = col3.number_input(f"colorbar min", value=vmin, key=f"{fkey}sim_vmin")
                option["vmax"] = col4.number_input(f"colorbar max", value=vmax, key=f"{fkey}sim_vmax")
            else:
                option["vmin"] = vmin
                option["vmax"] = vmax
                option["colorbar_ticks"] = colorbar_ticks

    with Save_tab:
        col1, col2, col3 = st.columns(3)
        option["x_wise"] = col1.number_input(f"X Length", min_value=0, value=12, key=f"{fkey}x_wise")
        option["y_wise"] = col2.number_input(f"y Length", min_value=0, value=6, key=f"{fkey}y_wise")
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

    st.divider()
    if option['plot_type']:
        geo_Showing(Figure_show, option, selected_item, refselect, simselect, (ref_var, sim_var), path)
    else:
        st.error('please choose first!')
