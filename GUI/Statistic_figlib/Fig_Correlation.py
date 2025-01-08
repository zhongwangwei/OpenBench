import os
from io import BytesIO

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import cm
from matplotlib import colors
from matplotlib import rcParams


def get_index(vmin, vmax, colormap):
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

    colorbar_ticks = get_ticks(vmin, vmax)
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
    norm = colors.BoundaryNorm(bnd, cmap.N)
    return mticks, norm, bnd, cmap


def map(file, ilon, ilat, data, option):
    from Namelist_lib.check_font import check_font
    check = check_font()
    check.check_font(option['font'])

    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.labelsize': option['fontsize'],
              'grid.linewidth': 0.2,
              'font.size': option['fontsize'],
              'xtick.labelsize': option['xticksize'],
              'xtick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    fig = plt.figure(figsize=(option['x_wise'], option['y_wise']))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    mticks, norm, bnd, cmap = get_index(option['vmin'], option['vmax'], option['cmap'])

    if option["map"] == 'imshow':
        extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])
        cs = ax.imshow(data, cmap=cmap, vmin=option['vmin'], vmax=option['vmax'], extent=extent,
                       origin=option['origin'])
    elif option['map'] == 'contourf':
        lon, lat = np.meshgrid(ilon, ilat)
        cs = ax.contourf(lon, lat, data, cmap=cmap, levels=bnd, norm=norm, extend=option['extend'])

    coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '50m', edgecolor='0.6', facecolor='none')
    rivers = cfeature.NaturalEarthFeature(
        'physical', 'rivers_lake_centerlines', '110m', edgecolor='0.6', facecolor='none')
    ax.add_feature(cfeature.LAND, facecolor='0.9')
    ax.add_feature(coastline, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=1, facecolor='white', edgecolor='white')
    ax.add_feature(rivers, linewidth=0.5)
    ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.5, color='grey', alpha=0.8)

    ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']])
    ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -60)[::-1], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -30)[::-1], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xlabel(option['xlabel'], fontsize=option['xticksize'] + 1, labelpad=10)
    ax.set_ylabel(option['ylabel'], fontsize=option['yticksize'] + 1, labelpad=30)
    plt.title(option['title'], fontsize=option['title_size'])

    if not option['colorbar_position_set']:
        pos = ax.get_position()  # .bounds
        left, right, bottom, width, height = pos.x0, pos.x1, pos.y0, pos.width, pos.height
        if option['colorbar_position'] == 'horizontal':
            if len(option['xlabel']) == 0:
                cbaxes = fig.add_axes([left + width / 6, bottom - 0.12, width / 3 * 2, 0.04])
            else:
                cbaxes = fig.add_axes([left + width / 6, bottom - 0.17, width / 3 * 2, 0.04])
        else:
            cbaxes = fig.add_axes([right + 0.05, bottom, 0.03, height])
    else:
        cbaxes = fig.add_axes(
            [option["colorbar_left"], option["colorbar_bottom"], option["colorbar_width"], option["colorbar_height"]])
    cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, spacing='uniform', label=option['colorbar_label'],extend=option['extend'],
                      orientation=option['colorbar_position'])
    cb.solids.set_edgecolor("face")

    st.pyplot(fig)

    file2 = os.path.basename(file)[:-3]
    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)


def draw_Correlation(file, option):  # outpath, source

    ds = xr.open_dataset(file)
    data = ds['correlation']

    ilat = ds.lat.values
    ilon = ds.lon.values
    if ilat[0] - ilat[-1] < 0:
        option['origin'] = 'lower'
    else:
        option['origin'] = 'upper'
    map(file, ilon, ilat, data, option)


def prepare(icase, file, option):
    with st.container(height=None, border=True):
        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            option['title'] = st.text_input('Title', value=f'Correlation', label_visibility="visible",
                                            key=f"{icase}_title")
            option['title_size'] = st.number_input("Title label size", min_value=0, value=20, key=f"{icase}_title_size")

        with col2:
            option['xlabel'] = st.text_input('X labels', value='Longitude', label_visibility="visible",
                                             key=f"{icase}_xlabel")
            option['xticksize'] = st.number_input("Xtick label size", min_value=0, value=17, key=f"{icase}_xlabelsize")

        with col3:
            option['ylabel'] = st.text_input('Y labels', value='Latitude', label_visibility="visible",
                                             key=f"{icase}_ylabel")
            option['yticksize'] = st.number_input("Ytick label size", min_value=0, value=17, key=f"{icase}_ylabelsize")

        with col4:
            option['fontsize'] = st.number_input("Fontsize", min_value=0, value=17, key=f"{icase}_fontsize")
            option['axes_linewidth'] = st.number_input("axes linewidth", min_value=0, value=1,
                                                       key=f"{icase}_axes_linewidth")

        with st.expander("More info", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            # min_lon, max_lon, min_lat, max_lat
            option["min_lon"] = col1.number_input(f"minimal longitude", value=st.session_state['generals']['min_lon'],key=f"{icase}_min_lon")
            option["max_lon"] = col2.number_input(f"maximum longitude", value=st.session_state['generals']['max_lon'],key=f"{icase}_max_lon")
            option["min_lat"] = col3.number_input(f"minimal latitude", value=st.session_state['generals']['min_lat'],key=f"{icase}_min_lat")
            option["max_lat"] = col4.number_input(f"maximum latitude", value=st.session_state['generals']['max_lat'],key=f"{icase}_max_lat")

            with col1:
                option['cmap'] = st.selectbox('Colorbar',
                                              ['coolwarm', 'coolwarm_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
                                               'BuGn_r',
                                               'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu',
                                               'GnBu_r',
                                               'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
                                               'Oranges',
                                               'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
                                               'Pastel1_r',
                                               'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
                                               'PuBu_r',
                                               'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu',
                                               'RdBu_r',
                                               'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
                                               'RdYlGn_r',
                                               'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                                               'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
                                               'YlGnBu_r',
                                               'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot',
                                               'afmhot_r',
                                               'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg',
                                               'brg_r',
                                               'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'copper',
                                               'copper_r',
                                               'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
                                               'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey',
                                               'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
                                               'gist_rainbow_r', 'gray', 'gray_r',
                                               'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet',
                                               'jet_r',
                                               'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean',
                                               'ocean_r',
                                               'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
                                               'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
                                               'summer_r',
                                               'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter',
                                               'winter_r'], index=0, placeholder="Choose an option",
                                              key=f'{icase}_cmap',
                                              label_visibility="visible")
            with col2:
                option['colorbar_label'] = st.text_input('colorbar label',
                                                         key=f"{icase}_colorbar_label",
                                                         label_visibility="visible")
            with col3:
                option["colorbar_position"] = st.selectbox('colorbar position', ['horizontal', 'vertical'],
                                                           key=f"{icase}_colorbar_position",
                                                           index=0, placeholder="Choose an option",
                                                           label_visibility="visible")

            with col4:
                option["extend"] = st.selectbox(f"colorbar extend", ['neither', 'both', 'min', 'max'],
                                                index=0, placeholder="Choose an option", label_visibility="visible",
                                                key=f"{icase}_extend")
            if option["colorbar_position"] == 'vertical':
                left, bottom, right, top = 0.94, 0.24, 0.02, 0.5
            else:
                left, bottom, right, top = 0.26, 0.14, 0.5, 0.03

            st.divider()
            col1, col2, col3 = st.columns(3)
            option['colorbar_position_set'] = col1.toggle('Setting colorbar position', value=False,
                                                          key=f"{icase}_colorbar_position_set")
            if option['colorbar_position_set']:
                col1, col2, col3, col4 = st.columns(4)
                option["colorbar_left"] = col1.number_input(f"colorbar left", value=left)
                option["colorbar_bottom"] = col2.number_input(f"colorbar bottom", value=bottom)
                option["colorbar_width"] = col3.number_input(f"colorbar width", value=right)
                option["colorbar_height"] = col4.number_input(f"colorbar height", value=top)

            col1, col2, col3, col4 = st.columns(4)
            option["vmin_max_on"] = col1.toggle('Setting max min', value=False, key=f"{icase}_vmin_max_on")
            option["colorbar_ticks"] = col2.number_input(f"Colorbar Ticks locater", value=0.5, step=0.1,
                                                         key=f"{icase}_colorbar_ticks")

            if option["vmin_max_on"]:
                option["vmin"] = col3.number_input(f"colorbar min", value=-1.)
                option["vmax"] = col4.number_input(f"colorbar max", value=1.)
            else:
                option["vmin"] = -1.
                option["vmax"] = 1.

            st.divider()
            col1, col2, col3 = st.columns(3)
            option["map"] = col1.selectbox(f"Draw map", ['imshow', 'contourf'],
                                           index=0, placeholder="Choose an option", label_visibility="visible",
                                           key=f"{icase}_map")


        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10, key=f"{icase}_x_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{icase}_saving_format")
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6, key=f"{icase}_y_wise")
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica', 'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f"{icase}_font")

        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{icase}_dpi")

    # try:
    draw_Correlation(file, option)
    # except:
    #     st.error(f'Please check File: {file}')


def make_Correlation(dir_path, item, icase, file, item_data, option):
    st.write('make_Correlation')
    prepare(icase, file, option)