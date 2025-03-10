import math
import os
from io import BytesIO
import pandas as pd
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


def process_unit(ref_unit, metric):
    all_metrics_units = {
        'percent_bias': '%',  # Percent Bias
        'absolute_percent_bias': '%',  # Absolute Percent Bias
        'bias': 'Same as input data',  # Bias
        'mean_absolute_error': 'Same as input data',  # Mean Absolute Error
        'RMSE': 'Same as input data',  # Root Mean Squared Error
        'MSE': 'Square of input data unit',  # Mean Squared Error
        'ubRMSE': 'Same as input data',  # Unbiased Root Mean Squared Error
        'CRMSD': 'Same as input data',  # Centered Root Mean Square Difference
        'nrmse': 'Unitless',  # Normalized Root Mean Square Error
        'L': 'Unitless',  # Likelihood
        'correlation': 'Unitless',  # correlation coefficient
        'correlation_R2': 'Unitless',  # correlation coefficient R2
        'NSE': 'Unitless',  # Nash Sutcliffe efficiency coefficient
        'LNSE': 'Unitless',  # natural logarithm of NSE coefficient
        'KGE': 'Unitless',  # Kling-Gupta Efficiency
        'KGESS': 'Unitless',  # Normalized Kling-Gupta Efficiency
        'kappa_coeff': 'Unitless',  # Kappa coefficient
        'rv': 'Unitless',  # Relative variability (amplitude ratio)
        'ubNSE': 'Unitless',  # Unbiased Nash Sutcliffe efficiency coefficient
        'ubKGE': 'Unitless',  # Unbiased Kling-Gupta Efficiency
        'ubcorrelation': 'Unitless',  # Unbiased correlation
        'ubcorrelation_R2': 'Unitless',  # correlation coefficient R2
        'pc_max': '%',  # the bias of the maximum value
        'pc_min': '%',  # the bias of the minimum value
        'pc_ampli': '%',  # the bias of the amplitude value
        'rSD': 'Unitless',  # Ratio of standard deviations
        'PBIAS_HF': '%',  # Percent bias of flows ≥ Q98 (Yilmaz et al., 2008)
        'PBIAS_LF': '%',  # Percent bias of flows ≤ Q30(Yilmaz et al., 2008)
        'SMPI': 'Unitless',  # https://docs.esmvaltool.org/en/latest/recipes/recipe_smpi.html
        'ggof': 'Unitless',  # Graphical Goodness of Fit
        'gof': 'Unitless',  # Numerical Goodness-of-fit measures
        'KGEkm': 'Unitless',  # Kling-Gupta Efficiency with knowable-moments
        'KGElf': 'Unitless',  # Kling-Gupta Efficiency for low values
        'KGEnp': 'Unitless',  # Non-parametric version of the Kling-Gupta Efficiency
        'md': 'Unitless',  # Modified Index of Agreement
        'mNSE': 'Unitless',  # Modified Nash-Sutcliffe efficiency
        'pbiasfdc': '%',  # Percent Bias in the Slope of the Midsegment of the Flow Duration Curve
        'pfactor': '%',  # the percent of observations that are within the given uncertainty bounds.
        'rd': 'Unitless',  # Relative Index of Agreement
        'rfactor': 'Unitless',
        # the average width of the given uncertainty bounds divided by the standard deviation of the observations.
        'rNSE': 'Unitless',  # Relative Nash-Sutcliffe efficiency
        'rSpearman': 'Unitless',  # Spearman's rank correlation coefficient
        'rsr': 'Unitless',  # Ratio of RMSE to the standard deviation of the observations
        'sKGE': 'Unitless',  # Split Kling-Gupta Efficiency
        'ssq': 'Square of input data unit',  # Sum of the Squared Residuals
        'valindex': 'Unitless',  # Valid Indexes
        've': 'Unitless',  # Volumetric Efficiency
        'wNSE': 'Unitless',  # Weighted Nash-Sutcliffe efficiency
        'wsNSE': 'Unitless',  # Weighted seasonal Nash-Sutcliffe Efficiency
        'index_agreement': 'Unitless',  # Index of agreement
    }
    if metric not in all_metrics_units.keys():
        return '[None]'
    else:
        unit = all_metrics_units[metric]
        if unit == 'Unitless':
            return '[Unitless]'
        elif unit == '%':
            return '[%]'
        elif unit == 'Same as input data':
            return f'[{ref_unit}]'
        elif unit == 'Square of input data unit':
            return rf'[${ref_unit}^{{2}}$]'
        else:
            print('Warning: Missing metric unit!')
            return '[None]'


def get_ticks(vmin, vmax):
    if 2 >= vmax - vmin > 1:
        colorbar_ticks = 0.2
    elif 5 >= vmax - vmin > 2:
        colorbar_ticks = 0.5
    elif 10 >= vmax - vmin > 5:
        colorbar_ticks = 1.
    elif 20 >= vmax - vmin > 10:
        colorbar_ticks = 5.
    elif 50 >= vmax - vmin > 20:
        colorbar_ticks = 10.
    elif 80 >= vmax - vmin > 50:
        colorbar_ticks = 15.
    elif 100 >= vmax - vmin > 80:
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


def get_index(vmin_max_on, vmin, vmax, colormap, cticks):
    if not vmin_max_on:
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
        bnd = np.arange(mticks[0], mticks[-1] + colorbar_ticks / 2, colorbar_ticks / 2)
        norm = colors.BoundaryNorm(bnd, cmap.N)
    else:
        ticks = matplotlib.ticker.MultipleLocator(base=cticks)
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
        bnd = np.arange(float(mticks[0]), float(mticks[-1]) + cticks / 2., cticks / 2.)
        norm = colors.BoundaryNorm(bnd, cmap.N)
    return mticks, norm, bnd, cmap


def Stn_map(file, loc_lon, loc_lat, metric_data, option):
    from pylab import rcParams
    import matplotlib
    import matplotlib.pyplot as plt

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

    ax.set_extent([option['min_lon'], option['max_lon'], option['min_lat'], option['max_lat']], crs=ccrs.PlateCarree())

    mticks, norm, bnd, cmap = get_index(option['vmin_max_on'], option['vmin'], option['vmax'], option['cmap'],
                                        option['colorbar_ticks'])

    cs = ax.scatter(loc_lon, loc_lat, s=option['markersize'], c=metric_data, cmap=cmap, vmin=option['vmin'], vmax=option['vmax'],
                    marker=option['marker'],
                    edgecolors='none', alpha=0.9)

    coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '50m', edgecolor='0.6', facecolor='none')
    rivers = cfeature.NaturalEarthFeature(
        'physical', 'rivers_lake_centerlines', '110m', edgecolor='0.6', facecolor='none')
    ax.add_feature(cfeature.LAND, facecolor='0.8')
    ax.add_feature(coastline, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=1, facecolor='white', edgecolor='white')
    ax.add_feature(rivers, linewidth=0.5)

    ax.gridlines(draw_labels=False, linestyle=':', linewidth=0.5, color='grey', alpha=0.8)
    ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -1 * option["xtick"])[::-1], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -1 * option["ytick"])[::-1], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_xlabel(option['xlabel'], fontsize=option['xticksize'] + 1)
    ax.set_ylabel(option['ylabel'], fontsize=option['yticksize'] + 1)
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
    cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, spacing='uniform', label=option['colorbar_label'],
                      orientation=option['colorbar_position'], extend=option['extend'])

    cb.solids.set_edgecolor("face")

    file2 = os.path.basename(file)[:-3]
    st.pyplot(fig)

    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False, key=file2)


def draw_Diff_Stn_Plot(file, var, option):  # outpath, source
    data = pd.read_csv(f'{file}', header=0)
    lon_select = data['lon'].values
    lat_select = data['lat'].values
    plotvar = data[var].values
    Stn_map(file, lon_select, lat_select, plotvar, option)


def prepare_stn(dir_path, icase, file, selected_item, score, ref_source, sim_source, showing_format, ref_unit, option):
    with st.container(height=None, border=True):
        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            if showing_format == 'Anomaly':
                title = f"{selected_item.replace('_', ' ')} {score} anomaly for {sim_source}"
            else:
                title = f"{selected_item.replace('_', ' ')} {score} difference {sim_source[0]} vs {sim_source[1]}"
            option['title'] = st.text_input('Title', value=title, label_visibility="visible",
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
            option["min_lon"] = col1.number_input(f"minimal longitude", value=st.session_state['generals']['min_lon'],
                                                  key=f"{icase}_min_lon")
            option["max_lon"] = col2.number_input(f"maximum longitude", value=st.session_state['generals']['max_lon'],
                                                  key=f"{icase}_max_lon")
            option["min_lat"] = col3.number_input(f"minimal latitude", value=st.session_state['generals']['min_lat'],
                                                  key=f"{icase}_min_lat")
            option["max_lat"] = col4.number_input(f"maximum latitude", value=st.session_state['generals']['max_lat'],
                                                  key=f"{icase}_max_lat")
            option["xtick"] = col1.number_input(f"Set x tick scale", value=60, min_value=0, max_value=360, step=10,key=f"{icase}xtick")
            option["ytick"] = col2.number_input(f"Set y tick scale", value=30, min_value=0, max_value=180, step=10,key=f"{icase}ytick")

            col1, col2, col3, col4 = st.columns(4)
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
                                                         value=f'{score} {process_unit(ref_unit, score)}',
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

            option['marker'] = col1.selectbox(f'Marker style',
                                              ['.', 'x', 'o', ">", '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', "+",
                                               "^", "v"],
                                              index=2,
                                              placeholder="Choose an option",
                                              label_visibility="visible")

            option['markersize'] = col2.number_input(f"Markersize", min_value=0, value=15, step=1)

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

            df = pd.read_csv(f'{dir_path}/{file}', header=0)
            var = f'{score}_{"anomaly" if showing_format.lower() == "anomaly" else "diff"}'
            data = df[var].values
            if score in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF', 'NSE', 'KGE', 'KGESS', 'correlation',
                         'kappa_coeff',
                         'rSpearman']:
                quantiles = [np.nanpercentile(data, 5), np.nanpercentile(data, 95)]
                max_value = math.ceil(quantiles[1])
                min_value = math.floor(quantiles[0])
                if score == 'percent_bias':
                    if max_value > 100:
                        max_value = 100.
                    if min_value < -100:
                        min_value = -100.
            else:
                min_value, max_value = math.floor(np.nanmin(data)), math.ceil(np.nanmax(data))

            if option["vmin_max_on"]:
                option["colorbar_ticks"] = col2.number_input(f"Colorbar Ticks locater", value=get_ticks(min_value, max_value),
                                                             step=get_ticks(min_value, max_value) / 10,
                                                             key=f"{icase}_colorbar_ticks")
                option["vmin"] = col3.number_input(f"colorbar min", value=min_value)
                option["vmax"] = col4.number_input(f"colorbar max", value=max_value)
            else:
                option["colorbar_ticks"] = get_ticks(min_value, max_value)
                option["vmin"] = min_value
                option["vmax"] = max_value

        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10, key=f"{icase}_x_wise")
            option["map"] = st.selectbox(f"Draw map", ['None', 'interpolate'],
                                         index=0, placeholder="Choose an option", label_visibility="visible",
                                         key=f"{icase}_map")

        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6, key=f"{icase}_y_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{icase}_saving_format")

        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{icase}_dpi")
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica', 'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f"{icase}_font")
    draw_Diff_Stn_Plot(str(os.path.join(dir_path, file)), var, option)


def Geo_map(file, ilon, ilat, data, option):
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

    mticks, norm, bnd, cmap = get_index(option['vmin_max_on'], option['vmin'], option['vmax'], option['cmap'],
                                        option['colorbar_ticks'])

    if option['map'] == 'interpolate':
        lon, lat = np.meshgrid(ilon, ilat)
        cs = ax.contourf(lon, lat, data, cmap=cmap, levels=bnd, norm=norm, extend=option['extend'])
    else:
        extent = (ilon[0], ilon[-1], ilat[0], ilat[-1])
        cs = ax.imshow(data, cmap=cmap, vmin=option['vmin'], vmax=option['vmax'], extent=extent,
                       origin=option['origin'])

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
    ax.set_xticks(np.arange(option['max_lon'], option['min_lon'], -1 * option["xtick"])[::-1], crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(option['max_lat'], option['min_lat'], -1 * option["ytick"])[::-1], crs=ccrs.PlateCarree())
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
    cb = fig.colorbar(cs, cax=cbaxes, ticks=mticks, spacing='uniform', label=option['colorbar_label'], extend=option['extend'],
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


def draw_Diff_Geo_Plot(file, var, option):  # outpath, source

    ds = xr.open_dataset(file)
    data = ds[var]

    ilat = ds.lat.values
    ilon = ds.lon.values
    if ilat[0] - ilat[-1] < 0:
        option['origin'] = 'lower'
    else:
        option['origin'] = 'upper'
    Geo_map(file, ilon, ilat, data, option)


def prepare_geo(dir_path, icase, file, selected_item, score, ref_source, sim_source, showing_format, ref_unit, option):
    with st.container(height=None, border=True):
        col1, col2, col3, col4 = st.columns((3.5, 3, 3, 3))
        with col1:
            if showing_format == 'Anomaly':
                title = f"{selected_item.replace('_', ' ')} {score} anomaly for {sim_source}"
            else:
                title = f"{selected_item.replace('_', ' ')} {score} difference {sim_source[0]} vs {sim_source[1]}"
            option['title'] = st.text_input('Title', value=title, label_visibility="visible",
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
            option["min_lon"] = col1.number_input(f"minimal longitude", value=st.session_state['generals']['min_lon'],
                                                  key=f"{icase}_min_lon")
            option["max_lon"] = col2.number_input(f"maximum longitude", value=st.session_state['generals']['max_lon'],
                                                  key=f"{icase}_max_lon")
            option["min_lat"] = col3.number_input(f"minimal latitude", value=st.session_state['generals']['min_lat'],
                                                  key=f"{icase}_min_lat")
            option["max_lat"] = col4.number_input(f"maximum latitude", value=st.session_state['generals']['max_lat'],
                                                  key=f"{icase}_max_lat")
            option["xtick"] = col1.number_input(f"Set x tick scale", value=60, min_value=0, max_value=360, step=10,key=f"{icase}xtick")
            option["ytick"] = col2.number_input(f"Set y tick scale", value=30, min_value=0, max_value=180, step=10,key=f"{icase}ytick")

            col1, col2, col3, col4 = st.columns(4)
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
                                                         value=f'{score} {process_unit(ref_unit, score)}',
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

            ds = xr.open_dataset(str(os.path.join(dir_path, file)))
            var = f'{score}_{"anomaly" if showing_format.lower() == "anomaly" else "diff"}'
            data = ds[var]
            if score in ['bias', 'percent_bias', 'rSD', 'PBIAS_HF', 'PBIAS_LF', 'NSE', 'KGE', 'KGESS', 'correlation',
                         'kappa_coeff',
                         'rSpearman']:
                quantiles = data.quantile([0.05, 0.95], dim=['lat', 'lon'])
                max_value = math.ceil(quantiles[1].values)
                min_value = math.floor(quantiles[0].values)
                if score == 'percent_bias':
                    if max_value > 100:
                        max_value = 100.
                    if min_value < -100:
                        min_value = -100.
            else:
                min_value, max_value = math.floor(np.nanmin(data)), math.ceil(np.nanmax(data))
            # st.write(type(min_value), max_value)
            if option["vmin_max_on"]:
                option["colorbar_ticks"] = col2.number_input(f"Colorbar Ticks locater", value=get_ticks(min_value, max_value),
                                                             step=get_ticks(min_value, max_value) / 10,
                                                             key=f"{icase}_colorbar_ticks")
                option["vmin"] = col3.number_input(f"colorbar min", value=min_value)
                option["vmax"] = col4.number_input(f"colorbar max", value=max_value)
            else:
                option["colorbar_ticks"] = get_ticks(min_value, max_value)
                option["vmin"] = min_value
                option["vmax"] = max_value

        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=10, key=f"{icase}_x_wise")
            option["map"] = st.selectbox(f"Draw map", ['None', 'interpolate'],
                                         index=0, placeholder="Choose an option", label_visibility="visible",
                                         key=f"{icase}_map")

        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6, key=f"{icase}_y_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"{icase}_saving_format")

        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"{icase}_dpi")
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica', 'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f"{icase}_font")
    draw_Diff_Geo_Plot(str(os.path.join(dir_path, file)), var, option)


def make_scenarios_comparison_Diff_Plot(dir_path, file, selected_item, score, ref_source, sim_source, showing_format,
                                        ref_data_type, ref_unit, option):
    icase = f'Diff_Plot_{showing_format}_{ref_data_type}'
    if ref_data_type != 'stn':
        try:
            prepare_geo(dir_path, icase, file, selected_item, score, ref_source, sim_source, showing_format, ref_unit, option)
        except:
            st.error(f'Missing file for {file}')
    else:
        prepare_stn(dir_path, icase, file, selected_item, score, ref_source, sim_source, showing_format, ref_unit, option)
