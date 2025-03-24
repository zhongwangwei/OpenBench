import warnings
from typing import Union
import os
import pandas as pd
import re
import xarray as xr
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rcParams
from matplotlib import ticker
import math
import matplotlib.colors as clr
import itertools

from io import BytesIO
import streamlit as st


def make_scenarios_comparison_Taylor_Diagram(self, dir_path, selected_item, ref_source):
    Figure_show = st.container()
    Labels_tab, Case_tab, Line_tab, Marker_tab, Save_tab = st.tabs(['Labels', 'Simulation', 'Lines', 'Markers', 'Save'])

    option = {}
    with Labels_tab:
        col1, col2, col3, col4 = st.columns((3, 3, 3, 3))
        option['title'] = col1.text_input('Title', value=f'{selected_item.replace("_", " ")}', label_visibility="visible",
                                          key=f"taylor_title")
        option['title_size'] = col2.number_input("Title label size", min_value=0, value=18, key=f"taylor_title_size")
        option['axes_linewidth'] = col3.number_input("axes linewidth", min_value=0, value=1, key=f"taylor_axes_linewidth")
        option['fontsize'] = col4.number_input("font size", min_value=0, value=16, key=f"taylor_fontsize")

        option['STDlabelsize'] = col1.number_input("STD label size", min_value=0, value=16, key=f"taylor_STD_size")
        option['CORlabelsize'] = col2.number_input("COR label size", min_value=0, value=16, key=f"taylor_COR_size")
        option['RMSlabelsize'] = col3.number_input("RMS label size", min_value=0, value=16, key=f"taylor_RMS_size")
        option['Normalized'] = col4.toggle('Normalized', value=True, key=f"taylor_Normalized")

    def get_cases(items, title):
            case_item = {}
            for item in items:
                case_item[item] = True
            import itertools
            color = '#9DA79A'
            st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; color:{color}; border-bottom:3px solid {color}; padding: 5px;">
                 Showing {title}....
            </div>
            """, unsafe_allow_html=True)
            st.write('')
            cols = itertools.cycle(st.columns(2))
            for item in case_item:
                col = next(cols)
                case_item[item] = col.checkbox(item, key=f'{item}__taylor',
                                               value=case_item[item])
            return [item for item, value in case_item.items() if value]

    with Case_tab:
        sim_sources = self.sim['general'][f'{selected_item}_sim_source']
        sim_sources = get_cases(sim_sources, 'cases')

    with Line_tab:
        col1, col2, col3, col4 = st.columns((3, 3, 3, 3))

        col1.write('##### :blue[Ticksize]')
        col2.write('##### :blue[Line style]')
        col3.write('##### :blue[Line width]')
        col4.write('##### :blue[Line color]')

        with col1:
            option['ticksizeSTD'] = st.number_input("STD", min_value=0, value=14, step=1, key=f"taylor_ticksizeSTD",
                                                    label_visibility='visible')
            option['ticksizeCOR'] = st.number_input("R", min_value=0, value=14, step=1, key=f"taylor_ticksizeCOR",
                                                    label_visibility='visible')
            option['ticksizeRMS'] = st.number_input("RMSD", min_value=0, value=14, step=1,
                                                    key=f"taylor_ticksizeRMS", label_visibility='visible')
            option['markersizeobs'] = st.number_input("Observation marker size", min_value=0, value=10, step=1,
                                                      key=f"taylor_markersizeobs")
            option['markerobs'] = st.selectbox(f'Observation Marker',
                                               ['.', 'x', 'o', ">", '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
                                                "+", "^", "v"], index=2, placeholder="Choose an option",
                                               label_visibility="visible")

        with col2:
            option['styleSTD'] = st.selectbox(f'STD', ['solid', 'dotted', 'dashed', 'dashdot', ':', '-', '--'],
                                              index=5, placeholder="Choose an option",
                                              key=f"taylor_styleSTD", label_visibility='visible')
            option['styleCOR'] = st.selectbox(f'R', ['solid', 'dotted', 'dashed', 'dashdot', ':', '-', '--'],
                                              index=6, placeholder="Choose an option", label_visibility='visible')
            option['styleRMS'] = st.selectbox(f'RMSD',
                                              ['solid', 'dotted', 'dashed', 'dashdot', ':', '-', '--'],
                                              index=4, placeholder="Choose an option", label_visibility='visible')
            option['styleOBS'] = st.selectbox(f'Observation',
                                              ['solid', 'dotted', 'dashed', 'dashdot', ':', '-', '--'],
                                              index=5, placeholder="Choose an option",
                                              label_visibility="visible")
        with col3:
            option['widthSTD'] = st.number_input("STD", min_value=0., value=1., key=f"taylor_widthSTD",
                                                 label_visibility='visible')
            option['widthCOR'] = st.number_input("R", min_value=0., value=1., key=f"taylor_widthCOR",
                                                 label_visibility='visible')
            option['widthRMS'] = st.number_input("RMSD", min_value=0., value=2., key=f"taylor_widthRMS",
                                                 label_visibility='visible')
            option['widthOBS'] = st.number_input("Observation", min_value=0., value=1.)

        with col4:
            option['colSTD'] = st.text_input("STD", value='k', label_visibility='visible', key=f"taylor_colSTD")
            option['colCOR'] = st.text_input("R ", value='k', label_visibility='visible', key=f"taylor_colCOR")
            option['colRMS'] = st.text_input("RMSD ", value='green', label_visibility='visible', key=f"taylor_colRMS")
            option['colOBS'] = st.text_input("Observation", value='m', label_visibility="visible")

    with Marker_tab:
        stds, RMSs, cors = [], [], []
        df = pd.read_csv(dir_path, sep=r'\s+', header=0)
        df.set_index('Item', inplace=True)

        stds.append(df['Reference_std'].values[0])
        RMSs.append(np.array(0))
        cors.append(np.array(0))

        import matplotlib.colors as mcolors
        from matplotlib import cm

        hex_colors = ['#4C6EF5', '#F9C74F', '#90BE6D', '#5BC0EB', '#43AA8B', '#F3722C', '#855456', '#F9AFAF',
                      '#F8961E', '#277DA1', '#5A189A']
        # hex_colors = cm.Set3(np.linspace(0, 1, len(self.sim['general'][f'{selected_item}_sim_source']) + 1))
        colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])
        symbols = itertools.cycle(["+", ".", "o", "*", "x", "s", "D", "^", "v", ">", "<", "p"])
        markers = {}
        col1, col2, col3, col4 = st.columns((1.8, 2, 2, 2))
        col1.write('##### :blue[Colors]')
        col2.write('##### :blue[Marker]')
        col3.write('##### :blue[Markersize]')
        col4.write('##### :blue[FaceColor]')

        for sim_source in sim_sources:
            st.write('Case: ', sim_source)
            col1, col2, col3, col4 = st.columns((1, 2, 2, 2))
            stds.append(df[f'{sim_source}_std'].values[0])
            RMSs.append(df[f'{sim_source}_RMS'].values[0])
            cors.append(df[f'{sim_source}_COR'].values[0])
            markers[sim_source] = {}
            with col1:
                markers[sim_source]['labelColor'] = st.color_picker(f'{sim_source}Colors', value=next(colors),
                                                                    key=None,
                                                                    help=None,
                                                                    on_change=None, args=None, kwargs=None,
                                                                    disabled=False,
                                                                    label_visibility="collapsed")
                markers[sim_source]['edgeColor'] = markers[sim_source]['labelColor']
            with col2:
                Marker = ['.', 'x', 'o', ">", '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', "+", "^", "v"]
                markers[sim_source]['symbol'] = st.selectbox(f'{sim_source}Marker', Marker,
                                                             index=Marker.index(next(symbols)),
                                                             placeholder="Choose an15 option",
                                                             label_visibility="collapsed")
            with col3:
                markers[sim_source]['size'] = st.number_input(f"{sim_source}Markersize", min_value=0, value=10,
                                                              label_visibility="collapsed")

            with col4:
                markers[sim_source]['faceColor'] = st.selectbox(f'{sim_source}FaceColor', ['w', 'b', 'k', 'r', 'none'],
                                                                index=0, placeholder="Choose an option",
                                                                label_visibility="collapsed")
        st.info('If you choose Facecolor as "none", then markers will not be padded.')
        option['MARKERS'] = markers

        legend_on = st.toggle('Turn on to set the location of the legend manually', value=False, key=f'taylor_legend_on')
        col1, col2, col3, col4 = st.columns((4, 4, 4, 4))
        if legend_on:
            if len(self.sim['general'][f'{selected_item}_sim_source']) < 6:
                bbox_to_anchor_x = col1.number_input("X position of legend", value=1.5, step=0.1,
                                                     key=f'taylor_bbox_to_anchor_x')
                bbox_to_anchor_y = col2.number_input("Y position of legend", value=1., step=0.1,
                                                     key=f'taylor_bbox_to_anchor_y')
            else:
                bbox_to_anchor_x = col1.number_input("X position of legend", value=1.1, step=0.1,
                                                     key=f'taylor_bbox_to_anchor_x')
                bbox_to_anchor_y = col2.number_input("Y position of legend", value=0.25, step=0.1,
                                                     key=f'taylor_bbox_to_anchor_y')
        else:
            bbox_to_anchor_x = 1.4
            bbox_to_anchor_y = 1.1
        option['set_legend'] = dict(legend_on=legend_on, bbox_to_anchor_x=bbox_to_anchor_x, bbox_to_anchor_y=bbox_to_anchor_y)

    with Save_tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            option["x_wise"] = st.number_input(f"X Length", min_value=0, value=6, key=f"taylor_x_wise")
            option['saving_format'] = st.selectbox('Image saving format', ['png', 'jpg', 'eps'],
                                                   index=1, placeholder="Choose an option", label_visibility="visible",
                                                   key=f"taylor_saving_format")
        with col2:
            option["y_wise"] = st.number_input(f"y Length", min_value=0, value=6, key=f"taylor_y_wise")
            option['font'] = st.selectbox('Image saving format',
                                          ['Times new roman', 'Arial', 'Courier New', 'Comic Sans MS', 'Verdana',
                                           'Helvetica',
                                           'Georgia', 'Tahoma', 'Trebuchet MS', 'Lucida Grande'],
                                          index=0, placeholder="Choose an option", label_visibility="visible",
                                          key=f"taylor_font")
        with col3:
            option['dpi'] = st.number_input(f"Figure dpi", min_value=0, value=300, key=f"taylor_dpi")
    st.divider()

    if option['Normalized']:
        std = np.array(stds) / df['Reference_std'].values[0]
        RMS = np.array(RMSs)
        cor = np.array(cors)
    else:
        std = np.array(stds)
        RMS = np.array(RMSs)
        cor = np.array(cors)
    draw_scenarios_comparison_Taylor_Diagram(Figure_show,option, selected_item, std, RMS, cor, ref_source)


def draw_scenarios_comparison_Taylor_Diagram(Figure_show,option, evaluation_item, stds, RMSs, cors, ref_source):
    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option['fontsize'],
              'xtick.direction': 'out',
              'ytick.direction': 'out',
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    fig, ax = plt.subplots(figsize=(option['x_wise'], option['y_wise']))

    taylor_diagram(stds, RMSs, cors, markers=option['MARKERS'],
                   titleRMS='on', markerLegend='on',
                   colRMS=option['colRMS'], colSTD=option['colSTD'], colCOR=option['colCOR'],
                   styleRMS=option['styleRMS'], styleSTD=option['styleSTD'], styleCOR=option['styleCOR'],
                   widthRMS=option['widthRMS'], widthSTD=option['widthSTD'], widthCOR=option['widthCOR'],
                   ticksizerms=option['ticksizeRMS'], ticksizeSTD=option['ticksizeSTD'], ticksizecor=option['ticksizeCOR'],
                   rmslabelsize=option['RMSlabelsize'], stdlabelsize=option['STDlabelsize'], corlabelsize=option['CORlabelsize'],
                   normalizedstd=option['Normalized'], set_legend=option['set_legend'],
                   styleOBS=option['styleOBS'], colOBS=option['colOBS'], widthOBS=option['widthOBS'],
                   markerobs=option['markerobs'], markersizeobs=option['markersizeobs'],
                   )

    ax.set_title(option['title'], fontsize=option['title_size'], pad=30)

    Figure_show.pyplot(fig)

    file2 = f'taylor_diagram_{evaluation_item}_{ref_source}'
    # 将图像保存到 BytesIO 对象
    buffer = BytesIO()
    fig.savefig(buffer, format=option['saving_format'], dpi=option['dpi'])
    buffer.seek(0)
    buffer.seek(0)
    st.download_button('Download image', buffer, file_name=f'{file2}.{option["saving_format"]}',
                       mime=f"image/{option['saving_format']}",
                       type="secondary", disabled=False, use_container_width=False)


def taylor_diagram(*args, **kwargs):
    '''
    ***   modified from SkillMetric  s*****
    Plot a Taylor diagram from statistics of different series.

    taylor_diagram(STDs,RMSs,CORs,keyword=value)

    The first 3 arguments must be the inputs as described below followed by
    keywords in the format OPTION = value. An example call to the function 
    would be:

    taylor_diagram(STDs,RMSs,CORs,markerdisplayed='marker')

    INPUTS:
    STDs: Standard deviations
    RMSs: Centered Root Mean Square Difference 
    CORs: Correlation

    Each of these inputs are one-dimensional with the same length. First
    index corresponds to the reference series for the diagram. For 
    example STDs[1] is the standard deviation of the reference series 
    and STDs[2:N] are the standard deviations of the other series. Note 
    that only the latter are plotted.

    Note that by definition the following relation must be true for all 
    series i:

    RMSs(i) = sqrt(STDs(i).^2 + STDs(1)^2 - 2*STDs(i)*STDs(1).*CORs(i))

    This relation is checked if the checkStats option is used, and if not 
    verified an error message is sent. This relation is not checked by
    default. Please see Taylor's JGR article for more informations about 
    this relation.

    OUTPUTS:
    None.

    Reference:

    Taylor, K. E. (2001), Summarizing multiple aspects of model 
    performance in a single diagram, J. Geophys. Res., 106(D7),
    7183-7192, doi:10.1029/2000JD900719.
    
    Author: Peter A. Rochford
            rochford.peter1@gmail.com

    Created on Dec 3, 2016
    Revised on Aug 23, 2022
    '''
    # Check for no arguments
    if len(args) == 0: return

    # Process arguments (if given)
    ax, STDs, RMSs, CORs = _get_taylor_diagram_arguments(*args)

    # Get options
    options = _get_taylor_diagram_options(CORs, **kwargs)

    # Check the input statistics if requested.
    _check_taylor_stats(STDs, RMSs, CORs, 0.01) if options['checkstats'] == 'on' else None

    # Express statistics in polar coordinates.
    rho, theta = STDs, np.arccos(CORs)

    #  Get axis values for plot
    axes = _get_taylor_diagram_axes(ax, rho, options)

    if options['overlay'] == 'off':
        # Draw circles about origin
        _overlay_taylor_diagram_circles(ax, axes, options)

        # Draw lines emanating from origin
        _overlay_taylor_diagram_lines(ax, axes, options)

        # Plot axes for Taylor diagram
        axes_handles = _plot_taylor_axes(ax, axes, options)

        # Plot marker on axis indicating observation STD
        _plot_taylor_obs(ax, axes_handles, STDs[0], axes, options)

        del axes_handles

    # Plot data points. Note that only rho[1:N] and theta[1:N] are 
    # plotted.
    X = np.multiply(rho[1:], np.cos(theta[1:]))
    Y = np.multiply(rho[1:], np.sin(theta[1:]))

    # Plot data points
    lowcase = options['markerdisplayed'].lower()
    if lowcase == 'marker':
        _plot_pattern_diagram_markers(ax, X, Y, options)
    elif lowcase == 'colorbar':
        nZdata = len(options['cmapzdata'])
        if nZdata == 0:
            # Use Centered Root Mean Square Difference for colors
            _plot_pattern_diagram_colorbar(ax, X, Y, RMSs[1:], options)
        else:
            # Use Bias values for colors
            _plot_pattern_diagram_colorbar(ax, X, Y, options['cmapzdata'][1:], options)
    else:
        raise ValueError('Unrecognized option: ' +
                         options['markerdisplayed'])

    return None


def _display_taylor_diagram_options() -> None:
    '''
    Displays available options for taylor_diagram_subplot() function.
    '''
    _disp('General options:')

    _dispopt("'alpha'", "Blending of symbol face color (0.0 transparent through 1.0 opaque)" +
             "\n\t\t" + "(Default: 1.0)")

    _dispopt("'axisMax'", 'Maximum for the radial contours')

    _dispopt("'colFrame'", "Color for both the y (left) and x (bottom) spines. " +
             "(Default: '#000000' (black))")

    _dispopt("'colorMap'", "'on'/ 'off' (default): " +
             "Switch to map color shading of markers to colormap ('on')\n\t\t" +
             "or min to max range of RMSDz values ('off').")

    _dispopt("'labelWeight'", "weight of the x & y axis labels")

    _dispopt("'numberPanels'", '1 or 2: Panels to display (1 for ' +
             'positive correlations, 2 for positive and negative' +
             ' correlations). \n\t\tDefault value depends on ' +
             'correlations (CORs)')

    _dispopt("'overlay'", "'on' / 'off' (default): " +
             'Switch to overlay current statistics on Taylor diagram. ' +
             '\n\t\tOnly markers will be displayed.')

    _disp("OPTIONS when 'colormap' == 'on'")
    _dispopt("'cmap'", "Choice of colormap. (Default: 'jet')")
    _dispopt("'cmap_marker'", "Marker to use with colormap (Default: 'd')")
    _dispopt("'cmap_vmax'", "Maximum range of colormap (Default: None)")
    _dispopt("'cmap_vmin'", "Minimum range of colormap (Default: None)")
    _disp('')

    _disp('Marker options:')

    _dispopt("'MarkerDisplayed'",
             "'marker' (default): Experiments are represented by individual " +
             "symbols\n\t\t" +
             "'colorBar': Experiments are represented by a color described " + \
             "in a colorbar")

    _disp("OPTIONS when 'MarkerDisplayed' == 'marker'")

    _dispopt("'markerColor'", 'Single color to use for all markers' +
             ' (Default: red)')

    _dispopt("'markerColors'", "Dictionary with up to two colors as keys ('face', 'edge') " +
             "to use for all markers " +
             "\n\t\twhen 'markerlegend' == 'off' or None." +
             "\n\t\tIf None or 'markerlegend' == 'on', then uses only the " +
             "value of 'markercolor'. (Default: None)")

    _dispopt("'markerLabel'", 'Labels for markers')

    _dispopt("'markerLabelColor'", 'Marker label color (Default: black)')

    _dispopt("'markerLayout'", "Matrix layout for markers in legend [nrow, ncolumn]." + "\n\t\t" +
             "(Default: [15, no. markers/15])'")

    _dispopt("'markerLegend'", "'on' / 'off' (default): " +
             'Use legend for markers')

    _dispopt("'markers'", 'Dictionary providing individual control of the marker ' +
             'label, label color, symbol, size, face color, and edge color' +
             ' (Default: None)')

    _dispopt("'markerSize'", 'Marker size (Default: 10)')

    _dispopt("'markerSymbol'", "Marker symbol (Default: '.')")

    _disp("OPTIONS when MarkerDisplayed' == 'colorbar'")

    _dispopt("'cmapzdata'", "Data values to use for color mapping of markers, " +
             "e.g. RMSD or BIAS." +
             "\n\t\t(Used to make range of RMSDs values appear above color bar.)")

    _dispopt("'locationColorBar'", "Location for the colorbar, 'NorthOutside' " +
             "or 'EastOutside'")

    _dispopt("'titleColorBar'", 'Title of the colorbar.')

    _disp('')

    _disp('RMS axis options:')

    _dispopt("'colRMS'", 'Color for RMS labels (Default: medium green)')

    _dispopt("'labelRMS'", "RMS axis label (Default 'RMSD')")

    _dispopt("'rincRMS'", 'Axis tick increment for RMS values')

    _dispopt("'rmsLabelFormat'", "String format for RMS contour labels, e.g. '0:.2f'.\n\t\t" +
             "(Default '0', format as specified by str function.)")

    _dispopt("'showlabelsRMS'", "'on' (default) / 'off': " +
             'Show the RMS tick labels')

    _dispopt("'styleRMS'", 'Line style of the RMS grid')

    _dispopt("'tickRMS'", 'RMS values to plot grid circles from ' +
             'observation point')

    _dispopt("'tickRMSangle'", 'Angle for RMS tick labels with the ' +
             'observation point. Default: 135 deg.')

    _dispopt("'titleRMS'", "'on' (default) / 'off': " +
             'Show RMSD axis title')

    _dispopt("'titleRMSDangle'", "angle at which to display the 'RMSD' label for the\n\t\t" +
             "RMSD contours (Default: 160 degrees)")

    _dispopt("'widthRMS'", 'Line width of the RMS grid')
    _disp('')

    _disp('STD axis options:')

    _dispopt("'colSTD'", 'STD grid and tick labels color. (Default: black)')

    _dispopt("'colsSTD'", "STD dictionary of grid colors with: " +
             "'grid', 'tick_labels', 'title' keys/values." +
             "\n\t\tIf not provided or None, considers the monotonic 'colSTD' argument. " +
             "(Default: None")  # subplot-specific

    _dispopt("'rincSTD'", 'axis tick increment for STD values')

    _dispopt("'showlabelsSTD'", "'on' (default) / 'off': " +
             'Show the STD tick labels')

    _dispopt("'styleSTD'", 'Line style of the STD grid')

    _dispopt("'tickSTD'", 'STD values to plot gridding circles from ' +
             'origin')

    _dispopt("'titleSTD'", "'on' (default) / 'off': " +
             'Show STD axis title')

    _dispopt("'widthSTD'", 'Line width of the STD grid')
    _disp('')

    _disp('CORRELATION axis options:')

    _dispopt("'colCOR'", 'CORRELATION grid color. Default: blue')

    _dispopt("'colsCOR'", "CORRELATION dictionary of grid colors with: " +
             "'grid', 'tick_labels', 'title' keys/values." +
             "\n\t\tIf not provided or None, considers the monotonic 'colCOR' argument." +
             "Default: None")  # subplot-specific

    _dispopt("'showlabelsCOR'", "'on' (default) / 'off': " +
             'Show the CORRELATION tick labels')

    _dispopt("'styleCOR'", 'Line style of the CORRELATION grid')

    _dispopt("'tickCOR[panel]'", "Tick values for correlation coefficients for " +
             "two types of panels")

    _dispopt("'titleCOR'", "'on' (default) / 'off': " +
             'Show CORRELATION axis title')

    _dispopt("'titleCORshape'", "The shape of the label 'correlation coefficient'. " +
             "\n\t\tAccepted values are 'curved' or 'linear' " +
             "(Default: 'curved'),")

    _dispopt("'widthCOR'", 'Line width of the COR grid')
    _disp('')

    _disp('Observation Point options:')

    _dispopt("'colObs'", "Observation STD color. (Default: magenta)")

    _dispopt("'markerObs'", "Marker to use for x-axis indicating observed STD." +
             "\n\t\tA choice of 'None' will suppress appearance of marker. (Default None)")

    _dispopt("'styleObs'", "Line style for observation grid line. A choice of empty string ('')\n\t\t" +
             "will suppress appearance of the grid line. (Default: '')")

    _dispopt("'titleOBS'", "Label for observation point (Default: '')")

    _dispopt("'widthOBS'", 'Line width for observation grid line')

    _disp('')

    _disp('CONTROL options:')

    _dispopt("'checkStats'", "'on' / 'off' (default): " +
             'Check input statistics satisfy Taylor relationship')

    _disp('Plotting Options from File:')

    _dispopt("'taylor_options_file'", "name of CSV file containing values for optional " +
             "arguments" +
             "\n\t\t" + "of the taylor_diagram function. If no file suffix is given," +
             "\n\t\t" + "a '.csv' is assumed. (Default: empty string '')")


def _ensure_np_array_or_die(v, label: str) -> np.ndarray:
    '''
    Check variable has is correct data type.
    
    v: Value to be ensured
    label: Python data type
    '''
    import numbers
    from array import array
    ret_v = v
    if isinstance(ret_v, array):
        ret_v = np.array(v)
    if isinstance(ret_v, numbers.Number):
        ret_v = np.array(v, ndmin=1)
    if not isinstance(ret_v, np.ndarray):
        raise ValueError('Argument {0} is not a numeric array: {1}'.format(label, v))
    return ret_v


def _get_taylor_diagram_arguments(*args):
    '''
    Get arguments for taylor_diagram function.
    
    Retrieves the arguments supplied to the TAYLOR_DIAGRAM function as
    arguments and displays the optional arguments if none are supplied.
    Otherwise, tests the first 3 arguments are numeric quantities and 
    returns their values.
    
    INPUTS:
    args: variable-length input argument list with size 4
    
    OUTPUTS:
    CAX: Subplot axes 
    STDs: Standard deviations
    RMSs: Centered Root Mean Square Difference 
    CORs: Correlation
    '''

    # Check amount of values provided and display options list if needed
    import numbers

    nargin = len(args)
    if nargin == 0:
        # Display options list
        _display_taylor_diagram_options()
        return [], [], [], []
    elif nargin == 3:
        stds, rmss, cors = args
        CAX = plt.gca()
    elif nargin == 4:
        CAX, stds, rmss, cors = args
        if not hasattr(CAX, 'axes'):
            raise ValueError('First argument must be a matplotlib axes.')
    else:
        raise ValueError('Must supply 3 or 4 arguments.')
    del nargin

    # Check data validity
    STDs = _ensure_np_array_or_die(stds, "STDs")
    RMSs = _ensure_np_array_or_die(rmss, "RMSs")
    CORs = _ensure_np_array_or_die(cors, "CORs")

    return CAX, STDs, RMSs, CORs


def _plot_pattern_diagram_colorbar(ax: matplotlib.axes.Axes, X, Y, Z,
                                   option: dict) -> None:
    '''
    Plots color markers on a pattern diagram shaded according to a 
    supplied value.
    
    Values are indicated via a color bar on the plot.
    
    Plots color markers on a target diagram according their (X,Y) locations.
    The color shading is accomplished by plotting the markers as a scatter 
    plot in (X,Y) with the colors of each point specified using Z as a 
    vector.
    
    The color range is controlled by option['cmapzdata'].
    option['colormap'] = 'on' :
        the scatter function maps the elements in Z to colors in the 
        current colormap
    option['colormap']= 'off' : the color axis is mapped to the range
        [min(Z) max(Z)]       
    option.locationColorBar   : location for the colorbar, 'NorthOutside'
                                or 'eastoutside'
    
    The color bar is titled using the content of option['titleColorBar'] 
    (if non-empty string).
    
    INPUTS:
    ax     : matplotlib.axes.Axes object in which the Taylor diagram will be
            plotted
    x : x-coordinates of markers
    y : y-coordinates of markers
    z : z-coordinates of markers (used for color shading)
    option : dictionary containing option values.
    option['colormap'] : 'on'/'off' switch to map color shading of markers 
        to colormap ('on') or min to max range of Z values ('off').
    option['titleColorBar'] : title for the color bar
    
    OUTPUTS:
    None.
    
    Created on Nov 30, 2016
    Revised on Jan 1, 2019
    
    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    '''

    '''
    Plot color shaded data points using scatter plot
    Keyword s defines marker size in points^2
            c defines the sequence of numbers to be mapped to colors 
            using the cmap and norm
    '''
    fontSize = rcParams.get('font.size')
    cxscale = fontSize / 10  # scale color bar by font size
    markerSize = option['markersize'] * 2

    hp = plt.scatter(X, Y, s=markerSize, c=Z, marker=option['cmap_marker'],
                     cmap=option['cmap'], vmin=option['cmap_vmin'],
                     vmax=option['cmap_vmax'])
    hp.set_facecolor(hp.get_edgecolor())

    # Set parameters for color bar location
    location = option['locationcolorbar'].lower()
    xscale = 1.0
    labelpad = -25
    if location == 'northoutside':
        orientation = 'horizontal'
        aspect = 6
        fraction = 0.04
    elif location == 'eastoutside':
        orientation = 'vertical'
        aspect = 25
        fraction = 0.15
        if 'checkstats' in option:
            # Taylor diagram
            xscale = 0.5
            cxscale = 6 * fontSize / 10
            labelpad = -30
    else:
        raise ValueError('Invalid color bar location: ' + option['locationcolorbar']);

    # Add color bar to plot
    if option['colormap'] == 'on':
        # map color shading of markers to colormap 
        hc = plt.colorbar(hp, orientation=orientation, aspect=aspect,
                          fraction=fraction, pad=0.06, ax=ax)

        # Limit number of ticks on color bar to reasonable number
        if orientation == 'horizontal':
            _setColorBarTicks(hc, 5, 20)

    elif option['colormap'] == 'off':
        # map color shading of markers to min to max range of Z values
        if len(Z) > 1:
            ax.clim(min(Z), max(Z))
            hc = ax.colorbar(hp, orientation=orientation, aspect=aspect,
                             fraction=fraction, pad=0.06, ticks=[min(Z), max(Z)],
                             ax=ax)

            # Label just min/max range
            hc.set_ticklabels(['Min.', 'Max.'])
    else:
        raise ValueError('Invalid option for option.colormap: ' +
                         option['colormap']);

    if orientation == 'horizontal':
        location = _getColorBarLocation(hc, option, xscale=xscale,
                                        yscale=7.5, cxscale=cxscale)
    else:
        location = _getColorBarLocation(hc, option, xscale=xscale,
                                        yscale=1.0, cxscale=cxscale)

    hc.ax.set_position(location)  # set new position
    hc.ax.tick_params(labelsize=fontSize)  # set tick label size

    hc.ax.xaxis.set_ticks_position('top')
    hc.ax.xaxis.set_label_position('top')

    # Title the color bar
    if option['titlecolorbar']:
        if orientation == 'horizontal':
            hc.set_label(option['titlecolorbar'], fontsize=fontSize)
        else:
            hc.set_label(option['titlecolorbar'], fontsize=fontSize,
                         labelpad=labelpad, y=1.05, rotation=0)
    else:
        hc.set_label(hc, 'Color Scale', fontsize=fontSize)


def _getColorBarLocation(hc, option, **kwargs):
    '''
    Determine location for color bar.
    
    Determines location to place color bar for type of plot:
    target diagram and Taylor diagram. Optional scale arguments
    (xscale,yscale,cxscale) can be supplied to adjust the placement of
    the colorbar to accommodate different situations.

    INPUTS:
    hc     : handle returned by colorbar function
    option : dictionary containing option values. (Refer to 
            display_target_diagram_options function for more 
            information.)
    
    OUTPUTS:
    location : x, y, width, height for color bar
    
    KEYWORDS:
    xscale  : scale factor to adjust x-position of color bar
    yscale  : scale factor to adjust y-position of color bar
    cxscale : scale factor to adjust thickness of color bar
    '''

    # Check for optional arguments and set defaults if required
    if 'xscale' in kwargs:
        xscale = kwargs['xscale']
    else:
        xscale = 1.0
    if 'yscale' in kwargs:
        yscale = kwargs['yscale']
    else:
        yscale = 1.0
    if 'cxscale' in kwargs:
        cxscale = kwargs['cxscale']
    else:
        cxscale = 1.0

    # Get original position of color bar and not modified position
    # because of Axes.apply_aspect being called.
    cp = hc.ax.get_position(original=True)

    # Calculate location : [left, bottom, width, height]
    if 'checkstats' in option:
        # Taylor diagram
        location = [cp.x0 + xscale * 0.5 * (1 + math.cos(math.radians(45))) * cp.width, yscale * cp.y0,
                    cxscale * cp.width / 6, cp.height]
    else:
        # target diagram
        location = [cp.x0 + xscale * 0.5 * (1 + math.cos(math.radians(60))) * cp.width, yscale * cp.y0,
                    cxscale * cp.width / 6, cxscale * cp.height]

    return location


def _setColorBarTicks(hc, numBins, lenTick):
    '''
    Determine number of ticks for color bar.
    
    Determines number of ticks for colorbar so tick labels do not
    overlap.

    INPUTS:
    hc      : handle of colorbar
    numBins : number of bins to use for determining number of 
            tick values using ticker.MaxNLocator
    lenTick : maximum number of characters for all the tick labels
    
    OUTPUTS:
    None

    '''

    maxChar = 10
    lengthTick = lenTick
    while lengthTick > maxChar:
        # Limit number of ticks on color bar to numBins-1
        hc.locator = ticker.MaxNLocator(nbins=numBins, prune='both')
        hc.update_ticks()

        # Check number of characters in tick labels is 
        # acceptable, otherwise reduce number of bins
        locs = str(hc.get_ticks())
        locs = locs[1:-1].split()
        lengthTick = 0
        for tick in locs:
            tickStr = str(tick).rstrip('.')
            lengthTick += len(tickStr)
        if lengthTick > maxChar: numBins -= 1


def _plot_pattern_diagram_markers(ax: matplotlib.axes.Axes, X, Y, option: dict):
    '''
    Plots color markers on a pattern diagram in the provided subplot axis.
    
    Plots color markers on a target diagram according their (X,Y) 
    locations. The symbols and colors are chosen automatically with a 
    limit of 70 symbol & color combinations.
    
    The color bar is titled using the content of option['titleColorBar'] 
    (if non-empty string).

    It is a direct adaptation of the _plot_pattern_diagram_markers() function
    for the scenario in which the Taylor diagram is draw in an
    matplotlib.axes.Axes object.
    
    INPUTS:
    ax     : the matplotlib.axes.Axes to receive the plot
    x      : x-coordinates of markers
    y      : y-coordinates of markers
    z      : z-coordinates of markers (used for color shading)
    option : dictionary containing option values. (Refer to 
        GET_TARGET_DIAGRAM_OPTIONS function for more information.)
    option['axismax'] : maximum for the X & Y values. Used to limit
        maximum distance from origin to display markers
    option['markerlabel'] : labels for markers
    
    OUTPUTS:
    None

    Authors:
    Peter A. Rochford
    rochford.peter1@gmail.com

    Andre D. L. Zanchetta (adapting Peter A. Rochford's code)
        adlzanchetta@gmail.com

    Created on Nov 30, 2016
    Revised on Aug 14, 2022
    '''

    # Set face color transparency
    alpha = option['alpha']

    # Set font and marker size
    fontSize = matplotlib.rcParams.get('font.size') - 2
    markerSize = option['markersize']

    # Check enough labels provided if markerlabel provided. Not a problem if labels
    # provided via the markers option.
    numberLabel = len(option['markerlabel'])
    if numberLabel > 0:
        if isinstance(option['markerlabel'], list) and numberLabel < len(X):
            raise ValueError('Insufficient number of marker labels provided.\n' +
                             'target: No. labels=' + str(numberLabel) + ' < No. markers=' +
                             str(len(X)) + '\n' +
                             'taylor: No. labels=' + str(numberLabel + 1) + ' < No. markers=' +
                             str(len(X) + 1))
        elif isinstance(option['markerlabel'], dict) and numberLabel > 70:
            raise ValueError('Insufficient number of marker labels provided.\n' +
                             'target: No. labels=' + str(numberLabel) + ' > No. markers= 70')

    if option['markerlegend'] == 'on':
        # Check that marker labels have been provided
        if option['markerlabel'] == '' and option['markers'] == None:
            raise ValueError('No marker labels provided.')

        # Plot markers of different color and symbols with labels displayed in a legend
        limit = option['axismax']
        hp = ()
        rgba = None

        if option['markers'] is None:
            # Define default markers (function)
            marker, markercolor = _get_default_markers(X, option)

            # Plot markers at data points
            labelcolor = []
            markerlabel = []
            for i, xval in enumerate(X):
                if abs(X[i]) <= limit and abs(Y[i]) <= limit:
                    h = ax.plot(X[i], Y[i], marker[i], markersize=markerSize,
                                markerfacecolor=markercolor[i],
                                markeredgecolor=markercolor[i][0:3] + (1.0,),
                                markeredgewidth=2)
                    hp += tuple(h)
                    labelcolor.append(option['markerlabelcolor'])
                    markerlabel.append(option['markerlabel'][i])

        else:
            # Obtain markers from option['markers']
            labels, labelcolor, marker, markersize, markerfacecolor, markeredgecolor = \
                _get_single_markers(option['markers'])

            # Plot markers at data points

            markerlabel = []
            for i, xval in enumerate(X):
                if abs(X[i]) <= limit and abs(Y[i]) <= limit:
                    h = ax.plot(X[i], Y[i], marker[i], markersize=markersize[i],
                                markerfacecolor=markerfacecolor[i],  # markerfacecolor[i],
                                markeredgecolor=markeredgecolor[i],
                                markeredgewidth=2)
                    hp += tuple(h)
                    markerlabel.append(labels[i])

        # Add legend
        if len(markerlabel) == 0:
            warnings.warn('No markers within axis limit ranges.')
        else:
            _add_legend(markerlabel, labelcolor, option, rgba, markerSize, fontSize, hp)
    else:
        # Plot markers as dots of a single color with accompanying labels

        # Plot markers at data points
        limit = option['axismax']

        # Define edge and face colors of the markers
        edge_color = _get_from_dict_or_default(option, 'markercolor', 'markercolors', 'edge')
        if edge_color is None: edge_color = 'r'
        face_color = _get_from_dict_or_default(option, 'markercolor', 'markercolors', 'face')
        if face_color is None: face_color = edge_color
        face_color = clr.to_rgb(face_color) + (alpha,)

        labelcolor = []
        for i in range(len(X)):
            xval, yval = X[i], Y[i]
            if abs(xval) <= limit and abs(yval) <= limit:
                # Plot marker
                ax.plot(xval, yval, option['markersymbol'],
                        markersize=markerSize,
                        markerfacecolor=face_color,
                        markeredgecolor=edge_color)
                labelcolor.append(option['markerlabelcolor'])

                # Check if marker labels provided
                if type(option['markerlabel']) is list:
                    # Label marker
                    ax.text(xval, yval, option['markerlabel'][i],
                            color=option['markerlabelcolor'],
                            verticalalignment='bottom',
                            horizontalalignment='right',
                            fontsize=fontSize)

            del i, xval, yval

        # Add legend if labels provided as dictionary
        markerlabel = option['markerlabel']
        marker_label_color = clr.to_rgb(edge_color) + (alpha,)
        if type(markerlabel) is dict:
            _add_legend(markerlabel, labelcolor, option, marker_label_color, markerSize, fontSize)


def _get_default_markers(X, option: dict):
    # def _get_default_markers(X, option: dict) -> tuple[list, list]: #fails with Python 3.6
    '''
    Provides a list of default markers and marker colors.
    
    Returns a list of 70 marker symbol & color combinations.

    INPUTS:
    X      : x-coordinates of markers
    option : dictionary containing option values. (Refer to 
        GET_TARGET_DIAGRAM_OPTIONS function for more information.)
    option['markercolor'] : single color to use for all markers
    option['markerlabel'] : labels for markers
    
    OUTPUTS:
    marker      : list of marker symbols
    markercolor : list of marker colors

    Authors:
    Peter A. Rochford
    rochford.peter1@gmail.com

    Created on Mar 12, 2023
    Revised on Mar 12, 2023
    '''
    # Set face color transparency
    alpha = option['alpha']

    # Define list of marker symbols and colros
    kind = ['+', 'o', 'x', 's', 'd', '^', 'v', 'p', 'h', '*']
    colorm = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'gray']
    if len(X) > 80:
        print('You must introduce new markers to plot more than 70 cases.')
        print('The ''marker'' character array need to be extended inside the code.')

    if len(X) <= len(kind):
        # Define markers with specified color
        marker = []
        markercolor = []
        if option['markercolor'] is None:
            for i, color in enumerate(colorm):
                rgba = clr.to_rgb(color) + (alpha,)
                marker.append(kind[i] + color)
                markercolor.append(rgba)
        else:
            rgba = clr.to_rgb(option['markercolor']) + (alpha,)
            for symbol in kind:
                marker.append(symbol + option['markercolor'])
                markercolor.append(rgba)
    else:
        # Define markers and colors using predefined list
        marker = []
        markercolor = []
        for color in colorm:
            for symbol in kind:
                marker.append(symbol + color)
                rgba = clr.to_rgb(color) + (alpha,)
                markercolor.append(rgba)

    return marker, markercolor


def _add_legend(markerLabel, labelcolor, option, rgba, markerSize, fontSize, hp=[]):
    '''
    Adds a legend to a pattern diagram.
    
    Adds a legend to a plot according to the data type containing the 
    provided labels. If labels are provided as a list they will appear 
    in the legend beside the marker provided in the list of handles in 
    a one-to-one match. If labels are provided as a dictionary they will 
    appear beside a dot with the color value given to the label.
    
    INPUTS:
    markerLabel : list or dict variable containing markers and labels to
                appear in legend
                
                A list variable must have the format:
                markerLabel = ['M1', 'M2', 'M3']
                
                A dictionary variable must have the format:
                markerLabel = = {'ERA-5': 'r', 'TRMM': 'b'}
                where each key is the label and each value the color for 
                the marker
    labelcolor : color of marker label
    
    option : dictionary containing option values. (Refer to 
        GET_TARGET_DIAGRAM_OPTIONS function for more information.)
    option['numberpanels'] : Number of panels to display
                            = 1 for positive correlations
                            = 2 for positive and negative correlations
    rgba : a 4-tuple where the respective tuple components represent red, 
        green, blue, and alpha (opacity) values for a color
    markerSize : point size of markers
    fontSize : font size in points of labels
    hp : list of plot handles that match markerLabel when latter is a list
    
    OUTPUTS:
    None

    Created on Mar 2, 2019
    Revised on Mar 2, 2019
    
    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    '''

    if type(markerLabel) is list:

        # Check for empty list of plot handles
        if len(hp) == 0:
            raise ValueError('Empty list of plot handles')
        elif len(hp) != len(markerLabel):
            raise ValueError('Number of labels and plot handle do not match: ' +
                             str(len(markerLabel)) + ' != ' + str(len(hp)))

        # Add legend using labels provided as list
        if len(markerLabel) <= 6:
            # Put legend in a default location
            markerlabel = tuple(markerLabel)
            if option['set_legend']['legend_on']:
                leg = plt.legend(hp, markerlabel, loc='upper right',
                                 fontsize=fontSize, numpoints=1,
                                 bbox_to_anchor=(option['set_legend']['bbox_to_anchor_x'], option['set_legend']['bbox_to_anchor_y']))
            else:
                leg = plt.legend(hp, markerlabel, loc='upper right',
                                 fontsize=fontSize, numpoints=1,
                                 bbox_to_anchor=(1.5, 1.))
        else:
            # Put legend to right of the plot in multiple columns as needed

            nmarkers = len(markerLabel)
            if option['markerlayout'][1] is None:
                nrow = option['markerlayout'][0]
                ncol = int(math.ceil(nmarkers / nrow))
            else:
                ncol = option['markerlayout'][1]
            markerlabel = tuple(markerLabel)

            # Shift figure to include legend
            plt.gcf().subplots_adjust(right=0.6)

            # Plot legend of multi-column markers
            # Note: do not use bbox_to_anchor as this cuts off the legend
            if option['set_legend']['legend_on']:
                loc = (option['set_legend']['bbox_to_anchor_x'], option['set_legend']['bbox_to_anchor_y'])
            else:
                if 'circlelinespec' in option:
                    loc = (1.2, 0.25)
                else:
                    loc = (1.1, 0.25)
            leg = plt.legend(hp, markerlabel, loc=loc, fontsize=fontSize,
                             numpoints=1, ncol=ncol)

    elif type(markerLabel) is dict:

        # Add legend using labels provided as dictionary

        # Define legend elements
        legend_elements = []
        for key, value in markerLabel.items():
            legend_object = Line2D([0], [0], marker='.', markersize=markerSize,
                                   markerfacecolor=rgba, markeredgecolor=value, label=key, linestyle='')
            legend_elements.append(legend_object)

        # Put legend in a default location
        leg = plt.legend(handles=legend_elements, loc='upper right',
                         fontsize=fontSize, numpoints=1,
                         bbox_to_anchor=(1.4, 1.1))

        if _checkKey(option, 'numberpanels') and option['numberpanels'] == 2:
            # add padding so legend is not cut off
            plt.tight_layout(pad=1)
    else:
        raise Exception('markerLabel type is not a list or dictionary: ' +
                        str(type(markerLabel)))

    # Set color of text in legend
    for i, text in enumerate(leg.get_texts()):
        text.set_color(labelcolor[i])


def _checkKey(dictionary, key):
    if key in dictionary.keys():
        return True
    else:
        return False


def _get_from_dict_or_default(options: dict, default_key: str, dict_key: str, key_key: str):
    '''
    Gets values of keys from dictionary or returns defaults.

    Given a dictionary, the key of the default value (default_key), the key of a potential
    internal dictionary (dict_key) and the key of a potential value within dict_key
    (key_key), return the value of key_key if possible, or the value of default_key
    otherwise.

    INPUTS:
    options:     Dictionary containing 'default_key' and possibly 'dict_key.key_key'
    default_key: Key of the default value within 'options'
    dict_key:    Key of the potential internal dictionary within 'options'
    key_key:     Key of the potential value within 'dict_key'

    OUTPUTS:
    return: The value of 'options.dict_key.key_key' or of 'options.default_key'

    Author: Andre D. L. Zanchetta
        adlzanchetta@gmail.com

    Created on Aug 14, 2022
    '''

    if options[dict_key] is None:
        return options[default_key]
    elif key_key not in options[dict_key]:
        return options[default_key]
    elif options[dict_key][key_key] is None:
        return options[default_key]
    else:
        return options[dict_key][key_key]


def _get_single_markers(markers: dict):
    # def _get_single_markers(markers: dict) -> tuple[list, list, list, list, list, list]: #fails with Python 3.6
    '''
    Provides a list of markers and their properties as stored in a dictionary.
    
    Returns a list of properties for individual markers as given in the 'markers' 
    dictionary. Each marker can have its individual set of properties. 

    INPUTS:
    markers : Dictionary providing individual control of the marker
            key - text label for marker, e.g. '14197'
            key['labelColor'] - color of marker label, e.g. 'r' for red
            key['symbol'] - marker symbol, e.g. 's' for square
            key['size'] - marker size, e.g. 9
            key['faceColor'] - marker face color, e.g. 'b' for blue
            key['edgeColor'] - marker edge color, e.g. 'k' for black line
    
    OUTPUTS:
    markerlabel     : list of text labels for marker
    labelcolor      : list of color of marker label
    marker          : list of marker symbol & color
    markersize      : list of marker size
    markerfacecolor : list of marker face color
    markeredgecolor : list of marker edge color

    Authors:
    Peter A. Rochford
    rochford.peter1@gmail.com

    Created on Mar 12, 2023
    Revised on Mar 13, 2023
    '''
    if markers is None:
        raise ValueError("Empty dictionary provided for option['markers']")

    labelcolor = []
    marker = []
    markerfacecolor = []
    markeredgecolor = []
    markerlabel = []
    markersize = []

    # Iterate through keys in dictionary
    for key in markers:
        color = markers[key]['faceColor']
        symbol = markers[key]['symbol']
        SymbolColor = symbol + color
        if color == 'none':
            SymbolColor = symbol + 'w'
        marker.append(SymbolColor)
        markersize.append(markers[key]['size'])
        markerfacecolor.append(color)
        markeredgecolor.append(markers[key]['edgeColor'])
        markerlabel.append(key)  # store label
        labelcolor.append(markers[key]['labelColor'])

    return markerlabel, labelcolor, marker, markersize, markerfacecolor, markeredgecolor


def _check_taylor_stats(STDs, CRMSDs, CORs, threshold=0.01):
    '''
    Checks input statistics satisfy Taylor diagram relation to <1%.

    Function terminates with an error if not satisfied. The threshold is
    the ratio of the difference between the statistical metrics and the
    centered root mean square difference:

    abs(CRMSDs^2 - (STDs^2 + STDs(1)^2 - 2*STDs*STDs(1)*CORs))/CRMSDs^2

    Note that the first element of the statistics vectors must contain
    the value for the reference field.

    INPUTS:
    STDs      : Standard deviations
    CRMSDs    : Centered Root Mean Square Difference 
    CORs      : Correlation
    threshold : limit for acceptance, e.g. 0.1 for 10% (default 0.01)

    OUTPUTS:
    None.

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Dec 3, 2016
    '''
    if threshold < 1e-7:
        raise ValueError('threshold value must be positive: ' + str(threshold))

    diff = np.square(CRMSDs[1:]) \
           - (np.square(STDs[1:]) + np.square(STDs[0]) \
              - 2.0 * STDs[0] * np.multiply(STDs[1:], CORs[1:]))
    diff = np.abs(np.divide(diff, np.square(CRMSDs[1:])))
    index = np.where(diff > threshold)

    if np.any(index):
        ii = np.where(diff != 0)
        if len(ii) == len(diff):
            raise ValueError('Incompatible data\nYou must have:' +
                             '\nCRMSDs - sqrt(STDs.^2 + STDs[0]^2 - ' +
                             '2*STDs*STDs[0].*CORs) = 0 !')
        else:
            raise ValueError('Incompatible data indices: {}'.format(ii) +
                             '\nYou must have:\nCRMSDs - sqrt(STDs.^2 + STDs[0]^2 - ' +
                             '2*STDs*STDs[0].*CORs) = 0 !')

    return diff


def _calc_rinc(tick: list) -> float:
    '''
    Calculate axis tick increment given list of tick values.
    
    INPUTS:
    tick: axis values at which to plot grid circles

    return: axis tick increment
    '''
    rinc = (max(tick) - min(tick)) / len(tick)
    return rinc


def _check_dict_with_keys(variable_name: str, dict_obj: Union[dict, None],
                          accepted_keys: set, or_none: bool = False) -> None:
    '''
    Check if an argument in the form of dictionary has valid keys.
    :return: None. Raise 'ValueError' if evaluated variable is considered invalid. 
    '''

    # if variable is None, check if it can be None
    if dict_obj is None:
        if or_none:
            return None
        else:
            raise ValueError('%s cannot be None!' % variable_name)

    # check if every key provided is valid
    for key in dict_obj.keys():
        if key not in accepted_keys:
            raise ValueError('Unrecognized option of %s: %s' % (variable_name, key))
        del key

    return None


def _is_int(element):
    '''
    Check if variable is an integer. 
    '''
    try:
        int(element)
        return True
    except ValueError:
        return False


def _is_float(element):
    '''
    Check if variable is a float. 
    '''
    try:
        float(element)
        return True
    except ValueError:
        return False


def _is_list_in_string(element):
    '''
    Check if variable is list provided as string 
    '''
    return bool(re.search(r'\[|\]', element))


def _default_options(CORs: list) -> dict:
    '''
    Set default optional arguments for taylor_diagram function.
    
    Sets the default optional arguments for the TAYLOR_DIAGRAM 
    function in an OPTION dictionary. Default values are 
    assigned to selected optional arguments. 
    
    INPUTS:
    CORs : values of correlations
        
    OUTPUTS:
    option : dictionary containing option values. (Refer to 
            display_taylor_diagram_options function for more information.)
    option['alpha']           : blending of symbol face color (0.0 
                                transparent through 1.0 opaque). (Default : 1.0)
    option['axismax']         : maximum for the radial contours
    option['checkstats']      : Check input statistics satisfy Taylor 
                                relationship (Default : 'off')
    option['cmap']            : Choice of colormap. (Default : 'jet')
    option['cmap_vmin']       : minimum range of colormap (Default : None)
    option['cmap_vmax']       : maximum range of colormap (Default : None)
    option['cmap_marker']     : maximum range of colormap (Default : None)
    option['cmapzdata']       : data values to use for color mapping of
                                markers, e.g. RMSD or BIAS. (Default empty)
                                
    option['colcor']          : color for correlation coefficient labels (Default : blue)
    option['colscor']         : dictionary with two possible colors as keys ('grid',
                                'tick_labels') or None, if None then considers only the
                                value of 'colscor' (Default: None)
    option['colframe']        : color for the y (left) and x (bottom) spines
    option['colobs']          : color for observation labels (Default : magenta)
    option['colormap']        : 'on'/'off' switch to map color shading of
                                markers to CMapZData values ('on') or min to
                                max range of CMapZData values ('off').
                                (Default : 'on')
    option['colrms']          : color for RMS labels (Default : medium green)
    option['colstd']          : color for STD labels (Default : black)
    option['colsstd']         : dictionary with two possible colors keys ('ticks',
                                'tick_labels') or None, if None then considers only the
                                value of 'colstd' (Default: None)
    option['labelrms']        : RMS axis label (Default: 'RMSD')
    option['labelweight']     : weight of the x/y/angular axis labels
    option['locationcolorbar']: location for the colorbar, 'NorthOutside' or
                                'EastOutside'

    option['markercolor']     : single color to use for all markers (Default: None)
    option['markercolors']    : dictionary with two colors as keys ('face', 'edge')
                                or None. If None or 'markerlegend' == 'on' then
                                considers only the value of 'markercolor'. (Default: None)
    option['markerdisplayed'] : markers to use for individual experiments
    option['markerlabel']     : name of the experiment to use for marker
    option['markerlabelcolor']: marker label color (Default: 'k')
    option['markerlayout']    : matrix layout for markers in legend [nrow, ncolumn] 
                                (Default [15, no. markers/15] ) 
    option['markerlegend']    : 'on'/'off' switch to display marker legend
                                (Default 'off')
    option['markerobs']       : marker to use for x-axis indicating observed 
                                STD. A choice of 'none' will suppress 
                                appearance of marker. (Default 'none')
    option['markers']         : Dictionary providing individual control of the marker
                            key - text label for marker, e.g. '14197'
                            key['labelColor'] - color of marker label, e.g. 'r' for red
                            key['symbol'] - marker symbol, e.g. 's' for square
                            key['size'] - marker size, e.g. 9
                            key['faceColor'] - marker face color, e.g. 'b' for blue
                            key['edgeColor'] - marker edge color, e.g. 'k' for black line
                            (Default: None)
    option['markersize']      : marker size (Default 10)
    option['markersymbol']    : marker symbol (Default '.')

    option['numberpanels']    : Number of panels to display
                                = 1 for positive correlations
                                = 2 for positive and negative correlations
                            (Default value depends on correlations (CORs))
    option['markersizeobs']      : OBS marker size (Default 10)

    option['overlay']         : 'on'/'off' switch to overlay current
                                statistics on Taylor diagram (Default 'off')
                                Only markers will be displayed.
    option['rincrms']         : axis tick increment for RMS values
    option['rincstd']         : axis tick increment for STD values
    option['rmslabelformat']  : string format for RMS contour labels, e.g. '0:.2f'.
                                (Default '0', format as specified by str function)

    option['showlabelscor']   : show correlation coefficient labels 
                                (Default: 'on')
    option['showlabelsrms']   : show RMS labels (Default: 'on')
    option['showlabelsstd']   : show STD labels (Default: 'on')

    option['stylecor']        : line style for correlation coefficient grid 
                                lines (Default: dash-dot '-.')
    option['styleobs']        : line style for observation grid line. A choice of
                                empty string '' will suppress appearance of the
                                grid line (Default: '')
    option['stylerms']        : line style for RMS grid lines 
                                (Default: dash '--')
    option['stylestd']        : line style for STD grid lines 
                                (Default: dotted ':')

    option['taylor_options_file'] name of CSV file containing values for optional
                                arguments of the taylor_diagram function. If no file
                                suffix is given, a ".csv" is assumed. (Default: empty string '')

    option['tickcor'][panel]  : tick values for correlation coefficients for
                                two types of panels
    option['tickrms']         : RMS values to plot grid circles from
                                observation point 
    option['tickstd']         : STD values to plot grid circles from origin 
    option['tickrmsangle']    : tick RMS angle (Default: 135 degrees)
    option['titleColorBar']   : title for the colorbar
    option['titlecor']        : show correlation coefficient axis label 
                                (Default: 'on')
    option['titlecorshape']   : defines the shape of the label "correlation coefficient"
                                as either 'curved' or 'linear' (Default: 'curved')
    option['titleobs']        : label for observation point (Default: '')
    option['titlerms']        : show RMS axis label (Default: 'on')
    option['titlestd']        : show STD axis label (Default: 'on')
    option['titlermsdangle']  : angle at which to display the 'RMSD' label for the RMS contours
                                (Default: 160 degrees)

    option['widthcor']        : linewidth for correlation coefficient grid 
                                lines (Default: .8)
    option['widthobs']        : linewidth for observation grid line (Default: .8)
    option['widthrms']        : linewidth for RMS grid lines (Default: .8)
    option['widthstd']        : linewidth for STD grid lines (Default: .8)
    
    option['stdlabelsize']    : STD label size (Default 15)
    option['corlabelsize']    : correlation label size (Default: 15)
    option['rmslabelsize']    : RMS label size (Default: 15)

    option['ticksizestd']     : STD ticks size (Default: 13)
    option['ticksizecor']     : correlation ticks size (Default: 13)
    option['ticksizerms']     : RMS ticks size (Default: 13)

    option['NormalizedSTD']   : Normalized for STD (Default: False)

    Author:
    
    Peter A. Rochford
        rochford.peter1@gmail.com

    Created on Sep 12, 2022
    Revised on Sep 12, 2022
    '''

    from matplotlib import rcParams

    # Set default parameters for all options
    option = {}
    option['alpha'] = 1.0
    option['axismax'] = 0.0
    option['checkstats'] = 'off'

    option['cmap'] = 'jet'
    option['cmap_vmin'] = None
    option['cmap_vmax'] = None
    option['cmap_marker'] = 'd'
    option['cmapzdata'] = []

    option['colcor'] = (0, 0, 1)  # blue
    option['colscor'] = None  # if None, considers 'colcor' only
    option['colobs'] = 'm'  # magenta
    option['colrms'] = (0, .6, 0)  # medium green
    option['colstd'] = (0, 0, 0)  # black
    option['colsstd'] = None  # if None, considers 'colstd' only
    option['colframe'] = '#000000'  # black
    option['colormap'] = 'on'

    option['labelrms'] = 'RMSD'
    option['labelweight'] = 'bold'  # weight of the x/y labels ('light', 'normal', 'bold', ...)
    option['locationcolorbar'] = 'NorthOutside'

    option['markercolor'] = None
    option['markercolors'] = None  # if None, considers 'markercolor' only
    option['markerdisplayed'] = 'marker'
    option['markerlabel'] = ''
    option['markerlabelcolor'] = 'k'
    option['markerlayout'] = [15, None]
    option['markerlegend'] = 'off'
    option['set_legend'] = dict(legend_on=False, bbox_to_anchor_x=1.4, bbox_to_anchor_y=1.1)

    option['markerobs'] = 'none'
    option['markers'] = None
    option['markersize'] = 10
    option['markersymbol'] = '.'
    option['markersizeobs'] = 10

    option['titlecorshape'] = "curved"

    # panels: double (2) or single (1)
    negative = CORs[np.where(CORs < 0.0)]
    option['numberpanels'] = 2 if (len(negative) > 0) else 1
    del negative

    option['overlay'] = 'off'
    option['rincrms'] = []
    option['rincstd'] = []
    option['rmslabelformat'] = '0'

    option['showlabelscor'] = 'on'
    option['showlabelsrms'] = 'on'
    option['showlabelsstd'] = 'on'

    option['stylecor'] = '-.'
    option['styleobs'] = ''
    option['stylerms'] = '--'
    option['stylestd'] = ':'

    option['taylor_options_file'] = ''

    # Note that "0" must be explicitly given or a scientific number is
    # stored
    tickval1 = [1, 0.99, 0.95, 0]
    middle = np.linspace(0.9, 0.1, 9)
    tickval1[3:3] = middle
    tickval2 = tickval1[:]
    values = np.linspace(-0.1, -0.9, 9)
    tickval2.extend(values)
    tickval2.extend([-0.95, -0.99, -1])
    option['tickcor'] = (tickval1, tickval2)  # store as tuple
    del tickval1, tickval2, middle, values

    option['tickrms'] = []
    option['tickstd'] = []
    option['tickrmsangle'] = -1
    option['titlecolorbar'] = ''
    option['titlecor'] = 'on'
    option['titleobs'] = ''
    option['titlerms'] = 'on'
    option['titlermsdangle'] = 160.0
    option['titlestd'] = 'on'

    lineWidth = rcParams.get('lines.linewidth')
    option['widthcor'] = lineWidth
    option['widthobs'] = lineWidth
    option['widthrms'] = lineWidth
    option['widthstd'] = lineWidth

    option['stdlabelsize'] = 13
    option['corlabelsize'] = 13
    option['rmslabelsize'] = 13

    option['ticksizestd'] = 11
    option['ticksizecor'] = 11
    option['ticksizerms'] = 11

    option['normalizedstd'] = False

    return option


def _get_options(option: dict, **kwargs) -> dict:
    '''
    Get values for optional arguments for taylor_diagram function.
    
    Gets the default optional arguments for the TAYLOR_DIAGRAM 
    function in an OPTION dictionary. 
    
    INPUTS:
    option  : dictionary containing default option values
    *kwargs : variable-length keyword argument list. The keywords by 
            definition are dictionaries with keys that must correspond to 
            one of the choices given in the _default_options function.
        
    OUTPUTS:
    option : dictionary containing option values

    Author:
    
    Peter A. Rochford
        rochford.peter1@gmail.com

    Created on Sep 12, 2022
    Revised on Sep 12, 2022
    '''

    # Check for valid keys and values in dictionary
    for optname, optvalue in kwargs.items():
        optname = optname.lower()
        if optname == 'nonrmsdz':
            raise ValueError('nonrmsdz is an obsolete option. Use cmapzdata instead.')

        if not optname in option:
            raise ValueError('Unrecognized option: ' + optname)
        else:
            # Replace option value with that from arguments
            if optname == 'tickcor':
                list1 = option['tickcor'][0]
                list2 = option['tickcor'][1]
                if option['numberpanels'] == 1:
                    list1 = optvalue
                else:
                    list2 = optvalue
                option['tickcor'] = (list1, list2)
                del list1, list2
            else:
                option[optname] = optvalue

            # Check values for specific options
            if optname == 'checkstats':
                option['checkstats'] = check_on_off(option['checkstats'])

            elif optname == 'cmapzdata':
                if isinstance(option[optname], str):
                    raise ValueError('cmapzdata cannot be a string!')
                elif isinstance(option[optname], bool):
                    raise ValueError('cmapzdata cannot be a boolean!')
                option['cmapzdata'] = optvalue

            elif optname == 'markerlabel':
                if type(optvalue) is list:
                    option['markerlabel'] = optvalue[1:]
                elif type(optvalue) is dict:
                    option['markerlabel'] = optvalue
                else:
                    raise ValueError('markerlabel value is not a list or dictionary: ' +
                                     str(optvalue))

            elif optname == 'markerlegend':
                option['markerlegend'] = check_on_off(option['markerlegend'])

            elif optname == 'overlay':
                option['overlay'] = check_on_off(option['overlay'])

            elif optname == 'rmslabelformat':
                # Check for valid string format
                labelFormat = '{' + optvalue + '}'
                try:
                    labelFormat.format(99.0)
                except ValueError:
                    raise ValueError('Invalid string format for rmslabelformat: ' + optvalue)

            elif optname in {'showlabelscor', 'showlabelsrms', 'showlabelsstd'}:
                option[optname] = check_on_off(option[optname])

            elif optname == 'tickrms':
                option['tickrms'] = np.sort(optvalue)
                option['rincrms'] = _calc_rinc(option['tickrms'])

            elif optname == 'tickstd':
                option['tickstd'] = np.sort(optvalue)
                option['rincstd'] = _calc_rinc(option['tickstd'])

            elif optname in {'titlecor', 'titlerms', 'titlestd'}:
                option[optname] = check_on_off(option[optname])

            elif optname in {'markercolors', 'colscor', 'colsstd'}:
                accepted_keys = {
                    'markercolors': {'face', 'edge'},
                    'colscor': {'grid', 'title', 'tick_labels'},
                    'colsstd': {'grid', 'title', 'tick_labels', 'ticks'}
                }
                _check_dict_with_keys(optname, option[optname],
                                      accepted_keys[optname], or_none=True)
                del accepted_keys

            if optname in {'stdlabelsize', 'corlabelsize', 'rmslabelsize', 'ticksizestd', 'ticksizecor', 'ticksizerms'}:
                option[optname] = optvalue

            elif optname == 'normalizedstd':
                option[optname] = optvalue
            elif optname == 'set_legend':
                if option['markerlegend']:
                    option['set_legend'] = optvalue
            elif optname == 'markersizeobs':
                option['markersizeobs'] = optvalue
        del optname, optvalue

    return option


def _read_options(option: dict, **kwargs) -> dict:
    '''
    Reads the optional arguments from a CSV file. 
    
    Reads the optional arguments for taylor_diagram function from a 
    CSV file if a taylor_options_file parameter is provided that contains
    the name of a valid Comma Separated Value (CSV) file. Otherwise the
    function returns with no action taken. 
    
    INPUTS:
    option  : dictionary containing default option values

    *kwargs : variable-length keyword argument list. One of the keywords 
            must be in the list below for the function to perform any
            action.
    taylor_options_file : name of CSV file containing values for optional
                        arguments of the taylor_diagram function. If no file
                        suffix is given, a ".csv" is assumed. (Default: empty string '')
        
    OUTPUTS:
    option : dictionary containing option values

    Author:
    
    Kevin Wu, kevinwu5116@gmail.com

    Created on Sep 12, 2022
    Revised on Sep 12, 2022
    '''
    # Check if option filename provided
    name = ''
    for optname, optvalue in kwargs.items():
        optname = optname.lower()
        if optname == 'taylor_options_file':
            name = optvalue
            break
    if not name: return option

    # Check if CSV file suffix
    filename, file_extension = os.path.splitext(name)

    if file_extension == "":
        filename = name + '.csv'
    elif name.endswith('.csv'):
        filename = name
    else:
        raise Exception("Invalid file type: " + name)

    # Check if file exists
    if not os.path.isfile(filename):
        raise Exception("File does not exist: " + filename)

    # Load object from CSV file
    objectData = pd.read_csv(filename)

    # Parse object for keys and values
    keys = objectData.iloc[:, 0]
    values = objectData.iloc[:, 1].tolist()

    # Identify keys requiring special consideration   
    listkey = ['cmapzdata', 'rincrms', 'rincstd', 'tickcor', 'tickrms', 'tickstd']
    tuplekey = ['colcor', 'colrms', 'colstd']

    # Process for options read from CSV file
    for index in range(len(keys)):

        # Skip assignment if no value provided in CSV file
        if pd.isna(values[index]):
            continue

        # Convert list provided as string
        if _is_list_in_string(values[index]):
            # Remove brackets
            values[index] = values[index].replace('[', '').replace(']', '')

        if keys[index] in listkey:
            if pd.isna(values[index]):
                option[keys[index]] = []
            else:
                # Convert string to list of floats
                split_string = re.split(' |,', values[index])
                split_string = ' '.join(split_string).split()
                option[keys[index]] = [float(x) for x in split_string]

            if keys[index] == 'tickrms':
                option['rincrms'] = _calc_rinc(option[keys[index]])
            elif keys[index] == 'tickstd':
                option['rincstd'] = _calc_rinc(option[keys[index]])

        elif keys[index] in tuplekey:
            try:
                option[keys[index]] = eval(values[index])
            except NameError:
                raise Exception('Invalid ' + keys[index] + ': ' + values[index])
        elif keys[index] == 'rmslabelformat':
            option[keys[index]] = values[index]
        elif pd.isna(values[index]):
            option[keys[index]] = ''
        elif _is_int(values[index]):
            option[keys[index]] = int(values[index])
        elif _is_float(values[index]):
            option[keys[index]] = float(values[index])
        elif values[index] == 'None':
            option[keys[index]] = None
        else:
            option[keys[index]] = values[index]

    return option


def _get_taylor_diagram_options(*args, **kwargs) -> dict:
    '''
    Get optional arguments for taylor_diagram function.
    
    Retrieves the optional arguments supplied to the TAYLOR_DIAGRAM 
    function as a variable-length input argument list (*ARGS), and
    returns the values in an OPTION dictionary. Default values are 
    assigned to selected optional arguments. The function will terminate
    with an error if an unrecognized optional argument is supplied.
    
    INPUTS:
    *kwargs : variable-length keyword argument list. The keywords by 
            definition are dictionaries with keys that must correspond to 
            one choices given in the _default_options function.
    
    OUTPUTS:
    option : dictionary containing option values. (Refer to _default_options
            and display_taylor_diagram_options functions for more information.)

    Authors:
    
    Peter A. Rochford
        rochford.peter1@gmail.com
    
    Andre D. L. Zanchetta (adapting Peter A. Rochford's code)
        adlzanchetta@gmail.com

    Created on Nov 25, 2016
    Revised on Aug 14, 2022
    '''

    CORs = args[0]
    nargin = len(kwargs)

    # Set default parameters for all options
    option = _default_options(CORs)

    # No options requested, so return with only defaults
    if nargin == 0: return option

    # Check to see if the Key for the file exist
    name = ''
    for optname, optvalue in kwargs.items():
        optname = optname.lower()
        if optname == 'taylor_options_file':
            name = optvalue
            break

    if name:
        # Check if CSV file suffix
        filename, file_extension = os.path.splitext(name)

        if file_extension == "":
            filename = name + '.csv'
        elif name.endswith('.csv'):
            filename = name
        else:
            raise Exception("Invalid file type: " + name)

        # Check if file exists
        if not os.path.isfile(filename):
            raise Exception("File does not exist: " + filename)

        # Read the optional arguments for taylor_diagram function from a 
        # CSV file, if specified. 
        option = _read_options(option, **kwargs)

    # Check for valid keys and values in dictionary
    # Allows user to override options specified in CSV file
    option = _get_options(option, **kwargs)

    return option


def _get_taylor_diagram_axes(ax, rho, option) -> dict:
    '''
    Get axes value for taylor_diagram function.
    
    Determines the axes information for a Taylor diagram given the axis 
    values (X,Y) and the options in the dictionary OPTION returned by 
    the _get_taylor_diagram_options function.

    INPUTS:
    ax     : the matplotlib.axes.Axes to receive the plot
    rho    : radial coordinate
    option : dictionary containing option values. (Refer to 
            get_taylor_diagram_subplot_options() function for more information.)

    OUTPUTS:
    axes         : dictionary containing axes information for Taylor diagram
    axes['dx']   : observed standard deviation
    axes['next'] : directive on how to add next plot
    axes['rinc'] : increment for radial coordinate
    axes['rmax'] : maximum value for radial coordinate
    axes['rmin'] : minimum value for radial coordinate
    axes['tc']   : color for x-axis
    Also modifies the input variables 'ax' and 'option'
    
    Authors: Peter A. Rochford
        rochford.peter1@gmail.com

    Andre D. L. Zanchetta (adapting Peter A. Rochford's code)
        adlzanchetta@gmail.com

    Created on Nov 25, 2016
    Revised on Aug 14, 2022
    '''

    axes = {}
    axes['dx'] = rho[0]

    axes['tc'] = option['colframe']
    axes['next'] = 'replace'  # needed?

    # make a radial grid
    if option['axismax'] == 0.0:
        maxrho = max(abs(rho))
    else:
        maxrho = option['axismax']

    # Determine default number of tick marks
    if option['overlay'] == 'off':
        ax.set_xlim(-maxrho, maxrho)
    xt = ax.get_xticks()
    ticks = sum(xt >= 0)

    # Check radial limits and ticks
    axes['rmin'] = 0;
    if option['axismax'] == 0.0:
        axes['rmax'] = xt[-1]
        option['axismax'] = axes['rmax']
    else:
        axes['rmax'] = option['axismax']
    rticks = np.amax(ticks - 1, axis=0)
    if rticks > 5:  # see if we can reduce the number
        if rticks % 2 == 0:
            rticks = rticks / 2
        elif rticks % 3 == 0:
            rticks = rticks / 3
    axes['rinc'] = (axes['rmax'] - axes['rmin']) / rticks
    tick = np.arange(axes['rmin'] + axes['rinc'],
                     axes['rmax'] + axes['rinc'],
                     axes['rinc'])

    if len(option['tickrms']) == 0:
        option['tickrms'] = tick
        option['rincrms'] = axes['rinc']
    if len(option['tickstd']) == 0:
        option['tickstd'] = tick
        option['rincstd'] = axes['rinc']

    return axes


def _overlay_taylor_diagram_circles(ax: matplotlib.axes.Axes, axes: dict,
                                    option: dict) -> None:
    '''
    Overlays circle contours on a Taylor diagram.
    
    Plots circle contours on a Taylor diagram to indicate root mean square 
    (RMS) and standard deviation values.
    
    INPUTS:
    ax     : matplotlib.axes.Axes object in which the Taylor diagram will be
            plotted
    axes   : data structure containing axes information for Taylor diagram
    option : data structure containing option values. (See 
            _get_taylor_diagram_options for more information.)
    option['colrms']       : RMS grid and tick labels color (Default: green)
    option['rincrms']      : Increment spacing for RMS grid
    option['stylerms']     : Linestyle of the RMS grid
    option['tickrms']      : RMS values to plot gridding circles from 
                            observation point
    option['tickRMSangle'] : Angle for RMS tick labels with the observation 
                            point (Default: 135 deg.)
    option['widthrms']     : Line width of the RMS grid

    option['colstd']       : STD grid and tick labels color (Default: black)
    option['colsstd']      : dictionary with two possible colors keys ('ticks',
                                'tick_labels') or None, if None then considers only the
                                value of 'colsstd' (Default: None)
    option['rincstd']      : Increment spacing for STD grid
    option['stylestd']     : Linestyle of the STD grid
    option['tickstd']      : STD values to plot gridding circles from origin
    option['tickstdangle'] : Angle for STD tick labels with the observation 
                            point (Default: .8)
    option['widthstd']     : Line width of the STD grid

    OUTPUTS:
    None

    See also _get_taylor_diagram_options

    Author: Andre D. L. Zanchetta (adapting Peter A. Rochford's code)
        adlzanchetta@gmail.com
    '''

    th = np.arange(0, 2 * np.pi, np.pi / 150)
    xunit = np.cos(th)
    yunit = np.sin(th)

    # now really force points on x/y axes to lie on them exactly
    inds = range(0, len(th), (len(th) - 1) // 4)
    xunit[inds[1:5:2]] = np.zeros(2)
    yunit[inds[0:6:2]] = np.zeros(3)

    # DRAW RMS CIRCLES:
    # ANGLE OF THE TICK LABELS
    if option['tickrmsangle'] > 0:
        tickRMSAngle = option['tickrmsangle']
    else:
        phi = np.arctan2(option['tickstd'][-1], axes['dx'])
        tickRMSAngle = 180 - np.rad2deg(phi)

    cst = np.cos(tickRMSAngle * np.pi / 180)
    snt = np.sin(tickRMSAngle * np.pi / 180)
    radius = np.sqrt(axes['dx'] ** 2 + axes['rmax'] ** 2 -
                     2 * axes['dx'] * axes['rmax'] * xunit)

    # Define label format
    labelFormat = '{' + option['rmslabelformat'] + '}'
    fontSize = matplotlib.rcParams.get('font.size') + 2

    for iradius in option['tickrms']:
        phi = th[np.where(radius >= iradius)]
        if len(phi) != 0:
            phi = phi[0]
            ig = np.where(iradius * np.cos(th) + axes['dx'] <=
                          axes['rmax'] * np.cos(phi))
            hhh = ax.plot(xunit[ig] * iradius + axes['dx'], yunit[ig] * iradius,
                          linestyle=option['stylerms'], color=option['colrms'],
                          linewidth=option['widthrms'])
            if option['showlabelsrms'] == 'on':
                rt = (iradius + option['rincrms'] / 20)
                if option['tickrmsangle'] > 90:
                    xtextpos = (rt + abs(cst) * axes['rinc'] / 5) * cst + axes['dx']
                    ytextpos = (rt + abs(cst) * axes['rinc'] / 5) * snt
                else:
                    xtextpos = rt * cst + axes['dx']
                    ytextpos = rt * snt

                ax.text(xtextpos, ytextpos, labelFormat.format(iradius),
                        horizontalalignment='center', verticalalignment='center',
                        color=option['colrms'], rotation=tickRMSAngle - 90,
                        fontsize=option['ticksizerms'])

    # DRAW STD CIRCLES:
    # draw radial circles
    grid_color = _get_from_dict_or_default(option, 'colstd', 'colsstd', 'grid')
    for i in option['tickstd']:
        hhh = ax.plot(xunit * i, yunit * i,
                      linestyle=option['stylestd'],
                      color=grid_color,
                      linewidth=option['widthstd'])
        del i

    # Set tick values for axes
    tickValues = []
    if option['showlabelsstd'] == 'on':
        if option['numberpanels'] == 2:
            tickValues = -option['tickstd'] + option['tickstd']
            tickValues.sort()
        else:
            tickValues = option['tickstd']

    ax.set_xticks(tickValues)

    hhh[0].set_linestyle('-')  # Make outermost STD circle solid

    # Draw circle for outer boundary
    i = option['axismax']
    hhh = ax.plot(xunit * i, yunit * i,
                  linestyle=option['stylestd'],
                  color=grid_color,
                  linewidth=option['widthstd'])

    return None


def _overlay_taylor_diagram_lines(ax: matplotlib.axes.Axes, axes: dict,
                                  option: dict) -> None:
    '''
    Overlay lines emanating from origin on a Taylor diagram.

    Plots lines emanating from origin to indicate correlation values (CORs) 

    It is a direct adaptation of the _overlay_taylor_diagram_lines() function
    for the screnarion in which the Taylor diagram is draw in an 
    matplotlib.axes.Axes object.

    INPUTS:
    ax     : matplotlib.axes.Axes object in which the Taylor diagram will be plotted
    axes   : data structure containing axes information for target diagram
    cax    : handle for plot axes
    option : data structure containing option values. (Refer to 
            _get_taylor_diagram_options function for more information.)
    option['colcor']        : CORs grid and tick labels color (Default: blue)
    option['colscor']       : dictionary with two possible colors as keys ('grid',
                                'tick_labels') or None, if None then considers only the
                                value of 'colscor' (Default: None)
    option['numberpanels']  : number of panels
    option['showlabelscor'] : Show or not the CORRELATION tick labels
    option['stylecor']      : Linestyle of the CORs grid
    option['tickcor']       : CORs values to plot lines from origin
    option['widthcor']      : Line width of the CORs grid

    OUTPUTS:
    None.
    
    Author: Andre D. L. Zanchetta (adapting Peter A. Rochford's code)
        adlzanchetta@gmail.com

    Created on Aug 14, 2022
    '''

    # Get common information
    corr = option['tickcor'][option['numberpanels'] - 1]
    th = np.arccos(corr)
    cst, snt = np.cos(th), np.sin(th)
    del th

    # DRAW CORRELATION LINES EMANATING FROM THE ORIGIN:
    cs = np.append(-1.0 * cst, cst)
    sn = np.append(-1.0 * snt, snt)
    lines_col = _get_from_dict_or_default(option, 'colcor', 'colscor', 'grid')
    for i, val in enumerate(cs):
        ax.plot([0, axes['rmax'] * cs[i]],
                [0, axes['rmax'] * sn[i]],
                linestyle=option['stylecor'],
                color=lines_col,
                linewidth=option['widthcor'])
        del i, val
    del lines_col, sn, cs

    # annotate them in correlation coefficient
    if option['showlabelscor'] == 'on':
        ticklabels_col = _get_from_dict_or_default(option, 'colcor', 'colscor', 'tick_labels')
        fontSize = matplotlib.rcParams.get('font.size')
        rt = 1.05 * axes['rmax']
        for i, cc in enumerate(corr):
            if option['numberpanels'] == 2:
                x = (1.05 + abs(cst[i]) / 30) * axes['rmax'] * cst[i]
            else:
                x = rt * cst[i]
            y = rt * snt[i]
            ax.text(x, y,
                    str(round(cc, 2)),
                    horizontalalignment='center',
                    color=ticklabels_col,
                    fontsize=option['ticksizecor'])
            del i, cc
        del fontSize, rt, ticklabels_col

    return None


def _plot_taylor_axes(ax: matplotlib.axes.Axes, axes: dict, option: dict) \
        -> list:
    '''
    Plot axes for Taylor diagram.
    
    Plots the x & y axes for a Taylor diagram using the information 
    provided in the AXES dictionary returned by the 
    _get_taylor_diagram_axes function.

    INPUTS:
    ax     : matplotlib.axes.Axes object in which the Taylor diagram will be plotted
    axes   : data structure containing axes information for Taylor diagram
    option : data structure containing option values. (Refer to 
            _get_taylor_diagram_options function for more information.)
    option['colcor']        : CORs grid and tick labels color (Default: blue)
    option['colscor']       : dictionary with two possible colors as keys ('grid',
                                'tick_labels') or None, if None then considers only the
                                value of 'colscor' (Default: None)
    option['colrms']        : RMS grid and tick labels color (Default: green)
    option['colstd']        : STD grid and tick labels color (Default: black)
    option['colsstd']       : dictionary with two possible colors keys ('ticks',
                                'tick_labels') or None, if None then considers only the
                                value of 'colsstd' (Default: None)
    option['labelrms']      : RMS axis label, e.g. 'RMSD'
    option['numberpanels']  : number of panels (quadrants) to use for Taylor
                                diagram
    option['tickrms']       : RMS values to plot gridding circles from
                                observation point
    option['titlecor']      : title for CORRELATION axis
    option['titlerms']      : title for RMS axis
    option['titlestd']      : title for STD axis
    option['titlecorshape'] : defines the shape of the label "correlation coefficient"
                                as either 'curved' or 'linear' (Default: 'curved')

    option['stdlabelsize']    : STD label size (Default 15)
    option['corlabelsize']    : correlation label size (Default: 15)
    option['rmslabelsize']    : RMS label size (Default: 15)

    option['ticksizestd']     : STD ticks size (Default: 13)
    option['ticksizecor']     : correlation ticks size (Default: 13)
    option['ticksizerms']     : RMS ticks size (Default: 13)

    OUTPUTS:
    ax: returns a list of handles of axis labels
    
    Authors:
    Peter A. Rochford
    rochford.peter1@gmail.com

    Andre D. L. Zanchetta (adapting Peter A. Rochford's code)
        adlzanchetta@gmail.com

    Created on Dec 3, 2016
    Revised on Aug 14, 2022
    '''

    axes_handles = []
    axlabweight = option['labelweight']
    fontSize = rcParams.get('font.size') + 2
    lineWidth = rcParams.get('lines.linewidth')
    fontFamily = rcParams.get('font.family')

    if option['numberpanels'] == 1:
        # Single panel

        if option['titlestd'] == 'on':
            color = _get_from_dict_or_default(
                option, 'colstd', 'colsstd', 'title')
            if option['normalizedstd']:
                handle = ax.set_ylabel('Normalized Standard Deviation',
                                       color=color,
                                       fontweight=axlabweight,
                                       fontsize=option['stdlabelsize'],
                                       fontfamily=fontFamily)
            else:
                handle = ax.set_ylabel('Standard Deviation',
                                       color=color,
                                       fontweight=axlabweight,
                                       fontsize=option['stdlabelsize'],
                                       fontfamily=fontFamily)
            axes_handles.append(handle)
            del color, handle

        # plot correlation title
        if option['titlecor'] == 'on':
            color = _get_from_dict_or_default(
                option, 'colcor', 'colscor', 'title')
            pos1 = 45
            lab = 'Correlation Coefficient'

            if option['titlecorshape'] == 'curved':
                DA = 15
                c = np.fliplr([np.linspace(pos1 - DA, pos1 + DA, len(lab))])[0]
                dd = 1.1 * axes['rmax']
                for ii, ith in enumerate(c):
                    cur_x = dd * np.cos(ith * np.pi / 180)
                    cur_y = dd * np.sin(ith * np.pi / 180)
                    # print("%s: %.03f, %.03f, %.03f" % (lab[ii], cur_x, cur_y, ith))
                    handle = ax.text(dd * np.cos(ith * np.pi / 180),
                                     dd * np.sin(ith * np.pi / 180),
                                     lab[ii])
                    handle.set(rotation=ith - 90,
                               color=color,
                               horizontalalignment='center',
                               verticalalignment='bottom',
                               fontsize=option['corlabelsize'],
                               fontfamily=fontFamily,
                               fontweight=axlabweight)
                    axes_handles.append(handle)
                    del ii, ith, handle
                del DA, c, dd

            elif option['titlecorshape'] == 'linear':
                pos_x_y = 1.13 * axes['rmax'] * np.cos(pos1 * np.pi / 180)
                handle = ax.text(pos_x_y, pos_x_y, "Correlation Coefficient")
                handle.set(rotation=-45,
                           color=color,
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontsize=option['corlabelsize'],
                           fontfamily=fontFamily,
                           fontweight=axlabweight)
                del pos_x_y, handle

            else:
                raise ValueError("Invalid value for 'titlecorshape': %s" %
                                 option['titlecorshape'])

            del color, pos1, lab

        if option['titlerms'] == 'on':
            lab = option['labelrms']
            pos1 = option['titlermsdangle'];
            DA = 10
            c = np.fliplr([np.linspace(pos1 - DA, pos1 + DA, len(lab))])[0]
            if option['tickrms'][0] > 0:
                dd = 0.8 * option['tickrms'][0] + 0.2 * option['tickrms'][1]
            else:
                dd = 0.8 * option['tickrms'][1] + 0.2 * option['tickrms'][2]

            # Adjust spacing of label letters if on too small an arc
            posFraction = dd / axes['rmax']
            if posFraction < 0.35:
                DA = 2 * DA
                c = np.fliplr([np.linspace(pos1 - DA, pos1 + DA, len(lab))])[0]

            # Write label in a circular arc               
            for ii, ith in enumerate(c):
                xtextpos = axes['dx'] + dd * np.cos(ith * np.pi / 180)
                ytextpos = dd * np.sin(ith * np.pi / 180)
                handle = ax.text(xtextpos, ytextpos, lab[ii])
                handle.set(rotation=ith - 90, color=option['colrms'],
                           horizontalalignment='center',
                           verticalalignment='top',
                           fontsize=option['rmslabelsize'], fontweight=axlabweight)
                axes_handles.append(handle)

    else:
        # Double panel

        if option['titlestd'] == 'on':
            color = _get_from_dict_or_default(
                option, 'colstd', 'colsstd', 'title')
            handle = ax.set_xlabel('Standard Deviation',
                                   color=color,
                                   fontweight=axlabweight,
                                   fontsize=option['stdlabelsize'])

            axes_handles.append(handle)
            del color, handle

        if option['titlecor'] == 'on':
            color = _get_from_dict_or_default(
                option, 'colcor', 'colscor', 'title')
            pos1 = 90;
            DA = 25;
            lab = 'Correlation Coefficient'
            c = np.fliplr([np.linspace(pos1 - DA, pos1 + DA, len(lab))])[0]
            dd = 1.1 * axes['rmax']

            # Write label in a circular arc
            for ii, ith in enumerate(c):
                handle = ax.text(dd * np.cos(ith * np.pi / 180),
                                 dd * np.sin(ith * np.pi / 180), lab[ii])
                handle.set(rotation=ith - 90, color=color,
                           horizontalalignment='center',
                           verticalalignment='bottom',
                           fontsize=option['corlabelsize'],
                           fontweight=axlabweight)
                axes_handles.append(handle)

                del ii, ith, handle
            del color, pos1, DA, lab, c, dd

        if option['titlerms'] == 'on':
            lab = option['labelrms']
            pos1 = option['titlermsdangle'];
            DA = 10
            c = np.fliplr([np.linspace(pos1 - DA, pos1 + DA, len(lab))])[0]
            if option['tickrms'][0] > 0:
                dd = 0.7 * option['tickrms'][0] + 0.3 * option['tickrms'][1]
            else:
                dd = 0.7 * option['tickrms'][1] + 0.3 * option['tickrms'][2]

            # Adjust spacing of label letters if on too small an arc
            posFraction = dd / axes['rmax']
            if posFraction < 0.35:
                DA = 2 * DA
                c = np.fliplr([np.linspace(pos1 - DA, pos1 + DA, len(lab))])[0]

            for ii, ith in enumerate(c):
                xtextpos = axes['dx'] + dd * np.cos(ith * np.pi / 180)
                ytextpos = dd * np.sin(ith * np.pi / 180)
                handle = ax.text(xtextpos, ytextpos, lab[ii])
                handle.set(rotation=ith - 90, color=option['colrms'],
                           horizontalalignment='center',
                           verticalalignment='bottom',
                           fontsize=option['rmslabelsize'],
                           fontweight=axlabweight)
                axes_handles.append(handle)

    #  Set color of tick labels to that specified for STD contours
    labels_color = _get_from_dict_or_default(option, 'colstd', 'colsstd', 'tick_labels')
    ticks_color = _get_from_dict_or_default(option, 'colstd', 'colsstd', 'ticks')
    ax.tick_params(axis='both', color=ticks_color, labelcolor=labels_color)
    del labels_color, ticks_color

    # VARIOUS ADJUSTMENTS TO THE PLOT:
    ax.set_aspect('equal')
    ax.set_frame_on(None)

    # set axes limits, set ticks, and draw axes lines
    ylabel = []
    if option['numberpanels'] == 2:
        xtick = [-option['tickstd'], option['tickstd']]
        if 0 in option['tickstd']:
            xtick = np.concatenate((-option['tickstd'][1:], option['tickstd']), axis=None)
        else:
            xtick = np.concatenate((-option['tickstd'][0:], 0, option['tickstd']), axis=None)
        xtick = np.sort(xtick)

        # Set x tick labels
        xlabel = [];
        for i in range(len(xtick)):
            if xtick[i] == 0:
                label = '0'
            else:
                label = _get_axis_tick_label(abs(xtick[i]))
            xlabel.append(label)

        ax.set_xticks(xtick)
        ax.set_xticklabels(xlabel, fontfamily=fontFamily, fontsize=option['ticksizestd'])

        axislim = [axes['rmax'] * x for x in [-1, 1, 0, 1]]
        ax.set_xlim(axislim[0:2])
        ax.set_ylim(axislim[2:])
        ax.plot([-axes['rmax'], axes['rmax']], [0, 0],
                color=axes['tc'], linewidth=lineWidth + 1)
        ax.plot([0, 0], [0, axes['rmax']], color=axes['tc'])

        # hide y-axis line
        ax.axes.get_yaxis().set_visible(False)
    else:
        ytick = ax.get_yticks()
        ytick = list(filter(lambda x: x >= 0 and x <= axes['rmax'], ytick))
        axislim = [axes['rmax'] * x for x in [0, 1, 0, 1]]
        ax.set_xlim(axislim[0:2])
        ax.set_ylim(axislim[2:])

        # Set y tick labels
        for i in range(len(ytick)):
            label = _get_axis_tick_label(ytick[i])
            ylabel.append(label)

        ax.set_xticks(ytick)
        ax.set_yticks(ytick)
        ax.set_xticklabels(ylabel, fontfamily=fontFamily, fontsize=option['ticksizestd'])
        ax.set_yticklabels(ylabel, fontfamily=fontFamily, fontsize=option['ticksizestd'])

        ax.plot([0,
                 axes['rmax']], [0, 0],
                color=axes['tc'],
                linewidth=lineWidth + 2)
        ax.plot([0, 0],
                [0, axes['rmax']],
                color=axes['tc'],
                linewidth=lineWidth + 1)

    return axes_handles


def _plot_taylor_obs(ax: matplotlib.axes.Axes, axes_handle: list, obsSTD,
                     axes_info: dict, option: dict) -> None:
    '''
    Plots observation STD on Taylor diagram.
    
    Optionally plots a marker on the x-axis indicating observation STD, 
    a label for this point, and a contour circle indicating the STD 
    value.
    
    INPUTS:
    ax     : the matplotlib.axes.Axes in which the Taylor diagram will be plotted
    obsSTD : observation standard deviation
    axes   : axes information of Taylor diagram
    option : data structure containing option values. (Refer to 
            get_taylor_diagram_subplot_options() function for more information.)
    option['colobs']       : color for observation labels (Default : magenta)
    option['markerobs']    : marker to use for x-axis indicating observed STD
    option['styleobs']     : line style for observation grid line
    option['titleobs']     : label for observation point label (Default: '')
    option['widthobs']     : linewidth for observation grid line (Default: .8)

    OUTPUTS:
    None
    
    Authors:
    Peter A. Rochford
    rochford.peter1@gmail.com

    Andre D. L. Zanchetta (adapting Peter A. Rochford's code)
        adlzanchetta@gmail.com

    Created on Feb 19, 2017
    Revised on Aug 14, 2022
    '''

    if option['markerobs'] != 'none':
        # Display marker on x-axis indicating observed STD
        # markersize = option['markersize'] - 4
        yobsSTD = 0.001 * axes_info['rmax'] - axes_info['rmin']
        ax.plot(obsSTD, yobsSTD, option['markerobs'], color=option['colobs'],
                markersize=option['markersizeobs'], markerfacecolor=option['colobs'],
                markeredgecolor=option['colobs'],
                linewidth=1.0, clip_on=False);

    if option['titleobs'] != '':
        # Put label below the marker
        labelsize = axes_handle[0].get_fontsize()  # get label size of STD axes
        ax.set_xlabel(option['titleobs'], color=option['colobs'],
                      fontweight='bold', fontsize=labelsize)
        xlabelh = ax.xaxis.get_label()
        xypos = xlabelh.get_position()
        markerpos = ax.transLimits.transform((obsSTD, 0))
        xlabelh.set_position([markerpos[0], xypos[1]])
        xlabelh.set_horizontalalignment('center')

    if option['styleobs'] != '':
        # Draw circle for observation STD
        theta = np.arange(0, 2 * np.pi, np.pi / 150)
        xunit = obsSTD * np.cos(theta)
        yunit = obsSTD * np.sin(theta)
        ax.plot(xunit, yunit, linestyle=option['styleobs'],
                color=option['colobs'], linewidth=option['widthobs'])


def _get_axis_tick_label(value):
    '''
    Get label for number on axis without trailing zeros.
    
    Converts a numerical value to a string for labeling the tick increments along an 
    axis in plots. This function removes trailing zeros in numerical values that may
    occur due to floating point precision. For example, a floating point number such as
    
    59.400000000000006
    
    will be returned as a string 
    
    '59.4'
    
    without the trailing insignificant figures.
        
    INPUTS:
    value : value to be displayed at tick increment on axis
    
    OUTPUTS:
    label: string containing number to display below tick increment on axis
    '''
    number_digits = 0
    if not _use_sci_notation(value):
        label = str(value)

        # Get substring after period
        trailing = label.partition('.')[2]
        number_sigfig = 0
        if len(trailing) > 0:
            # Find number of non-zero digits after decimal
            number_sigfig = 1
            before = trailing[0]
            number_digits = 1
            go = True
            while go and number_digits < len(trailing):
                if trailing[number_digits] == before:
                    number_sigfig = number_digits - 1
                    if (number_sigfig > 5): go = False
                else:
                    before = trailing[number_digits]
                    number_sigfig = number_digits - 1
                number_digits += 1

        if number_digits == len(trailing): number_sigfig = number_digits

        # Round up the number to desired significant figures
        label = str(round(value, number_sigfig))
    else:
        label = "{:.1e}".format(value)

    return label


def _use_sci_notation(value):
    '''
    Boolean function to determine if scientific notation to be used for value

    Input:
    Value : value to be tested

    Return:
        True if absolute value is > 100 or < 1.e-3
        False otherwise

    Author: Peter A. Rochford
        Symplectic, LLC

    Created on May 10, 2022
    '''
    if (abs(value) > 0 and abs(value) < 1e-3):
        return True
    else:
        return False


def check_on_off(value):
    '''
    Check whether variable contains a value of 'on', 'off', True, or False.
    Returns an error if neither for the first two, and sets True to 'on',
    and False to 'off'. The 'on' and 'off' can be provided in any combination
    of upper and lower case letters.
    
    INPUTS:
    value : string or boolean to check
    
    OUTPUTS:
    None.

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    '''

    if isinstance(value, str):
        lowcase = value.lower()
        if lowcase == 'off':
            return lowcase
        elif lowcase == 'on':
            return lowcase
        else:
            raise ValueError('Invalid value: ' + str(value))
    elif isinstance(value, bool):
        if value == False:
            value = 'off'
        elif value == True:
            value = 'on'
    else:
        raise ValueError('Invalid value: ' + str(value))

    return value


def generate_markers(data_names):
    import itertools
    import matplotlib.colors as mcolors
    markers = {}
    # add colors and symbols
    hex_colors = ['#4C6EF5', '#F9C74F', '#90BE6D', '#5BC0EB', '#43AA8B', '#F3722C', '#855456', '#F9AFAF', '#F8961E'
        , '#277DA1', '#5A189A']
    colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])
    # colors = itertools.cycle(["r", "b", "g", "c", "m", "y", "k", "orange", "purple", "brown", "pink", "gray"])  # Cycle through colors
    symbols = itertools.cycle(["+", ".", "o", "*", "x", "s", "D", "^", "v", ">", "<", "p"])  # Cycle through symbols

    for name in data_names:
        if not option['faceColor']:
            faceColor = next(colors)
        else:
            faceColor = option['faceColor']
        markers[name] = {
            "labelColor": next(colors),
            "symbol": next(symbols),
            "size": 10,
            "faceColor": "w",
            "edgeColor": next(colors),
        }
    return markers
