import math
import numbers
import os
import warnings
from array import array
from typing import Union

import matplotlib
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


def make_scenarios_comparison_Target_Diagram(basedir, evaluation_item, bias, crmsd, rmsd, ref_source, sim_sources, option):
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import rcParams
    import os

    font = {'family': option['font']}
    matplotlib.rc('font', **font)

    params = {'backend': 'ps',
              'axes.linewidth': option['axes_linewidth'],
              'font.size': option['fontsize'],
              'xtick.direction': 'out',
              'xtick.labelsize': option['xticksize'],
              'ytick.direction': 'out',
              'ytick.labelsize': option['yticksize'],
              'savefig.bbox': 'tight',
              'axes.unicode_minus': False,
              'text.usetex': False}
    rcParams.update(params)

    fig, ax = plt.subplots(figsize=(option['x_wise'], option['y_wise']))

    option['MARKERS'] = generate_markers(sim_sources, option)


    target_diagram(bias,crmsd,rmsd, markerLabel = sim_sources,
                   markers=option['MARKERS'],
                   markerLegend=option['markerLegend'],

                   normalized=option['Normalized'],

                   circlecolor=option['circlecolor'],
                   circlestyle=option['circlestyle'],
                   circleLineWidth=option['widthcircle'],
                   circlelabelsize=option['circlelabelsize'],

                   legend={option['set_legend'], option['bbox_to_anchor_x'], option['bbox_to_anchor_y']}
                   )

    #if not option['title']:
    #    option['title'] = evaluation_item.replace('_', " ")
    # ax.set_title(option['title'], fontsize=option['title_size'], pad=30)

    output_file_path = os.path.join(f'{basedir}', f'Target_diagram_{evaluation_item}_{ref_source}.{option["saving_format"]}')
    plt.savefig(output_file_path, format=f'{option["saving_format"]}', dpi=option['dpi'], bbox_inches='tight')


def target_diagram(*args, **kwargs):
    '''
    Plot a target diagram from statistics of different series.
    
    target_diagram(Bs,RMSDs,RMSDz,keyword=value)
    
    The first 3 arguments must be the inputs as described below followed by
    keywords in the format OPTION = value. An example call to the function 
    would be:
    
    target_diagram(Bs,RMSDs,RMSDz,markerdisplayed='marker')
    
    INPUTS:
    Bs    : Bias (B) or Normalized Bias (B*). Plotted along y-axis
            as "Bias".
    RMSDs : unbiased Root-Mean-Square Difference (RMSD') or normalized
            unbiased Root-Mean-Square Difference (RMSD*'). Plotted along 
            x-axis as "uRMSD".
    RMSDz : total Root-Mean-Square Difference (RMSD). Labeled on plot as "RMSD".
    
    OUTPUTS:
    None.
    
    LIST OF OPTIONS:
    For an exhaustive list of options to customize your diagram, call the 
    function without arguments at a Python command line:
    % python
    >>> import skill_metrics as sm
    >>> sm.target_diagram()
    
    Reference:

    Jolliff, J. K., J. C. Kindle, I. Shulman, B. Penta, M. Friedrichs, 
    R. Helber, and R. Arnone (2009), Skill assessment for coupled 
    biological/physical models of marine systems, J. Mar. Sys., 76(1-2),
    64-82, doi:10.1016/j.jmarsys.2008.05.014

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Nov 25, 2016
    '''

    # Check for no arguments
    if len(args) == 0: return
        
    # Process arguments (if given)
    ax, Bs, RMSDs, RMSDz = _get_target_diagram_arguments(*args)

    # Get options
    option = get_target_diagram_options(**kwargs)

    #  Get axis values for plot
    axes = get_target_diagram_axes(RMSDs,Bs,option)

    # Overlay circles
    overlay_target_diagram_circles(ax, option)

    # Modify axes for target diagram (no overlay)
    if option['overlay'] == 'off':
        axes_handles = plot_target_axes(ax, axes, option)

    # Plot data points
    lowcase = option['markerdisplayed'].lower()
    if lowcase == 'marker':
        plot_pattern_diagram_markers(ax,RMSDs,Bs,option)
    elif lowcase == 'colorbar':
        plot_pattern_diagram_colorbar(ax,RMSDs,Bs,RMSDz,option)
    else:
        raise ValueError('Unrecognized option: ' + 
                        option['markerdisplayed'])

def _display_target_diagram_options():
    '''
    Displays available options for TARGET_DIAGRAM function.
    '''

    _disp('General options:')
    _dispopt("'colormap'","'on'/ 'off' (default): "  + 
        "Switch to map color shading of markers to colormap ('on')\n\t\t"  +
        "or min to max range of RMSDz values ('off').")
    _dispopt("'overlay'","'on' / 'off' (default): " + 
            'Switch to overlay current statistics on target diagram. ' +
            '\n\t\tOnly markers will be displayed.')
    _disp("OPTIONS when 'colormap' == 'on'")
    _dispopt("'cmap'","Choice of colormap. (Default: 'jet')")
    _dispopt("'cmap_marker'","Marker to use with colormap (Default: 'd')")
    _dispopt("'cmap_vmax'","Maximum range of colormap (Default: None)")
    _dispopt("'cmap_vmax'","Minimum range of colormap (Default: None)")
    _disp('')
    
    _disp('Marker options:')
    _dispopt("'MarkerDisplayed'", 
        "'marker' (default): Experiments are represented by individual symbols\n\t\t" +
        "'colorBar': Experiments are represented by a color described " + 
        'in a colorbar')
    
    _disp("OPTIONS when 'MarkerDisplayed' == 'marker'")
    _dispopt("'markerColor'",'Single color to use for all markers'  +
        ' (Default: None)')
    _dispopt("'markerColors'","Dictionary with two colors as keys ('face', 'edge')" +
            "or None." + "\n\t\t" + 
            "If None or 'markerlegend' == 'on' then considers only the value of " + 
            "'markerColor'. (Default: None)")
    _dispopt("'markerLabel'",'Labels for markers')
    _dispopt("'markerLabelColor'",'Marker label color (Default: black)')
    _dispopt("'markerLayout'","Matrix layout for markers in legend [nrow, ncolumn]." + "\n\t\t" + 
            "(Default: [15, no. markers/15])'")
    _dispopt("'markerLegend'","'on' / 'off' (default): Use legend for markers'")

    _dispopt("'markers'",'Dictionary providing individual control of the marker ' +
            'label, label color, symbol, size, face color, and edge color'  +
        ' (Default: none)')

    _dispopt("'markerSize'",'Marker size (Default: 10)')
    _dispopt("'markerSymbol'","Marker symbol (Default: '.')")
    
    _disp("OPTIONS when 'MarkerDisplayed' == 'colorbar'")
    _dispopt("'cmapZData'","Data values to use for " +
            'color mapping of markers, e.g. RMSD or BIAS.\n\t\t' +
            '(Used to make range of RMSDs values appear above color bar.)')
    _dispopt("'locationColorBar'","Location for the colorbar, 'NorthOutside' " +
            "or 'EastOutside'")
    _dispopt("'titleColorBar'",'Title of the colorbar.')
    _disp('')
    
    _disp('Axes options:')
    _dispopt("'axismax'",'Maximum for the Bias & uRMSD axis')
    _dispopt("'colFrame'",'Color for the y and x spines')
    _dispopt("'equalAxes'","'on' (default) / 'off': Set axes to be equal")
    _dispopt("'labelWeight'","Weight of the x & y axis labels")
    _dispopt("'ticks'",'Define tick positions ' +
            '(default is that used by axis function)')
    _dispopt("'xtickLabelPos'",'position of the tick labels ' +
            'along the x-axis (empty by default)')
    _dispopt("'ytickLabelPos'",'position of the tick labels ' +
            'along the y-axis (empty by default)')
    _disp('')
    
    _disp('Diagram options:')
    _dispopt("'alpha'","Blending of symbol face color (0.0 transparent through 1.0 opaque)" +
            "\n\t\t" + "(Default: 1.0)")
    _dispopt("'circles'",'Define the radii of circles to draw ' +
            '(default of (maximum RMSDs)*[.7 1], [.7 1] when normalized diagram)')
    _dispopt("'circleColor'",'Circle line color specification (default None)')
    _dispopt("'circleCols'","Dictionary with two possible colors keys ('ticks'," +
            "'tick_labels')" +
            "\n\t\t or None, if None then considers only the value of 'circlecolor'" +
            "(Default: None)")
    _dispopt("'circleLineSpec'",'Circle line specification (default ' +
            "dashed black, '--k')")
    _dispopt("'circleLineWidth'",'Circle line width')
    _dispopt("'circleStyle'",'Line style for circles, e.g. "--" (Default: None)')
    _dispopt("'normalized'","'on' / 'off' (default): normalized target diagram")
    _dispopt("'obsUncertainty'",'Observational Uncertainty (default of 0)')
    
    _disp('Plotting Options from File:')
    _dispopt("'target_options_file'","name of CSV file containing values for optional" +
            " arguments" +
            "\n\t\t" + "of the target_diagram function. If no file suffix is given," +
            "\n\t\t" + "a '.csv' is assumed. (Default: empty string '')")

def _disp(text):
    print(text)

def _dispopt(optname,optval):
    '''
    Displays option name and values

    This is a support function for the DISPLAY_TARGET_DIAGRAM_OPTIONS function.
    It displays the option name OPTNAME on a line by itself followed by its 
    value OPTVAL on the following line.
    '''

    _disp('\t%s' % optname)
    _disp('\t\t%s' % optval)

def _ensure_np_array_or_die(v, label: str) -> np.ndarray:
    '''
    Check variable has is correct data type.
    
    v: Value to be ensured
    label: Python data type
    '''

    ret_v = v
    if isinstance(ret_v, array):
        ret_v = np.array(v)
    if isinstance(ret_v, numbers.Number):
        ret_v = np.array(v, ndmin=1)
    if not isinstance(ret_v, np.ndarray):
        raise ValueError('Argument {0} is not a numeric array: {1}'.format(label, v))
    return ret_v

def _get_target_diagram_arguments(*args):
    '''
    Get arguments for target_diagram function.
    
    Retrieves the arguments supplied to the TARGET_DIAGRAM function as
    arguments and displays the optional arguments if none are supplied.
    Otherwise, tests the first 3 arguments are numeric quantities and 
    returns their values.
    
    INPUTS:
    args : variable-length input argument list
    
    OUTPUTS:
    Bs    : Bias (B) or Normalized Bias (B*). Plotted along y-axis
            as "Bias".
    RMSDs : unbiased Root-Mean-Square Difference (RMSD') or normalized
            unbiased Root-Mean-Square Difference (RMSD*'). Plotted along 
            x-axis as "uRMSD".
    RMSDz : total Root-Mean-Square Difference (RMSD). Labeled on plot as "RMSD".
    '''

    # Check amount of values provided and display options list if needed

    nargin = len(args)
    if nargin == 0:
        # Display options list
        _display_target_diagram_options()
        return [], [], [], []
    elif nargin == 3:
        bs, rmsds, rmsdz = args
        CAX = plt.gca()
    elif nargin == 4:
        CAX, bs, rmsds, rmsdz = args
        if not hasattr(CAX, 'axes'):
            raise ValueError('First argument must be a matplotlib axes.')
    else:
        raise ValueError('Must supply 3 or 4 arguments.')
    del nargin
        
    # Check data validity
    Bs = _ensure_np_array_or_die(bs, "Bs")
    RMSDs = _ensure_np_array_or_die(rmsds, "RMSDs")
    RMSDz = _ensure_np_array_or_die(rmsdz, "RMSDz")

    return CAX, Bs, RMSDs, RMSDz


def plot_target_axes(ax: matplotlib.axes.Axes, axes: dict, option: dict) -> list:
    '''
    Plot axes for target diagram.
    
    Plots the x & y axes for a target diagram using the information 
    provided in the AXES dictionary returned by the 
    GET_TARGET_DIAGRAM_AXES function.
    
    INPUTS:
    axes   : dictionary containing axes information for target diagram
    
    OUTPUTS:
    None

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    Hello
    '''
    
    class ScalarFormatterClass(ScalarFormatter):
        def _set_format(self):
            self.format = "%1.1f"
    
    class Labeloffset():
        def __init__(self,  ax, label="", axis="y"):
            self.axis = {"y":ax.yaxis, "x":ax.xaxis}[axis]
            self.label=label
            ax.callbacks.connect(axis+'lim_changed', self.update)
            ax.figure.canvas.draw()
            self.update(None)

    def update(self, lim):
        fmt = self.axis.get_major_formatter()
        self.axis.offsetText.set_visible(False)
        self.axis.set_label_text(self.label + " ("+ fmt.get_offset()+")" )
        print(fmt.get_offset())
    
    axes_handles = []
    fontFamily = rcParams.get('font.family')

    # Center axes location by moving spines of bounding box
    # Note: Center axes location not available in matplotlib
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # Make axes square
    ax.set_aspect('equal')

    # Set new ticks and tick labels
    ax.set_xticks(axes['xtick'])
    ax.set_xticklabels(axes['xlabel'], fontfamily=fontFamily)
    ax.set_yticks(axes['ytick'])
    ax.set_yticklabels(axes['ylabel'], fontfamily=fontFamily)
    
    # Set axes limits
    axislim = [axes['xtick'][0], axes['xtick'][-1], axes['ytick'][0], axes['ytick'][-1]]
    ax.set_xlim(axislim[0:2])
    ax.set_ylim(axislim[2:])

    # Label x-axis
    # fontSize = matplotlib.rcParams.get('font.size')
    fontSize = option['circlelabelsize']
    xpos = axes['xtick'][-1] + 2*axes['xtick'][-1]/30
    ypos = axes['xtick'][-1]/30
    if axes['xoffset'] == 'None':
        ax.set_xlabel('uRMSD', fontsize = fontSize)
    else:
        ax.set_xlabel('uRMSD' + '\n(' + axes['xoffset'] + ')', fontsize = fontSize)

    xlabelh = ax.xaxis.get_label()
    xlabelh.set_horizontalalignment('left')
    ax.xaxis.set_label_coords(xpos, ypos, transform=ax.transData)
    ax.tick_params(axis='x', direction='in') # have ticks above axis
    
    # Label y-axis
    xpos = 0
    ypos = axes['ytick'][-1] + 2*axes['ytick'][-1]/30
    if axes['yoffset'] == 'None':        
        ax.set_ylabel('Bias ', fontsize = fontSize, rotation=0)
    else:
        ax.set_ylabel('Bias ' + '(' + axes['yoffset'] + ')', fontsize = fontSize, rotation=0)

    ylabelh = ax.yaxis.get_label()
    ylabelh.set_horizontalalignment('center')
    ax.yaxis.set_label_coords(xpos, ypos, transform=ax.transData)
    ax.tick_params(axis='y', direction='in') # have ticks on right side of axis
    
    # Set axes line width
    lineWidth = rcParams.get('lines.linewidth')
    ax.spines['left'].set_linewidth(lineWidth)
    ax.spines['bottom'].set_linewidth(lineWidth)
    
    return axes_handles

def plot_pattern_diagram_markers(ax: matplotlib.axes.Axes, X, Y, option: dict):
    '''
    Plots color markers on a pattern diagram in the provided subplot axis.
    
    Plots color markers on a target diagram according their (X,Y) 
    locations. The symbols and colors are chosen automatically with a 
    limit of 70 symbol & color combinations.
    
    The color bar is titled using the content of option['titleColorBar'] 
    (if non-empty string).

    It is a direct adaptation of the plot_pattern_diagram_markers() function
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
                            'taylor: No. labels=' + str(numberLabel+1) + ' < No. markers=' +
                            str(len(X)+1))
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
            marker, markercolor = get_default_markers(X, option)
        
            # Plot markers at data points
            labelcolor = []
            markerlabel = []
            for i, xval in enumerate(X):
                if abs(X[i]) <= limit and abs(Y[i]) <= limit:
                    h = ax.plot(X[i],Y[i],marker[i], markersize = markerSize,
                        markerfacecolor = markercolor[i],
                        markeredgecolor = markercolor[i][0:3] + (1.0,),
                        markeredgewidth = 2)
                    hp += tuple(h)
                    labelcolor.append(option['markerlabelcolor'])
                    markerlabel.append(option['markerlabel'][i])

        else:
            # Obtain markers from option['markers']
            labels, labelcolor, marker, markersize, markerfacecolor, markeredgecolor = \
                get_single_markers(option['markers'])
        
            # Plot markers at data points
            markerlabel = []
            for i, xval in enumerate(X):
                if abs(X[i]) <= limit and abs(Y[i]) <= limit:
                    h = ax.plot(X[i],Y[i],marker[i], markersize = markersize[i],
                        markerfacecolor = markerfacecolor[i],
                        markeredgecolor = markeredgecolor[i],
                        markeredgewidth = 2)
                    hp += tuple(h)
                    markerlabel.append(labels[i])

        # Add legend
        if len(markerlabel) == 0:
            warnings.warn('No markers within axis limit ranges.')
        else:
            add_legend(markerlabel, labelcolor, option, rgba, markerSize, fontSize, hp)
    else:
        # Plot markers as dots of a single color with accompanying labels

        
        # Plot markers at data points
        limit = option['axismax']

        # Define edge and face colors of the markers
        edge_color = get_from_dict_or_default(option, 'markercolor', 'markercolors', 'edge')
        if edge_color is None: edge_color = 'r'
        face_color = get_from_dict_or_default(option, 'markercolor', 'markercolors', 'face')
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
            add_legend(markerlabel, labelcolor, option, marker_label_color, markerSize, fontSize)

def get_single_markers(markers: dict):
    #def get_single_markers(markers: dict) -> tuple[list, list, list, list, list, list]: #fails with Python 3.6
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
        marker.append(SymbolColor)
        markersize.append(markers[key]['size'])
        markerfacecolor.append(color)
        markeredgecolor.append(markers[key]['edgeColor'])
        markerlabel.append(key) # store label
        labelcolor.append(markers[key]['labelColor'])

    return markerlabel, labelcolor, marker, markersize, markerfacecolor, markeredgecolor

def get_default_markers(X, option: dict):
    #def get_default_markers(X, option: dict) -> tuple[list, list]: #fails with Python 3.6
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
    kind = ['+','o','x','s','d','^','v','p','h','*']
    colorm = ['r','b','g','c','m','y','k','gray']
    if len(X) > 80:
        _disp('You must introduce new markers to plot more than 70 cases.')
        _disp('The ''marker'' character array need to be extended inside the code.')
    
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

def add_legend(markerLabel, labelcolor, option, rgba, markerSize, fontSize, hp = []):
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
            if option['legend']['set_legend']:
                leg = plt.legend(hp, markerlabel, loc='upper right',
                                 fontsize=fontSize, numpoints=1,
                                 bbox_to_anchor=(option['legend']['bbox_to_anchor_x'], option['legend']['bbox_to_anchor_y']))
            else:
                leg = plt.legend(hp, markerlabel, loc='upper right',
                                 fontsize=fontSize, numpoints=1,
                                 bbox_to_anchor=(1.55, 1.05))
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
            if option['legend']['set_legend']:
                loc = (option['legend']['bbox_to_anchor_x'], option['legend']['bbox_to_anchor_y'])
            else:
                if 'circlelinespec' in option:
                    loc = (1.2, 0.25)
                else:
                    loc = (1.1, 0.25)
            leg = plt.legend(hp, markerlabel, loc = loc, fontsize = fontSize,
                            numpoints=1, ncol = ncol)

    elif type(markerLabel) is dict:
        
        # Add legend using labels provided as dictionary
            
        # Define legend elements
        legend_elements = []
        for key, value in markerLabel.items():
            legend_object = Line2D([0], [0], marker='.', markersize = markerSize,
                markerfacecolor = rgba, markeredgecolor = value, label=key, linestyle='')
            legend_elements.append(legend_object)

        # Put legend in a default location
        leg = plt.legend(handles=legend_elements, loc = 'upper right',
                            fontsize = fontSize, numpoints=1,
                            bbox_to_anchor=(1.2,1.0))

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

def get_from_dict_or_default(options: dict, default_key: str, dict_key: str, key_key: str):
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

def _check_dict_with_keys(variable_name: str, dict_obj: Union[dict, None],
                        accepted_keys: set, or_none: bool = False) -> None:
    """
    Check if an argument in the form of dictionary has valid keys.
    :return: None. Raise 'ValueError' if evaluated variable is considered invalid. 
    """
    
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

def _circle_color_style(option : dict) -> dict:
    '''
    Set color and style of grid circles from option['circlecolor'] and
    option['circlestyle'] 
    '''
    # decipher into color and style components
    if option['circlelinespec'][-1].isalpha():
        option['circlecolor'] = option['circlelinespec'][-1]
        option['circlestyle'] = option['circlelinespec'][0:-1]
    else:
        option['circlecolor'] = option['circlelinespec'][0]
        option['circlestyle'] = option['circlelinespec'][1:]
    
    return option

def is_int(element):
    '''
    Check if variable is an integer. 
    '''
    try:
        int(element)
        return True
    except ValueError:
        return False

def is_float(element):
    '''
    Check if variable is a float. 
    '''
    try:
        float(element)
        return True
    except ValueError:
        return False
    
def is_list_in_string(element):
    '''
    Check if variable is list provided as string 
    '''
    return bool(re.search(r'\[|\]', element))

def _default_options() -> dict:
    '''
    Set default optional arguments for target_diagram function.
    
    Sets the default optional arguments for the TARGET_DIAGRAM 
    function in an OPTION dictionary. Default values are 
    assigned to selected optional arguments. 
    
    INPUTS:
    None
        
    OUTPUTS:
    option : dictionary containing option values
    option : dictionary containing option values. (Refer to 
            display_target_diagram_options function for more information.)
    option['alpha']           : blending of symbol face color (0.0 
                                transparent through 1.0 opaque). (Default : 1.0)
    option['axismax']         : maximum for the Bias & uRMSD axis
    option['circlecolor']     : circle line color specification (default None)
    option['circlecols']      : dictionary with two possible colors keys ('ticks',
                                'tick_labels') or None, if None then considers only the
                                value of 'circlecolor' (Default: None)
    option['circlelinespec']  : circle line specification (default dashed 
                                black, '--k')
    option['circlelinewidth'] : circle line width specification (default 0.5)
    option['circles']         : radii of circles to draw to indicate 
                                isopleths of standard deviation (empty by default)
    option['circlestyle']     : line style for circles (Default: None)
    option['circlelabelsize'] : circle labels (default 12)

    option['cmap']            : Choice of colormap. (Default : 'jet')
    option['cmap_vmin']       : minimum range of colormap (Default : None)
    option['cmap_vmax']       : maximum range of colormap (Default : None)
    option['cmap_marker']     : marker to use with colormap (Default : 'd')
    option['cmapzdata']       : data values to use for color mapping of
                                markers, e.g. RMSD or BIAS. (Default empty)
    option['colframe']        : color for the y (left) and x (bottom) spines

    option['colormap']        : 'on'/'off' switch to map color shading of
                                markers to CMapZData values ('on') or min to
                                max range of CMapZData values ('off').
                                (Default : 'on')

    option['equalAxes']       : 'on'/'off' switch to set axes to be equal 
                                (Default 'on')
                                
    option['labelweight']     : weight of the x & y axis labels
    option['locationcolorbar'] : location for the colorbar, 'NorthOutside' or
                                'EastOutside'
    option['markercolor']     : single color to use for all markers (Default: None)
    option['markercolors']    : dictionary with two colors as keys ('face', 'edge')
                                or None. If None or 'markerlegend' == 'on' then
                                considers only the value of 'markercolor'. (Default: None)
    option['markerdisplayed'] : markers to use for individual experiments
    option['markerlabel']     : name of the experiment to use for marker
    option['markerlabelcolor']: marker label color (Default: 'k')
    option['markerlayout']    : matrix layout for markers in legend [nrow, ncol] 
                                (Default [15, no. markers/15] ) 
    option['markerlegend']    : 'on'/'off' switch to display marker legend
                                (Default 'off')
    option['markers']         : Dictionary providing individual control of the marker
                                key - text label for marker, e.g. '14197'
                                key['labelColor'] - color of marker label, e.g. 'r' for red
                                key['symbol'] - marker symbol, e.g. 's' for square
                                key['size'] - marker size, e.g. 9
                                key['faceColor'] - marker face color, e.g. 'b' for blue
                                key['edgeColor'] - marker edge color, e.g. 'k' for black line
                                (Default: None)
    option['markersize']      : marker size (Default 10)
    option['markersymbol']    : marker symbol (Default 'o')

    option['normalized']      : statistics supplied are normalized with 
                                respect to the standard deviation of reference
                                values (Default 'off')
    option['obsUncertainty']  : Observational Uncertainty (default of 0)
    option['overlay']         : 'on'/'off' switch to overlay current
                                statistics on target diagram (Default 'off').
                                Only markers will be displayed.
    option['stylebias']       : line style for bias grid lines (Default: solid line '-')

    option['target_options_file'] : name of CSV file containing values for optional
                                arguments of the target_diagram function. If no file
                                suffix is given, a ".csv" is assumed. (Default: empty string '')

    option['ticks']           : define tick positions (default is that used 
                                by the axis function)
    option['titlecolorbar']   : title for the colorbar
    option['xticklabelpos']   : position of the tick labels along the x-axis 
                                (empty by default)
    option['yticklabelpos']   : position of the tick labels along the y-axis 
                                (empty by default)

    Author: Peter A. Rochford
        rochford.peter1@gmail.com

    Created on Sep 17, 2022
    Revised on Sep 17, 2022
    '''

    # Set default parameters for all options
    option = {}
    option['alpha'] = 1.0
    option['axismax'] = 0.0
    option['circlecols'] = None   # if None, considers 'colstd' only
    option['circlelinespec'] = 'k--'
    option['circlelinewidth'] = rcParams.get('lines.linewidth')
    option['circles'] = None
    option['circlestyle'] = None # circlelinespec by default
    option['circlelabelsize'] = 12

    option['circlecolor'] = option['circlelinespec'][0]
    option['circlestyle'] = option['circlelinespec'][1:]

    option['cmap'] = 'jet'
    option['cmap_vmin'] = None
    option['cmap_vmax'] = None
    option['cmap_marker'] = 'd'
    option['cmapzdata'] = []

    option['colframe'] = '#000000' # black

    option['colormap'] = 'on'
    option['equalaxes'] = 'on'
    
    option['labelweight'] = 'bold' # weight of the x/y labels ('light', 'normal', 'bold', ...)
    option['locationcolorbar'] = 'NorthOutside'

    option['markercolor'] = None
    option['markercolors'] = None  # if None, considers 'markercolor' only
    option['markerdisplayed'] = 'marker'
    option['markerlabel'] = ''
    option['markerlabelcolor'] = 'k'
    option['markerlayout'] = [15, None]
    option['markerlegend'] = 'off'
    option['markerobs'] = 'none'
    option['markers'] = None
    option['markersize'] = 10
    option['markersymbol'] = 'o'

    option['normalized'] = 'off'
    option['obsuncertainty'] = 0.0
    option['overlay'] = 'off'
    
    option['stylebias'] = '-.'
        
    option['target_options_file'] = ''
    
    option['ticks'] = []
    option['titlecolorbar'] = ''
    option['xticklabelpos'] = []
    option['yticklabelpos'] = []

    option['legend'] = dict(set_legend=False, bbox_to_anchor_x=1.4, bbox_to_anchor_y=1.1)

    return option

def _get_options(option, **kwargs) -> dict:
    '''
    Get values for optional arguments for target_diagram function.
    
    Gets the default optional arguments for the TARGET_DIAGRAM 
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

    Created on Sep 17, 2022
    Revised on Sep 17, 2022
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
            option[optname] = optvalue

            # Check values for specific options
            if optname == 'circlelinespec':
                option = _circle_color_style(option)
            elif optname == 'cmapzdata':
                if isinstance(option[optname], str):
                    raise ValueError('cmapzdata cannot be a string!')
                elif isinstance(option[optname], bool):
                    raise ValueError('cmapzdata cannot be a boolean!')
                option['cmapzdata'] = optvalue
            elif optname == 'equalaxes':
                option['equalaxes'] = check_on_off(option['equalaxes'])
            elif optname == 'markerlabel':
                if type(optvalue) is list:
                    option['markerlabel'] = optvalue
                elif type(optvalue) is dict:
                    option['markerlabel'] = optvalue
                else:
                    raise ValueError('markerlabel value is not a list or dictionary: ' +
                                    str(optvalue))
            elif optname == 'markerlegend':
                option['markerlegend'] = check_on_off(option['markerlegend'])
                
            elif optname in {'markercolors'}:
                accepted_keys = {
                    'markercolors': {'face', 'edge'},
                }
                _check_dict_with_keys(optname, option[optname],
                                    accepted_keys[optname], or_none=True)
                del accepted_keys

            elif optname == 'normalized':
                option['normalized'] = check_on_off(option['normalized'])
            elif optname == 'overlay':
                option['overlay'] = check_on_off(option['overlay'])
            elif optname == 'legend':
                if option['markerlegend'] == 'on':
                    option['legend'] = dict(set_legend=list(optvalue)[0], bbox_to_anchor_x=list(optvalue)[1],
                                            bbox_to_anchor_y=list(optvalue)[2])
            elif optname == 'circlelabelsize':
                option['circlelabelsize'] = optvalue

        del optname, optvalue   
    
    return option

def _read_options(option, **kwargs) -> dict:
    '''
    Reads the optional arguments from a CSV file. 
    
    Reads the optional arguments for target_diagram function from a 
    CSV file if a target_options_file parameter is provided that contains
    the name of a valid Comma Separated Value (CSV) file. Otherwise the
    function returns with no action taken. 
    
    INPUTS:
    option  : dictionary containing default option values

    *kwargs : variable-length keyword argument list. One of the keywords 
            must be in the list below for the function to perform any
            action.
    target_options_file : name of CSV file containing values for optional
                        arguments of the target_diagram function. If no file
                        suffix is given, a ".csv" is assumed. (Default: empty string '')
        
    OUTPUTS:
    option : dictionary containing option values

    Author:
    
    Peter Rochford, rochford.peter1@gmail.com

    Created on Sep 17, 2022
    Revised on Sep 17, 2022
    '''
    # Check if option filename provided
    name = ''
    for optname, optvalue in kwargs.items():
        optname = optname.lower()
        if optname == 'target_options_file':
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
    keys = objectData.iloc[:,0]
    values = objectData.iloc[:,1].tolist()

    # Identify keys requiring special consideration   
    listkey = ['cmapzdata', 'circles']
    tuplekey = []
    
    # Process for options read from CSV file
    for index in range(len(keys)):
        
        # Skip assignment if no value provided in CSV file
        if pd.isna(values[index]):
            continue
        
        # Convert list provided as string
        if is_list_in_string(values[index]):
            # Remove brackets
            values[index] = values[index].replace('[','').replace(']','')
        
        if keys[index] in listkey:
            if pd.isna(values[index]):
                option[keys[index]]=[]
            else:
                # Convert string to list of floats
                split_string = re.split(' |,', values[index])
                split_string = ' '.join(split_string).split()
                option[keys[index]] = [float(x) for x in split_string]
        
        elif keys[index] in tuplekey:
            try:
                option[keys[index]]=eval(values[index])
            except NameError:
                raise Exception('Invalid ' + keys[index] + ': '+ values[index])
        elif pd.isna(values[index]):
            option[keys[index]]=''
        elif is_int(values[index]):
            option[keys[index]] = int(values[index])
        elif is_float(values[index]):
            option[keys[index]] = float(values[index])
        elif values[index]=='None':
            option[keys[index]] = None
        else:
            option[keys[index]] = values[index]

    # Check values for specific options
    if option['circlelinespec']:
        option = _circle_color_style(option)

    return option

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
        if lowcase == 'off': return lowcase
        elif lowcase == 'on': return lowcase
        else:
            raise ValueError('Invalid value: ' + str(value))
    elif isinstance(value, bool):
        if value == False: value = 'off'
        elif value == True: value = 'on'
    else:
        raise ValueError('Invalid value: ' + str(value))

    return value
    
def get_target_diagram_options(**kwargs) -> dict:
    '''
    Get optional arguments for target_diagram function.
    
    Retrieves the optional arguments supplied to the TARGET_DIAGRAM 
    function as a variable-length keyword argument list (*KWARGS), and
    returns the values in an OPTION dictionary. Default values are 
    assigned to selected optional arguments. The function will terminate
    with an error if an unrecognized optional argument is supplied.
    
    INPUTS:
    *kwargs : variable-length keyword argument list. The keywords by 
            definition are dictionaries with keys that must correspond to 
            one choices given in OUTPUTS below.
    
    OUTPUTS:
    option : dictionary containing option values. (Refer to _default_options
            and display_taylor_diagram_options functions for more information.)

    Authors:
    
    Peter A. Rochford
        rochford.peter1@gmail.com
    
    Created on Nov 25, 2016
    Revised on Sep 17, 2022
    '''

    nargin = len(kwargs)

    # Set default parameters for all options
    option = _default_options()

    # No options requested, so return with only defaults
    if nargin == 0: return option

    # Read the optional arguments for taylor_diagram function from a 
    # CSV file, if specified. 
    option = _read_options(option, **kwargs)

    # Check for valid keys and values in dictionary
    # Allows user to override options specified in CSV file
    option = _get_options(option, **kwargs)
    
    return option

def find_exp(number) -> int:
    base10 =math.log10(abs(number))
    return math.floor(base10)

def blank_at_zero(tick,label):
    tolerance = 1.e-14
    if type(tick) is np.ndarray:
        index = np.where(abs(tick) < tolerance)
    else:
        temp = np.array(tick)
        index = np.where(abs(temp) < tolerance)
        del temp

    if np.size(index) == 0:
        raise ValueError('Array must span negative to positive values tick=',tick)
    else:
        index = index[0].item()
        label[index] = ''

def get_target_diagram_axes(x,y,option) -> dict:
    '''
    Get axes value for target_diagram function.
    
    Determines the axes information for a target diagram given the axis 
    values (X,Y) and the options in the data structure OPTION returned by 
    the GET_TARGET_DIAGRAM_OPTIONS function.
    
    INPUTS:
    x      : values for x-axis
    y      : values for y-axis
    option : dictionary containing option values. (Refer to 
            GET_TARGET_DIAGRAM_OPTIONS function for more information.)
    
    OUTPUTS:
    axes           : dictionary containing axes information for target diagram
    axes['xtick']  : x-values at which to place tick marks
    axes['ytick']  : y-values at which to place tick marks
    axes['xlabel'] : labels for xtick values
    axes['ylabel'] : labels for ytick values
    Also modifies the input variables 'ax' and 'option'

    Author: Peter A. Rochford
        rochford.peter1@gmail.com

    Created on Nov 25, 2016
    Revised on Aug 14, 2022
    '''
    # Specify max/min for axes
    foundmax = 1 if option['axismax'] != 0.0 else 0
    if foundmax == 0:
        # Axis limit not specified
        maxx = np.amax(np.absolute(x))
        maxy = np.amax(np.absolute(y))
    else:
        # Axis limit is specified
        maxx = option['axismax']
        maxy = option['axismax']

    # Determine default number of tick marks
    xtickvals = ticker.AutoLocator().tick_values(-1.0*maxx, maxx)
    ytickvals = ticker.AutoLocator().tick_values(-1.0*maxy, maxy)
    ntest = np.sum(xtickvals > 0)
    if ntest > 0:
        nxticks = np.sum(xtickvals > 0)
        nyticks = np.sum(ytickvals > 0)
        
        # Save nxticks and nyticks as function attributes for later 
        # retrieval in function calls
        get_target_diagram_axes.nxticks = nxticks
        get_target_diagram_axes.nyticks = nyticks
    else:
        # Use function attributes for nxticks and nyticks
        if hasattr(get_target_diagram_axes, 'nxticks') and \
            hasattr(get_target_diagram_axes, 'nxticks'):
            nxticks = get_target_diagram_axes.nxticks
            nyticks = get_target_diagram_axes.nyticks
        else:
            raise ValueError('No saved values for nxticks & nyticks.')
    
    # Set default tick increment and maximum axis values
    if foundmax == 0:
        maxx = xtickvals[-1]
        maxy = ytickvals[-1]
        option['axismax'] = max(maxx, maxy)

    # Check if equal axes requested
    if option['equalaxes'] == 'on':
        if maxx > maxy:
            maxy = maxx
            nyticks = nxticks
        else:
            maxx = maxy
            nxticks = nyticks

    # Convert to integer if whole number
    if type(maxx) is float and maxx.is_integer(): maxx = int(round(maxx))
    if type(maxx) is float and maxy.is_integer(): maxy = int(round(maxy))
    minx = -maxx; miny = -maxy
    
    # Determine tick values
    if len(option['ticks']) > 0:
        xtick = option['ticks']
        ytick = option['ticks']
    else:
        tincx = maxx/nxticks
        tincy = maxy/nyticks
        xtick = np.arange(minx, maxx+tincx, tincx)
        ytick = np.arange(miny, maxy+tincy, tincy)

    # Assign tick label positions
    if len(option['xticklabelpos']) == 0:
        option['xticklabelpos'] = xtick
    if len(option['yticklabelpos']) == 0:
        option['yticklabelpos'] = ytick
    
    #define x offset
    thexoffset = find_exp(maxx)
    if use_sci_notation(maxx): 
        ixsoffset = True
        xsoffset_str = "$\tx\mathdefault{10^{"+ str(thexoffset) +"}}\mathdefault{}$"
    else:
        ixsoffset = False
        xsoffset_str = 'None'

    theyoffset = find_exp(maxy)
    if use_sci_notation(maxy): 
        iysoffset = True
        ysoffset_str = "$\tx\mathdefault{10^{"+str(theyoffset)+"}}\mathdefault{}$"
    else:
        iysoffset = False
        ysoffset_str = 'None'
    
    # Set tick labels using provided tick label positions
    xlabel =[]; ylabel = [];
    
    # Set x tick labels
    for i in range(len(xtick)):
        index = np.where(option['xticklabelpos'] == xtick[i])
        if len(index) > 0:
            thevalue = xtick[i]
            if ixsoffset: 
                thevalue = xtick[i] * (10**(-1*thexoffset))
                label = get_axis_tick_label(thevalue)
                xlabel.append(label)
            else:
                label = get_axis_tick_label(xtick[i])
                xlabel.append(label)
        else:
            xlabel.append('')

    # Set tick labels at 0 to blank
    blank_at_zero(xtick,xlabel)
    
    # Set y tick labels
    for i in range(len(ytick)):
        index = np.where(option['yticklabelpos'] == ytick[i])
        if len(index) > 0:
            thevalue = ytick[i]
            if iysoffset: 
                thevalue = ytick[i] * (10**(-1*theyoffset)) 
                label = get_axis_tick_label(thevalue)
                ylabel.append(label)
            else:
                label = get_axis_tick_label(ytick[i])
                ylabel.append(label)
        else:
            ylabel.append('')

    # Set tick labels at 0 to blank
    blank_at_zero(ytick,ylabel)
    
    # Store output variables in data structure
    axes = {}
    axes['xtick'] = xtick
    axes['ytick'] = ytick
    axes['xlabel'] = xlabel
    axes['ylabel'] = ylabel
    axes['xoffset'] = xsoffset_str
    axes['yoffset'] = ysoffset_str
    
    return axes

def get_axis_tick_label(value):
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
    if not use_sci_notation(value):
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
                    if(number_sigfig > 5): go = False
                else:
                    before = trailing[number_digits]
                    number_sigfig = number_digits - 1
                number_digits+=1
    
        if number_digits == len(trailing): number_sigfig = number_digits

        # Round up the number to desired significant figures
        label = str(round(value, number_sigfig))
    else:
        label = "{:.1e}".format(value)

    return label

def use_sci_notation(value):
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
    if (abs(value)>0 and abs(value) < 1e-3):
        return True
    else:
        return False

def pol2cart(phi, rho):
    '''
    Transforms corresponding elements of polar coordinate arrays to 
    Cartesian coordinates.
    
    INPUTS:
    phi : polar angle counter-clockwise from x-axis in radians
    rho : radius
    
    OUTPUTS:
    x   : Cartesian x-coordinate
    y   : Cartesian y-coordinate
    '''

    x = np.multiply(rho, np.cos(phi))
    y = np.multiply(rho, np.sin(phi))
    return x, y

def overlay_target_diagram_circles(ax: matplotlib.axes.Axes, option: dict) -> None:
    '''
    Overlays circle contours on a target diagram.
    
    Plots circle contours on a target diagram to indicate standard
    deviation ranges and observational uncertainty threshold.
    
    INPUTS:
    ax     : matplotlib.axes.Axes object in which the Taylor diagram will be
            plotted
    option['axismax'] : maximum for the X & Y values. Used to set
            default circles when no contours specified
    option['circles'] : radii of circles to draw to indicate isopleths 
            of standard deviation
    option['circleLineSpec'] : circle line specification (default dashed 
            black, '--k')
    option['normalized']     : statistics supplied are normalized with 
            respect to the standard deviation of reference values
    option['obsUncertainty'] : Observational Uncertainty (default of 0)
    
    OUTPUTS:
    None.

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    '''

    theta = np.arange(0, 2*np.pi, 0.01)
    unit = np.ones(len(theta))
    # 1 - reference circle if normalized
    if option['normalized'] == 'on':
        rho = unit
        X, Y = pol2cart(theta, rho)
        # ax.plot(X, Y, 'k', 'LineWidth', option['circleLineWidth'])

    # Set range for target circles
    if option['normalized'] == 'on':
        circles = [x for x in np.arange(0.5, option['axismax'] + 0.5, 0.5)]
        if math.log10(option['axismax']) < 0:
            circles = [x for x in np.arange(0.5 * 10 ** math.floor(math.log10(option['axismax'])),
                                            option['axismax'] + 0.5 * 10 ** math.floor(math.log10(option['axismax'])),
                                            0.5 * 10 ** math.floor(math.log10(option['axismax'])))]
        elif math.log10(option['axismax']) >= 0 and math.log10(option['axismax']) < 1:
            circles = [x for x in np.arange(0.5, option['axismax'] + 0.5, 0.5)]
        else:
            circles = [x for x in np.arange(0.5 * 10 ** math.floor(math.log10(option['axismax'])),
                                            option['axismax'] + 0.5 * 10 ** math.floor(math.log10(option['axismax'])),
                                            0.5 * 10 ** math.floor(math.log10(option['axismax'])))]
    else:
        if option['circles'] is None:
            circles = [option['axismax'] * x for x in [.7, 1]]
        else:
            circles = np.asarray(option['circles'])
            index = np.where(circles <= option['axismax'])
            circles = [option['circles'][i] for i in index[0]]
    
    # 2 - secondary circles
    for c in circles:
        rho = c * unit
        X, Y = pol2cart(theta, rho)
        ax.plot(X, Y, linestyle=option['circlestyle'],
                color=option['circlecolor'],
                linewidth=option['circlelinewidth'])
    del c

    # 3 - Observational Uncertainty threshold
    if option['obsuncertainty'] > 0:
        rho = option['obsuncertainty'] * unit
        X, Y = pol2cart(theta, rho)
        ax.plot(X, Y, '--b')

def plot_pattern_diagram_colorbar(ax: matplotlib.axes.Axes, X, Y, Z,
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
    cxscale = fontSize/10 # scale color bar by font size
    markerSize = option['markersize']*2

    hp = plt.scatter(X,Y, s=markerSize, c=Z, marker=option['cmap_marker'],
                    cmap=option['cmap'], vmin=option['cmap_vmin'],
                    vmax=option['cmap_vmax'])
    hp.set_facecolor(hp.get_edgecolor())
    
    # Set parameters for color bar location
    location = option['locationcolorbar'].lower()
    xscale= 1.0
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
            cxscale = 6*fontSize/10
            labelpad = -30
    else:
        raise ValueError('Invalid color bar location: ' + option['locationcolorbar']);
    
    # Add color bar to plot
    if option['colormap'] == 'on':
        # map color shading of markers to colormap 
        hc = plt.colorbar(hp,orientation = orientation, aspect = aspect,
                        fraction = fraction, pad=0.06, ax = ax)

        # Limit number of ticks on color bar to reasonable number
        if orientation == 'horizontal':
            _setColorBarTicks(hc,5,20)
        
    elif option['colormap'] == 'off':
        # map color shading of markers to min to max range of Z values
        if len(Z) > 1:
            ax.clim(min(Z), max(Z))
            hc = ax.colorbar(hp,orientation = orientation, aspect = aspect,
                            fraction = fraction, pad=0.06, ticks=[min(Z), max(Z)],
                            ax = ax)
            
            # Label just min/max range
            hc.set_ticklabels(['Min.', 'Max.'])
    else:
        raise ValueError('Invalid option for option.colormap: ' + 
                        option['colormap']);
    
    if orientation == 'horizontal':
        location = _getColorBarLocation(hc, option, xscale = xscale,
                                    yscale = 7.5, cxscale = cxscale)
    else:
        location = _getColorBarLocation(hc, option, xscale = xscale,
                                    yscale = 1.0, cxscale = cxscale)

    hc.ax.set_position(location) # set new position
    hc.ax.tick_params(labelsize=fontSize) # set tick label size

    hc.ax.xaxis.set_ticks_position('top')
    hc.ax.xaxis.set_label_position('top')

    # Title the color bar
    if option['titlecolorbar']:
        if orientation == 'horizontal':
            hc.set_label(option['titlecolorbar'],fontsize=fontSize)
        else:
            hc.set_label(option['titlecolorbar'],fontsize=fontSize, 
                        labelpad=labelpad, y=1.05, rotation=0)
    else:
        hc.set_label(hc,'Color Scale',fontsize=fontSize)

def _getColorBarLocation(hc,option,**kwargs):
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
        location = [cp.x0 + xscale*0.5*(1+math.cos(math.radians(45)))*cp.width, yscale*cp.y0,
                    cxscale*cp.width/6, cp.height]
    else:
        # target diagram
        location = [cp.x0 + xscale*0.5*(1+math.cos(math.radians(60)))*cp.width, yscale*cp.y0,
                    cxscale*cp.width/6, cxscale*cp.height]

    return location

def _setColorBarTicks(hc,numBins,lenTick):
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
        hc.locator = ticker.MaxNLocator(nbins=numBins, prune = 'both')
        hc.update_ticks()
        
        # Check number of characters in tick labels is 
        # acceptable, otherwise reduce number of bins
        locs = str(hc.get_ticks())
        locs = locs[1:-1].split()
        lengthTick = 0
        for tick in locs:
            tickStr = str(tick).rstrip('.')
            lengthTick += len(tickStr)
        if lengthTick > maxChar: numBins -=1

def _disp(text):
    print(text)


def generate_markers(data_names, option):
    import itertools
    import matplotlib.colors as mcolors
    markers = {}

    # add colors and symbols
    hex_colors = ['#4C6EF5', '#F9C74F', '#90BE6D', '#5BC0EB', '#43AA8B', '#F3722C', '#855456', '#F9AFAF', '#F8961E'
        , '#277DA1', '#5A189A']
    colors = itertools.cycle([mcolors.rgb2hex(color) for color in hex_colors])
    symbols = itertools.cycle(["+", ".", "o", "*", "x", "s", "D", "^", "v", ">", "<", "p"])  # Cycle through symbols

    for name in data_names:
        color = next(colors)

        if not option['faceColor']:
            faceColor = color
        else:
            faceColor = option['faceColor']

        markers[name] = {
            "labelColor": color,
            "edgeColor": color,
            "symbol": next(symbols),
            "size": option['MARKERSsize'],
            "faceColor": 'none',
        }
    return markers
