import re
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib import colors
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import cmaps
import math

def process_unit(ref_unit,sim_unit,metric):
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

    unit = all_metrics_units[metric]
    if unit == 'Unitless':
        return '(-)'
    elif unit == '%':
        return '(%)'
    elif unit == 'Same as input data':
        return f'({ref_unit})'
    elif unit == 'Square of input data unit':
        return rf'($({ref_unit})^{{2}}$)'
    else:
        print('Warning: Missing metric unit!')
        return '(-)'

def convert_unit(input_str):
    """
    Convert a unit string according to the specified rules:
    1. Convert single lowercase 'w' to uppercase 'W'.
    2. Remove spaces; convert the nearest space before a single '-' to '/', or for multiple '-'s, convert the first space to '/(', others to '·', and append ')'.
    3. Convert numbers following '-' (e.g., -1, -2) to superscript, remove '-' and 1.
    4. Convert 'None' and 'unitless' to '-'.
    5. Convert 'Month' or 'month' to 'mon'.
    6. Convert 'CO2' to have subscript '2', add space before if none exists; convert 'mumol' to 'μmol'.
    7. Convert 'Day' to 'day'.
    8. Convert numbers in 'm2', 'm3', 'km2', 'km3' to superscript.
    9. Remove ' wind'.
    10. Convert standalone 'C' or 'c' to '°C'.
    11. Convert 'hpa' to 'hPa' and 'pa' to 'Pa'.
    12. Convert 'percentage' to '%'.

    Args:
        input_str (str): Input string to convert.

    Returns:
        str: Converted string.
    """
    if not isinstance(input_str, str):
        return input_str

    result = input_str

    # Rule 1: Convert single lowercase 'w' to uppercase 'W'
    result = re.sub(r'\bw\b', 'W', result)

    # Rule 3: Remove spaces, handle spaces before '-' as '/' or '·'
    # if '-' in result:
    #     parts = result.split('-')
    #     if len(parts) == 2:  # single '-'
    #         left = parts[0].rstrip()
    #         if ' ' in left:
    #             left = left.replace(' ', '/')
    #         result = left + '-'+ parts[1].lstrip()
    #     else:  # multiple '-'
    #         left = parts[0].rstrip()
    #         if ' ' in left:
    #             left = left.replace(' ', '/(')
    #         middle = '-'.join(part.strip() for part in parts[1:-1])
    #         if ' ' in middle:
    #             middle = middle.replace(' ', '·')
    #         right = parts[-1].lstrip()
    #         result = left + '-'+ middle + '-'+ right + ')'
    # result = result.replace(' ', '·')

    # Rule 3: Convert numbers following '-' to superscript, remove '-' and 1
    def to_superscript(match):
        num = match.group(0)
        superscript_map = str.maketrans('123456789-', '¹²³⁴⁵⁶⁷⁸⁹⁻')
        return num.translate(superscript_map)
    result = re.sub(r'-\d+', to_superscript, result)
    # result = result.replace('¹', '')

    # Rule 4: Convert 'None' and 'unitless' to '-'
    result = re.sub(r'\b(None|unitless)\b', '-', result)

    # Rule 5: Convert 'Month' or 'month' to 'mon'
    result = re.sub(r'\b[Mm]onth\b', 'mon', result)

    # Rule 6: Convert 'CO2' to subscript '2', add space before if needed; convert 'mumol' to 'μmol'
    result = re.sub(r'(\S)(CO2)', r'\1 CO₂', result)
    result = re.sub(r'\bCO2\b', r' CO₂', result)
    result = re.sub(r'\bmumol\b', r'μmol', result)

    # Rule 7: Convert 'Day' to 'day'
    result = re.sub(r'\bDay\b', 'day', result)

    # Rule 8: Convert numbers in 'm2', 'm3', 'km2', 'km3' to superscript
    result = re.sub(r'\b([km])(\d)\b', lambda m: m.group(1) + '⁰¹²³⁴⁵⁶⁷⁸⁹'[int(m.group(2))], result)

    # Rule 9: Remove ' wind'
    result = re.sub(r'\s+wind\b', '', result)

    # Rule 10: Convert standalone 'C' or 'c' to '°C'
    result = re.sub(r'\b[Cc]\b', '°C', result)

    # Rule 11: Convert 'hpa' to 'hPa' and 'pa' to 'Pa'
    result = re.sub(r'\bhpa\b', 'hPa', result)
    result = re.sub(r'\bpa\b', 'Pa', result)

    # Rule 12: Convert 'percentage' to '%'
    result = re.sub(r'\bpercentage\b', '%', result)

    return result

def get_colormap(cmap_name):
    """
    Dynamically retrieve the colormap attribute from the cmaps object based on the input string.
    
    Parameters:
        cmap_name (str): The name of the colormap, e.g., 'NMCVel2'
    
    Returns:
        colormap object
    
    Raises:
        ValueError: If the specified colormap name does not exist
    """
    try:
        return getattr(cmaps, cmap_name)
    except AttributeError:
        raise ValueError(f"Cannot find colormap named '{cmap_name}'")

def get_index(vmin, vmax, colormap='Spectral', varname=''):
    def get_ticks(vmin, vmax):
        diff = vmax - vmin
        small_value = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        large_value = sorted(set([int(i * 10**e) if i * 10**e > 10 else i * 10**e for e in range(4) for i in [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10] if i * 10**e <= 10000]))
        ALLOWED_TICKS = small_value + large_value
        TARGET_NUM_TICKS = 4
        ideal_tick = diff / TARGET_NUM_TICKS
        for tick in ALLOWED_TICKS:
            if tick >= ideal_tick:
                return tick
        return ALLOWED_TICKS[-1]

    # list1: (-∞,+∞)
    # list2: (-∞,1]
    # list3: [0,+∞)
    # list4: [0,1]
    # score_list: [0,1]
    # list5: [-1,1]
    # list6: uncertain

    list1 = ['percent_bias','bias','PBIAS_HF','PBIAS_LF','pbiasfdc','pc_max','pc_min','pc_ampli','rSD','rv',]
    list2 = ['NSE','KGE','KGESS','ubNSE','ubKGE','KGEkm','KGElf','KGEnp','rNSE','sKGE','wNSE','wsNSE']
    list3 = ['absolute_percent_bias','mean_absolute_error','RMSE','MSE','ubRMSE','CRMSD','nrmse','rsr','SMPI','ssq',]
    list4 = ['correlation_R2','index_agreement','LNSE','mNSE','valindex','L','rd','pfactor']
    score_list = ['BiasScore', 'RMSEScore', 'PhaseScore', 'IavScore', 'SpatialScore', 'Overall_Score', 'The_Ideal_Point_score']
    list5 = ['correlation','ubcorrelation','ubcorrelation_R2','rSpearman','kappa_coeff']
    list6 = ['md','ve','rfactor',]
        
    # Calculate ticks
    colorbar_ticks = get_ticks(vmin, vmax)
    ticks = matplotlib.ticker.MultipleLocator(base=colorbar_ticks)
    mticks = ticks.tick_values(vmin=vmin, vmax=vmax)
    mticks = [round(tick, 2) if isinstance(tick, float) and len(str(tick).split('.')[1]) > 2 else tick for tick in
              mticks]


    # if (varname in list1) or (varname in list3):
    if max((vmin-mticks[0]),(mticks[-1]-vmax)) <= (colorbar_ticks/2):
        mticks = mticks[:]
    else:
        if (vmin-mticks[0]) >= (colorbar_ticks/2):
            mticks = mticks[1:]
        if (mticks[-1]-vmax) >= (colorbar_ticks/2):
            mticks = mticks[:-1]

    # make sure metrics in range
    if (varname in list2) or (varname in list5):
        if vmin < 0:
            mticks = [-1,0,1]
        else:
            mtcisk = [x for x in mticks if x <= 1]
    elif varname in list3:
        mticks = [x for x in mticks if x >= 0]
    elif (varname in list4) or (varname in score_list):
        mticks = [x for x in mticks if 1>= x >= 0]
    else:
        if (mticks[0] < 0) & (mticks[-1] > 0):
            max_num = max(-mticks[0], mticks[-1])
            mticks = np.linspace(-max_num,max_num,5)

    if mticks[0] == mticks[-1]:
        n = get_least_significant_digit(mticks[-1])
        mticks[-1] = mticks[-1] + n
        mticks[0] = mticks[0] - n

    cmap = get_colormap('cmp_b2r')

    # if (mticks[0] < 0) & (mticks[-1] > 0):
    #     cmap = get_colormap('cmocean_balance')
    # elif (mticks[0] == 0) & (mticks[-1] == 1):
    #     cmap = get_colormap('cmp_b2r')
    # elif (mticks[-1] <= 0):
    #     cmap = get_colormap('cmocean_ice')
    # elif (mticks[0] >= 0):
    #     cmap = get_colormap('cmocean_amp')
    
    bnd = np.arange(vmin, vmax + colorbar_ticks / 2, colorbar_ticks / 2)
    # norm = colors.BoundaryNorm(bnd, cmap.N)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    if vmin < mticks[0] and vmax > mticks[-1]:
        extend = 'both'
    elif vmin < mticks[0]:
        extend = 'min'
    elif vmax > mticks[-1]:
        extend = 'max'
    else:
        extend = 'neither'
    return cmap, mticks, norm, bnd, extend

def tick_length(num):
    """
    Return the significant length of the number (excluding trailing zeros).

    Parameter:
    num (float): The number to be checked
    Return:
    int: The length of a number
    """
    num_str = str(num)
    
    if '.' not in num_str:
        length = len(num_str)
    else:
        integer_part = num_str.split('.')[0]
        decimal_part = num_str.split('.')[1]
        point_part = 0.5
        # decimal_part = decimal_part.rstrip('0')
        length = len(decimal_part) + len(integer_part) + point_part
    
    return length

def get_least_significant_digit(num):
    if num == 0:
        return 0  # 0 没有数量级
    magnitude = math.floor(math.log10(abs(num)))
    return 10 ** magnitude