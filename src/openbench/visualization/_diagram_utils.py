"""Shared helpers for the Taylor and Target diagram modules.

Both `Fig_taylor_diagram.py` and `Fig_target_diagram.py` historically
carried verbatim copies of these tiny utilities (one prefixed with `_`,
one without). Centralising them here removes ~100 lines of duplication
and ensures any future bug fix propagates to both.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Union

from matplotlib import ticker


def is_int(element) -> bool:
    """Check if value can be parsed as an integer."""
    try:
        int(element)
        return True
    except (ValueError, TypeError):
        return False


def is_float(element) -> bool:
    """Check if value can be parsed as a float."""
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def is_list_in_string(element: str) -> bool:
    """Check if a string contains list-like brackets."""
    return bool(re.search(r"\[|\]", element))


def disp(text: str = "") -> None:
    """Print a single line — used by the diagram-options help text."""
    print(text)


def dispopt(name: str, description: str) -> None:
    """Print an `option_name: description` line indented two spaces."""
    print(f"  {name}: {description}")


def parse_literal_option(value: str, option_name: str):
    """Parse a CSV option value with ``ast.literal_eval``.

    Diagram option files historically used Python tuple syntax for color
    triples, e.g. ``(0, 0.6, 0)``.  Use literal parsing rather than
    ``eval`` so option files cannot execute code.
    """
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {option_name}: {value}") from exc


# --- shared pattern-diagram helpers (were byte-identical copies in both Fig modules) ---


def _checkKey(dictionary, key):
    if key in dictionary.keys():
        return True
    else:
        return False


def _getColorBarLocation(hc, option, **kwargs):
    """
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
    """

    # Check for optional arguments and set defaults if required
    if "xscale" in kwargs:
        xscale = kwargs["xscale"]
    else:
        xscale = 1.0
    if "yscale" in kwargs:
        yscale = kwargs["yscale"]
    else:
        yscale = 1.0
    if "cxscale" in kwargs:
        cxscale = kwargs["cxscale"]
    else:
        cxscale = 1.0

    # Get original position of color bar and not modified position
    # because of Axes.apply_aspect being called.
    cp = hc.ax.get_position(original=True)

    # Calculate location : [left, bottom, width, height]
    if "checkstats" in option:
        # Taylor diagram
        location = [
            cp.x0 + xscale * 0.5 * (1 + math.cos(math.radians(45))) * cp.width,
            yscale * cp.y0,
            cxscale * cp.width / 6,
            cp.height,
        ]
    else:
        # target diagram
        location = [
            cp.x0 + xscale * 0.5 * (1 + math.cos(math.radians(60))) * cp.width,
            yscale * cp.y0,
            cxscale * cp.width / 6,
            cxscale * cp.height,
        ]

    return location


def _setColorBarTicks(hc, numBins, lenTick):
    """
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

    """

    maxChar = 10
    lengthTick = lenTick
    while lengthTick > maxChar:
        # Limit number of ticks on color bar to numBins-1
        hc.locator = ticker.MaxNLocator(nbins=numBins, prune="both")
        hc.update_ticks()

        # Check number of characters in tick labels is
        # acceptable, otherwise reduce number of bins
        locs = str(hc.get_ticks())
        locs = locs[1:-1].split()
        lengthTick = 0
        for tick in locs:
            tickStr = str(tick).rstrip(".")
            lengthTick += len(tickStr)
        if lengthTick > maxChar:
            numBins -= 1


def check_on_off(value):
    """
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
    """

    if isinstance(value, str):
        lowcase = value.lower()
        if lowcase == "off":
            return lowcase
        elif lowcase == "on":
            return lowcase
        else:
            raise ValueError("Invalid value: " + str(value))
    elif isinstance(value, bool):
        if not value:
            value = "off"
        elif value:
            value = "on"
    else:
        raise ValueError("Invalid value: " + str(value))

    return value


def _check_dict_with_keys(
    variable_name: str, dict_obj: Union[dict, None], accepted_keys: set, or_none: bool = False
) -> None:
    """
    Check if an argument in the form of dictionary has valid keys.
    :return: None. Raise 'ValueError' if evaluated variable is considered invalid.
    """

    # if variable is None, check if it can be None
    if dict_obj is None:
        if or_none:
            return None
        else:
            raise ValueError("%s cannot be None!" % variable_name)

    # check if every key provided is valid
    for key in dict_obj.keys():
        if key not in accepted_keys:
            raise ValueError("Unrecognized option of %s: %s" % (variable_name, key))
        del key

    return None
