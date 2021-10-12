from matplotlib import rcParams

# Colors
DATA_COLOR = 'C7'
FIT_COLOR = 'C3'
DISTANCE_VALID_COLOR = 'C2'  # Green
DISTANCE_INVALID_COLOR = 'C3'  # Red

# For saving pdf plots
DATA_SAVE_STYLE = {'marker': '.', 'ls': '', 'label': 'Exp', 'ms': 6, 'color': DATA_COLOR}
FIT_SAVE_STYLE = {'marker': '', 'ls': '-', 'label': 'Fit', 'ms': 6, 'color': FIT_COLOR}

FIT_PROCESS_STYLE = FIT_SAVE_STYLE.copy()
del FIT_PROCESS_STYLE['color']

DATA_SUMMARY_STYLE = DATA_SAVE_STYLE.copy()
del DATA_SUMMARY_STYLE['color']

# For displaying plots during fitting
IPH_MARKER = '.'
IPH_LINESTYLE = ''
IPH_LABEL = 'Exp'
IPH_MARKER_SIZE = 6
IPH_STYLE = {'marker': IPH_MARKER,
             'ls': IPH_LINESTYLE,
             'label': IPH_LABEL,
             'ms': IPH_MARKER_SIZE,
             'color': DATA_COLOR}


PHASE_MARKER = '.'
PHASE_LINESTYLE = ''
PHASE_LABEL = ''
PHASE_MARKER_SIZE = 6
PHASE_STYLE = {'marker': PHASE_MARKER,
               'ls': PHASE_LINESTYLE,
               'label': PHASE_LABEL,
               'ms': PHASE_MARKER_SIZE,
               'color': DATA_COLOR}


RE_IPH_MARKER = '.'
RE_IPH_LINESTYLE = ''
RE_IPH_LABEL = ''
RE_IPH_MARKER_SIZE = 6
RE_IPH_STYLE = {'marker': RE_IPH_MARKER,
                'ls': RE_IPH_LINESTYLE,
                'label': RE_IPH_LABEL,
                'ms': RE_IPH_MARKER_SIZE,
                'color': DATA_COLOR}

IM_IPH_MARKER = 'x'
IM_IPH_LINESTYLE = ''
IM_IPH_LABEL = ''
IM_IPH_MARKER_SIZE = 6
IM_IPH_STYLE = {'marker': IM_IPH_MARKER,
                'ls': IM_IPH_LINESTYLE,
                'label': IM_IPH_LABEL,
                'ms': IM_IPH_MARKER_SIZE,
                'color': DATA_COLOR}

DISTANCE_MARKER = '.'
DISTANCE_LINESTYLE = '-'
DISTANCE_LABEL = ''
DISTANCE_MARKER_SIZE = 6
DISTANCE_STYLE = {'marker': DISTANCE_MARKER,
                  'ls': DISTANCE_LINESTYLE,
                  'label': DISTANCE_LABEL,
                  'ms': DISTANCE_MARKER_SIZE}

DISTANCE_VALID_STYLE = DISTANCE_STYLE.copy()
DISTANCE_VALID_STYLE.update({'color': DISTANCE_VALID_COLOR, 'ls': ''})

DISTANCE_INVALID_STYLE = DISTANCE_STYLE.copy()
DISTANCE_INVALID_STYLE.update({'color': DISTANCE_INVALID_COLOR, 'ls': ''})


def set_plotstyle():
    """
    Set the plot parameters.

    Parameters
    -----------

    Returns
    -------

    """

    rcParams['axes.grid'] = True
    rcParams['grid.linestyle'] = ':'

    rcParams['mathtext.default'] = 'rm'

    rcParams['xtick.labelsize'] = 'x-small'
    rcParams['ytick.labelsize'] = 'x-small'
    rcParams['axes.titlesize'] = 'small'
    rcParams['axes.labelsize'] = 'small'
    rcParams['legend.fontsize'] = 'x-small'

    rcParams['figure.subplot.left'] = 0.10  # the left side of the subplots of the figure
    rcParams['figure.subplot.right'] = 0.98  # the right side of the subplots of the figure
    rcParams['figure.subplot.bottom'] = 0.1  # the bottom of the subplots of the figure
    rcParams['figure.subplot.top'] = 0.90
    rcParams['figure.subplot.hspace'] = 0.9
    rcParams['figure.subplot.wspace'] = 0.9

    rcParams['backend'] = 'TkAgg'

    rcParams['pdf.compression'] = 0
    rcParams['savefig.dpi'] = 150
