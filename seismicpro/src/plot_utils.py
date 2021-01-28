""" Utilily functions for visualization """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import ScalarFormatter, AutoLocator, IndexFormatter, LinearLocator, FixedFormatter
from matplotlib import patches, colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import measure_gain_amplitude, to_list, collect_components_data, is_1d_array, is_2d_array, infer_array


def setup_imshow(ax, arr, **kwargs):
    """ Calls ax.imshow(arr) with some set of arguments being fixed """

    defaults = {
        'cmap': 'gray',
        'vmin': np.quantile(arr, 0.1),
        'vmax': np.quantile(arr, 0.9),
        'aspect': 'auto',
        'extent': (0, arr.shape[1], arr.shape[0], 0),
    }

    ax.imshow(arr, **{**defaults, **kwargs})

def infer_axis_tickers(batch, index, src, x_tick, y_tick):
    """ In case x_ticker / y_ticker strings, corresponding tickers will be infered from dataframe / meta. 
    In case x_ticker / y_ticker dicts, they should contains axis configuration and will be used as is. 
    """    
    df = batch.index.get_df(index)

    x_ticker, y_ticker = {}, {}
    
    # process x-ticker
    if isinstance(x_tick, str): # infer ticker from dataframe
        sorting = batch.meta[src]['sorting']
        if sorting is not None:
            xticks_labels = df.sort_values(sorting)[x_tick]
        else:
            xticks_labels = df[x_tick]
        x_ticker['formatter'] = IndexFormatter(xticks_labels)
    elif isinstance(x_tick, dict): # dict with config passed, it will be used directly
        x_ticker = x_tick
    elif isinstance(x_tick, (list, tuple, np.ndarray)): # ticker will be inferred later in utils.setup_tickers
        x_ticker = x_tick

    # process y-ticker
    if y_tick == 'time': # infer ticker from meta
        yticks_labels = batch.meta[src]['samples']
        yticks_labels = [int(i) if i.is_integer() else i for i in yticks_labels]  
        y_ticker['formatter'] = IndexFormatter(yticks_labels)
    elif y_tick == 'samples': # default ticker will be used
        pass 
    elif isinstance(y_tick, dict): 
        y_ticker = y_tick
    elif isinstance(x_tick, (list, tuple, np.ndarray)): # ticker will be inferred later in utils.setup_tickers
        y_ticker = y_tick
    
    return x_ticker, y_ticker

def setup_tickers(ax, x_ticker, y_ticker):
    """ Setup the x / y axis tickers from the configs. 
    In case config miss ticker - default ticker will be used.
    In case ticker is array-like - matplotlib.ticker.IndexFormatter(ticker) will be used. """

    def _cast_ticker(ticker):
        if isinstance(ticker, (list, tuple, np.ndarray)):
            return {'formatter': IndexFormatter(ticker)}
        else:
            return ticker

    x_ticker, y_ticker = [_cast_ticker(ticker) for ticker in [x_ticker, y_ticker]]

    ax.xaxis.set_major_locator(x_ticker.get('locator', AutoLocator()))
    ax.yaxis.set_major_locator(y_ticker.get('locator', AutoLocator()))

    ax.xaxis.set_major_formatter(x_ticker.get('formatter', ScalarFormatter()))
    ax.yaxis.set_major_formatter(y_ticker.get('formatter', ScalarFormatter()))

def scatter_on_top(ax, attribute):
    """" Plot additional scatter on top of the 'ax' axes.  """
    divider = make_axes_locatable(ax)
    top_subax = divider.append_axes("top", 0.65, pad=0.01,  sharex=ax)
    top_subax.scatter(range(len(attribute)), attribute, s=5, c='k')
    top_subax.invert_yaxis()
    top_subax.set_xticks([])
    top_subax.yaxis.tick_right()

def scatter_within(ax, points, c, s, **kwargs):
    if is_1d_array(points):
        points = (points, )
    c, s = [to_list(obj, len(points)) for obj in [c, s]]
    for ix_pts, ix_c, ix_s in zip(points, c, s):
        ax.scatter(range(len(ix_pts)), ix_pts, c=ix_c, s=ix_s, **kwargs)

def seismic_plot(arrs, xlim=None, ylim=None, wiggle=False, std=1, # pylint: disable=too-many-branches, too-many-arguments
                 event=None, s=None, c=None, attribute=None,  
                 figsize=(9, 6), columnwise=False, title=None, line_color='k', names=None,  
                 x_ticker={}, y_ticker={},
                 save_to=None, dpi=None,  **kwargs):
    """Plot seismic traces.

    Parameters
    ----------
    arrs : np.2darray, or iterable or iterable of iterables of such arrays 
        Containers with seismic gathers to plot.
    xlim : tuple, optional
        Range in x-axis to show.
    ylim : tuple, optional
        Range in y-axis to show.
    wiggle : bool, default to False
        Show traces in a wiggle form.
    std : scalar, optional
        Amplitude scale for traces in wiggle form.
    event : array or iterable of such arrays
        Data corresponding to events, happened on the trace. Measured in samples. 
        Plotted within the gather. For example first break picking.
    s : scalar or array_like
        The marker size for the events
    c : color, sequence, or sequence of color
        The marker color for the events.
    attribute: array or iterable of such arrays
        Data with the trace's attribute. Plotted on top of the gather. For example, offset or elevation. 
    names : str or array-like, optional
        Title names to identify subplots.
    figsize : array-like, optional
        Output plot size.
    columnwise: bool, default is False.
        Whether to plot multiple gathers as column or row.
    save_to : str or None, optional
        If not None, save plot to given path.
    dpi : int, optional, default: None
        The resolution argument for matplotlib.pyplot.savefig.
    line_color : color, sequence, or sequence of color, optional, default is 'k'
        The trace color.
    title : str
        Plot title.
    kwargs : dict
        Additional keyword arguments for plot.

    Returns
    -------
    Multi-column subplots.

    Raises
    ------
    ValueError
        If ```line_color``` is sequence and it length is not equal to the number of traces.

    """
    arrs, attribute, event = [infer_array(data, condition) for data, condition in zip([arrs,    attribute,       event],
                                                                                [is_2d_array, is_1d_array, is_1d_array])]
    names = to_list(names)

    # Implicitely rearange event array so it follows the logic.
    # In case multiple types of events and gathers passed, e.g. event =  (picking1 and picking2) and arrs = (raw, lift) 
    # then the events are splitted across gathers, i.e. picking1 plotted on raw and picking2 plotted on lift.
    # In case event = (picking1, picking2) and arrs = raw, both picking1 and picking2 are plotted on the raw
    if event is not None:
        if event.shape[0] > arrs.shape[0]:
            event = infer_array(event.T, is_2d_array)
        if event.shape[1] > arrs.shape[1]:
            event = infer_array(event, is_2d_array)

    nrows, ncols = arrs.shape if not columnwise else arrs.shape[::-1]
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize * np.array([ncols, nrows]), squeeze=False)
    ax = ax if not columnwise else ax.T

    for i, (arr, ax) in enumerate(zip(arrs.flatten(), ax.flatten())):

        if not wiggle:
            arr = np.squeeze(arr)
        xlim_curr = xlim or (0, len(arr))

        if arr.ndim == 2:
            setup_tickers(ax, x_ticker, y_ticker)

            ylim_curr = ylim or (0, len(arr[0]))

            if wiggle:
                offsets = np.arange(*xlim_curr)

                if isinstance(line_color, str):
                    line_color = [line_color] * len(offsets)

                if len(line_color) != len(offsets):
                    raise ValueError("Lenght of line_color must be equal to the number of traces.")

                y = np.arange(*ylim_curr)
                for ix, k in enumerate(offsets):
                    x = k + std * arr[k, slice(*ylim_curr)] / np.std(arr)
                    col = line_color[ix]
                    ax.plot(x, y, '{}-'.format(col))
                    ax.fill_betweenx(y, k, x, where=(x > k), color=col)

            else:
                setup_imshow(ax, arr.T, **kwargs)
        
            if attribute is not None:
                scatter_on_top(ax, attribute.flatten()[i])

            if event is not None:
                scatter_within(ax, event.flatten()[i], c=c, s=s)

        elif arr.ndim == 1:
            ax.plot(arr, **kwargs)
        else:
            raise ValueError('Invalid ndim to plot data.')

        if names is not None:
            ax.set_title(names[i])

        if arr.ndim == 2:
            ax.set_ylim([ylim_curr[1], ylim_curr[0]])
            if (not wiggle) or (event is not None):
                ax.set_xlim(xlim_curr)

        if arr.ndim == 1:
            plt.xlim(xlim_curr)

    if title is not None:
        fig.suptitle(title)

    if save_to is not None:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')

    plt.show()

def spectrum_plot(arrs, frame, rate, max_freq=None, names=None,
                  figsize=None, save_to=None, dpi=None, **kwargs):
    """Plot seismogram(s) and power spectrum of given region in the seismogram(s).

    Parameters
    ----------
    arrs : array-like
        Seismogram or sequence of seismograms.
    frame : tuple
        List of slices that frame region of interest.
    rate : scalar
        Sampling rate.
    max_freq : scalar
        Upper frequence limit.
    names : str or array-like, optional
        Title names to identify subplots.
    figsize : array-like, optional
        Output plot size.
    save_to : str or None, optional
        If not None, save plot to given path.
    kwargs : dict
        Named argumets to matplotlib.pyplot.imshow.

    Returns
    -------
    Plot of seismogram(s) and power spectrum(s).
    """
    names = to_list(names)
    arrs = infer_array(arrs, is_2d_array)

    _, ax = plt.subplots(2, arrs.shape[1], figsize=figsize * np.array([arrs.shape[1], 2]), squeeze=False)
    for i, arr in enumerate(arrs.flatten()):
        setup_imshow(ax[0, i], arr.T, **kwargs)
        rect = patches.Rectangle((frame[0].start, frame[1].start),
                                 frame[0].stop - frame[0].start,
                                 frame[1].stop - frame[1].start,
                                 edgecolor='r', facecolor='none', lw=2)

        ax[0, i].set_title(names[i])
        ax[0, i].add_patch(rect)
        ax[0, i].set_aspect('auto')
        spec = abs(np.fft.rfft(arr[frame], axis=1))**2
        freqs = np.fft.rfftfreq(len(arr[frame][0]), d=rate)
        if max_freq is None:
            max_freq = np.inf

        mask = freqs <= max_freq
        ax[1, i].plot(freqs[mask], np.mean(spec, axis=0)[mask], lw=2)
        ax[1, i].set_xlabel('Hz')
        ax[1, i].set_title('Spectrum plot {}'.format(names[i] if names is not None else ''))

    if save_to is not None:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')

    plt.show()

def gain_plot(arrs, window=51, xlim=None, ylim=None, figsize=(8, ), 
              names=None, dpi=None, save_to=None, **kwargs):# pylint: disable=too-many-branches
    r"""Gain's graph plots the ratio of the maximum mean value of
    the amplitude to the mean value of the smoothed amplitude at the moment t.

    First of all for each trace the smoothed version calculated by following formula:
        $$Am = \sqrt{\mathcal{H}(Am)^2 + Am^2}, \ where$$
    Am - Amplitude of trace.
    $\mathcal{H}$ - is a Hilbert transformaion.

    Then the average values of the amplitudes (Am) at each time (t) are calculated.
    After it the resulted value received from the following equation:

        $$ G(t) = - \frac{\max{(Am)}}{Am(t)} $$

    Parameters
    ----------
    arrs : array-like
        Seismogram.
    window : int, default 51
        Size of smoothing window of the median filter.
    xlim : tuple or list with size 2
        Bounds for plot's x-axis.
    ylim : tuple or list with size 2
        Bounds for plot's y-axis.
    figsize : array-like, optional
        Output plot size.
    names : str or array-like, optional
        Title names to identify subplots.

    Returns
    -------
    Gain's plot.
    """
    names = to_list(names)
    arrs = infer_array(arrs, is_2d_array)

    _, ax = plt.subplots(2, arrs.shape[1], figsize=figsize * np.array([arrs.shape[1], 2]), squeeze=False)
    
    for i, arr in enumerate(arrs.flatten()):
        setup_imshow(ax[0, i], arr.T, **kwargs)
        ax[0, i].set_title(names[i])

        result = measure_gain_amplitude(arr, window)
        ax[1, i].plot(result, range(len(result)), **kwargs)
        if xlim is None:
            set_xlim = (max(result)-min(result)*.1, max(result)+min(result)*1.1)
        elif isinstance(xlim[0], (int, float)):
            set_xlim = xlim
        elif len(xlim) != len(arrs):
            raise ValueError('Incorrect format for xbounds.')
        else:
            set_xlim = xlim[i]

        if ylim is None:
            set_ylim = (len(result)+100, -100)
        elif isinstance(ylim[0], (int, float)):
            set_ylim = ylim
        elif len(ylim) != len(arrs):
            raise ValueError('Incorrect format for ybounds.')
        else:
            set_ylim = ylim[i]

        ax[1, i].set_ylim(set_ylim)
        ax[1, i].set_xlim(set_xlim)
        ax[1, i].set_xlabel('Maxamp/Amp')
        ax[1, i].set_ylabel('Time')
        ax[1, i].set_title('Gain plot {}'.format(names[i] if names is not None else ''))

    if save_to is not None:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')

    plt.show()

def statistics_plot(arrs, stats, rate=None, figsize=None, names=None,
                    save_to=None, dpi=None, **kwargs):
    """Show seismograms and various trace statistics, e.g. rms amplitude and rms frequency.

    Parameters
    ----------
    arrs : array-like
        Seismogram or sequence of seismograms.
    stats : str, callable or array-like
        Name of statistics in statistics zoo, custom function to be avaluated or array of stats.
    rate : scalar
        Sampling rate for spectral statistics.
    figsize : array-like, optional
        Output plot size.
    names : str or array-like, optional
        Title names to identify subplots.
    save_to : str or None, optional
        If not None, save plot to given path.
    kwargs : dict
        Named argumets to matplotlib.pyplot.imshow.

    Returns
    -------
    Plots of seismorgams and trace statistics.
    """
    def rms_freq(x, rate):
        "Calculate rms frequency."
        spec = abs(np.fft.rfft(x, axis=1))**2
        spec = spec / spec.sum(axis=1).reshape((-1, 1))
        freqs = np.fft.rfftfreq(len(x[0]), d=rate)
        return  np.sqrt((freqs**2 * spec).sum(axis=1))

    statistics_zoo = dict(ma_ampl=lambda x, *args: np.mean(abs(x), axis=1),
                          rms_ampl=lambda x, *args: np.sqrt(np.mean(x**2, axis=1)),
                          std_ampl=lambda x, *args: np.std(x, axis=1),
                          rms_freq=rms_freq)


    names, stats = [to_list(obj) for obj in [names, stats]]
    arrs = infer_array(arrs, is_2d_array) 

    _, ax = plt.subplots(2, arrs.shape[1], figsize=figsize * np.array([arrs.shape[1], 2]), squeeze=False)
    for i, arr in enumerate(arrs.flatten()):
        setup_imshow(ax[0, i], arr.T, **kwargs)
        ax[0, i].set_title(names[i])

        for k in stats:
            if isinstance(k, str):
                func, label = statistics_zoo[k], k
            else:
                func, label = k, k.__name__

            ax[1, i].plot(func(arr, rate), label=label)

        ax[1, i].legend()
        ax[1, i].set_xlim([0, len(arr)])
        ax[1, i].set_aspect('auto')
        ax[1, i].set_title('Statisticks  {}'.format(names[i] if names is not None else ''))

    if save_to is not None:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')

    plt.show()

def draw_histogram(df, layout, n_last):
    """Draw histogram of following attribute.
    Parameters
    ----------
    df : DataFrame
        Research's results
    layout : str
        string where each element consists two parts that splited by /. First part is the type
        of calculated value wrote in the "name" column. Second is name of column  with the parameters
        that will be drawn.
    n_last : int, optional
        The number of iterations at the end of which the averaging takes place.
    """
    name, attr = layout.split('/')
    max_iter = df['iteration'].max()
    mean_val = df[(df['iteration'] > max_iter - n_last) & (df['name'] == name)].groupby('repetition').mean()[attr]
    plt.figure(figsize=(8, 6))
    plt.title('Histogram of {}'.format(attr))
    plt.hist(mean_val)
    plt.axvline(mean_val.mean(), color='b', linestyle='dashed', linewidth=1, label='mean {}'.format(attr))
    plt.legend()
    plt.show()
    print('Average value (Median) is {:.4}\nStd is {:.4}'.format(mean_val.median(), mean_val.std()))

def show_1d_heatmap(idf, figsize=None, save_to=None, dpi=300, **kwargs):
    """Plot point distribution within 1D bins.

    Parameters
    ----------
    idf : pandas.DataFrame
        Index DataFrame.
    figsize : tuple
        Output figure size.
    save_to : str, optional
        If given, save plot to the path specified.
    dpi : int
        Resolution for saved figure.
    kwargs : dict
        Named argumets for ```matplotlib.pyplot.imshow```.

    Returns
    -------
    Heatmap plot.
    """
    bin_counts = idf.groupby(level=[0]).size()
    bins = np.array([i.split('/') for i in bin_counts.index])

    bindf = pd.DataFrame(bins, columns=['line', 'pos'])
    bindf['line_code'] = bindf['line'].astype('category').cat.codes + 1
    bindf = bindf.astype({'pos': 'int'})
    bindf['counts'] = bin_counts.values
    bindf = bindf.sort_values(by='line')

    brange = np.max(bindf[['line_code', 'pos']].values, axis=0)
    hist = np.zeros(brange, dtype=int)
    hist[bindf['line_code'].values - 1, bindf['pos'].values - 1] = bindf['counts'].values

    if figsize is not None:
        plt.figure(figsize=figsize)

    heatmap = plt.imshow(hist, **kwargs)
    plt.colorbar(heatmap)
    plt.yticks(np.arange(brange[0]), bindf['line'].drop_duplicates().values, fontsize=8)
    plt.xlabel("Bins index")
    plt.ylabel("Line index")
    plt.axes().set_aspect('auto')
    if save_to is not None:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')

    plt.show()

def show_2d_heatmap(idf, figsize=None, save_to=None, dpi=300, **kwargs):
    """Plot point distribution within 2D bins.

    Parameters
    ----------
    idf : pandas.DataFrame
        Index DataFrame.
    figsize : tuple
        Output figure size.
    save_to : str, optional
        If given, save plot to the path specified.
    dpi : int
        Resolution for saved figure.
    kwargs : dict
        Named argumets for ```matplotlib.pyplot.imshow```.

    Returns
    -------
    Heatmap plot.
    """
    bin_counts = idf.groupby(level=[0]).size()
    bins = np.array([np.array(i.split('/')).astype(int) for i in bin_counts.index])
    brange = np.max(bins, axis=0)

    hist = np.zeros(brange, dtype=int)
    hist[bins[:, 0] - 1, bins[:, 1] - 1] = bin_counts.values

    if figsize is not None:
        plt.figure(figsize=figsize)

    heatmap = plt.imshow(hist.T, origin='lower', **kwargs)
    plt.colorbar(heatmap)
    plt.xlabel('x-Bins')
    plt.ylabel('y-Bins')
    if save_to is not None:
        plt.savefig(save_to, dpi=dpi)
    plt.show()

def plot_metrics_map(metrics_map, cmap=None, title=None, figsize=(10, 7), # pylint: disable= too-many-arguments
                     pad=False, fontsize=11, ticks_range_x=None, ticks_range_y=None,
                     x_ticks=15, y_ticks=15, save_to=None, dpi=300, **kwargs):
    """ Plot map with metrics values.

    Parameters
    ----------
    metrics_map : array-like
        Array with aggregated metrics values.
    cmap : str or `~matplotlib.colors.Colormap`, optional
        Passed directly to `~matplotlib.imshow`
    title : str, optional
        The title of the plot.
    figsize : array-like with length 2, optional, default (10, 7)
        Output figure size.
    pad : bool, optional
        If true, edges of the figure will be padded with a thin white line.
        otherwise, the figure will not change.
    fontsize : int, optional, default 11
        The size of text.
    ticks_range_x : array-like with length 2, optional
        Min and max value of labels on the x-axis.
    ticks_range_y : array-like with length 2, optional
        Min and max value of labels on the y-axis.
    x_ticks : int, optional, default 15
        The number of coordinates on the x-axis.
    y_ticks : int, optional, default 15
        The number of coordinates on the y-axis.
    save_to : str, optional
        If given, save plot to the path specified.
    dpi : int, optional, default 300
        Resolution for saved figure.
    kwargs : dict, optional
        Named arguments for :func:`matplotlib.pyplot.imshow`.

    Note
    ----
    1. The map is drawn with origin = 'lower' by default, keep it in mind when passing ticks_labels.
    """
    if cmap is None:
        colors = ((0.0, 0.6, 0.0), (.66, 1, 0), (0.9, 0.0, 0.0))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'cmap', colors)
        cmap.set_under('black')
        cmap.set_over('red')

    origin = kwargs.pop('origin', 'lower')
    aspect = kwargs.pop('aspect', 'auto')
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(metrics_map, origin=origin, cmap=cmap,
                     aspect=aspect, **kwargs)

    if pad:
        ax.use_sticky_edges = False
        ax.margins(x=0.01, y=0.01)

    ax.set_title(title, fontsize=fontsize)
    cbar = fig.colorbar(img, extend='both', ax=ax)
    cbar.ax.tick_params(labelsize=fontsize)

    _set_ticks(ax=ax, img_shape=metrics_map.T.shape, ticks_range_x=ticks_range_x,
               ticks_range_y=ticks_range_y, x_ticks=x_ticks, y_ticks=y_ticks,
               fontsize=fontsize)

    # Block bellow does the same as _set_ticks() above
    # x_ticker = {
    #     'locator': LinearLocator(x_ticks),
    #     'formatter': FixedFormatter(np.linspace(*ticks_range_x, x_ticks))
    # }
    
    # y_ticker = {
    #     'locator': LinearLocator(y_ticks),
    #     'formatter': FixedFormatter(np.linspace(*ticks_range_y, y_ticks))
    # }
    # setup_tickers(ax, x_ticker, y_ticker)

    if save_to:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def _set_ticks(ax, img_shape, ticks_range_x=None, ticks_range_y=None, x_ticks=15,
               y_ticks=15, fontsize=None):
    """ Set x and y ticks.

    Parameters
    ----------
    ax : matplotlib axes
        Axes to which coordinates are added.
    img_shape : array with length 2
        Shape of the image to add ticks to.
    ticks_range_x : array-like with length 2, optional
        Min and max value of labels on the x-axis.
    ticks_range_y : array-like with length 2, optional
        Min and max value of labels on the y-axis.
    x_ticks : int, optional, default 15
        The number of coordinates on the x-axis.
    y_ticks : int, optional, default 15
        The number of coordinates on the y-axis.
    fontsize : int, optional
        The size of text.
    """
    ax.set_xticks(np.linspace(0, img_shape[0]-1, x_ticks))
    ax.set_yticks(np.linspace(0, img_shape[1]-1, y_ticks))

    if ticks_range_x is not None:
        ticks_labels_x = np.linspace(*ticks_range_x, x_ticks).astype(np.int32)
        ax.set_xticklabels(ticks_labels_x, size=fontsize)
    if ticks_range_y is not None:
        ticks_labels_y = np.linspace(*ticks_range_y, y_ticks).astype(np.int32)
        ax.set_yticklabels(ticks_labels_y, size=fontsize)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
