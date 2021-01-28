""" Seismic batch tools """
import csv
import shutil
import tempfile
import functools

import segyio
import numpy as np
import pandas as pd
from scipy.signal import medfilt, hilbert
from matplotlib.ticker import IndexFormatter
from sklearn.linear_model import LinearRegression

from ..batchflow import FilesIndex

DEFAULT_SEGY_HEADERS = ['FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE', 'CDP']
SUPPORT_SEGY_HEADERS = ['GroupX', 'GroupY']
GATHER_HEADERS = ['FieldRecord', 'RecieverID', 'CDP']
FILE_DEPENDEND_COLUMNS = ['TRACE_SEQUENCE_FILE', 'file_id']


def make_index(paths, index_type, extra_headers=None, index_name=None):
    """
    make index given components and paths

    Parameters
    ----------
    paths : dict
        Dictionary which keys are components of an index and values are paths to corresponding files with data
    index_type : class
        Type of resulting index
    extra_headers : list of str or 'all', or None, optional
        Extra headers to include in index
    index_name : str or None, optional
        segyio.TraceField keyword that will be set as index if `index_type` is `CustomIndex`.

    Returns
    -------
    """

    if not extra_headers:
        extra_headers = []

    return functools.reduce(lambda x, y: x.merge(y),
                            (index_type(name=name, path=path, extra_headers=extra_headers, index_name=index_name)
                             for name, path in paths.items()))


def partialmethod(func, *frozen_args, **frozen_kwargs):
    """Wrap a method with partial application of given positional and keyword
    arguments.

    Parameters
    ----------
    func : callable
        A method to wrap.
    frozen_args : misc
        Fixed positional arguments.
    frozen_kwargs : misc
        Fixed keyword arguments.

    Returns
    -------
    method : callable
        Wrapped method.
    """
    @functools.wraps(func)
    def method(self, *args, **kwargs):
        """Wrapped method."""
        return func(self, *frozen_args, *args, **frozen_kwargs, **kwargs)
    return method

def line_inclination(x, y):
    """Get regression line inclination towards x-axis.

    Parameters
    ----------
    x : array-like
        Data x coordinates.
    y : array-like
        Data y coordinates.

    Returns
    -------
    phi : float
        Inclination towards x-axis. The value is within (-pi/2, pi/2) range.
    """
    if np.std(y) < np.std(x):
        reg = LinearRegression().fit(x.reshape((-1, 1)), y)
        return np.arctan(reg.coef_[0])
    reg = LinearRegression().fit(y.reshape((-1, 1)), x)
    if reg.coef_[0] < 0.:
        return -(np.pi / 2) - np.arctan(reg.coef_[0])
    return (np.pi / 2) - np.arctan(reg.coef_[0])

def get_phi(dfr, dfs):
    """Get median absolute inclination for R and S lines.

    Parameters
    ----------
    dfr : pandas.DataFrame
        Data from R file SPS.
    dfs : pandas.DataFrame
        Data from S file SPS.

    Returns
    -------
    phi : float
        Median absolute inclination of R and S lines towards x-axis.
        The value is within (0, pi/2) range.
    """
    incl = []
    for _, group in dfs.groupby('sline'):
        x, y = group[['x', 'y']].values.T
        incl.append(line_inclination(x, y))
    for _, group in dfr.groupby('rline'):
        x, y = group[['x', 'y']].values.T
        incl.append(line_inclination(x, y))
    return np.median(np.array(incl) % (np.pi / 2))

def random_bins_shift(pts, bin_size, iters=100):
    """Monte-Carlo best shift estimation.

    Parameters
    ----------
    pts : array-like
        Point coordinates.
    bin_size : scalar or tuple of scalars
        Bin size of 1D or 2D grid.
    iters : int
        Number of samples.

    Returns
    -------
    shift : float or tuple of floats
        Optimal grid shift from its default origin that is np.min(pts, axis=0).
    """
    t = np.max(pts, axis=0).reshape((-1, 1))
    min_unif = np.inf
    best_shift = np.zeros(pts.ndim)
    for _ in range(iters):
        shift = -bin_size * np.random.random(pts.ndim)
        s = bin_size * ((np.min(pts, axis=0) - shift) // bin_size)
        bins = [np.arange(a, b + bin_size, bin_size) for a, b in zip(s + shift, t)]
        if pts.ndim == 2:
            h = np.histogram2d(*pts.T, bins=bins)[0]
        elif pts.ndim == 1:
            h = np.histogram(pts, bins=bins[0])[0]
        else:
            raise ValueError("pts should be ndim = 1 or 2.")

        unif = np.std(h[h > 0])
        if unif < min_unif:
            min_unif = unif
            best_shift = shift

    return best_shift

def gradient_bins_shift(pts, bin_size, max_iters=10, eps=1e-3):
    """Iterative best shift estimation.

    Parameters
    ----------
    pts : array-like
        Point coordinates.
    bin_size : scalar or tuple of scalars
        Bin size of 1D or 2D grid.
    max_iters : int
        Maximal number of iterations.
    eps : float
        Iterations stop criteria.

    Returns
    -------
    shift : float or tuple of floats
        Optimal grid shift from its default origin that is np.min(pts, axis=0).
    """
    t = np.max(pts, axis=0).reshape((-1, 1))
    shift = np.zeros(pts.ndim)
    states = []
    states_std = []
    for _ in range(max_iters):
        s = bin_size * ((np.min(pts, axis=0) - shift) // bin_size)
        bins = [np.arange(a, b + bin_size, bin_size) for a, b in zip(s + shift, t)]
        if pts.ndim == 2:
            h = np.histogram2d(*pts.T, bins=bins)[0]
            dif = np.diff(h, axis=0) / 2.
            vmax = np.vstack([np.max(h[i: i + 2], axis=0) for i in range(h.shape[0] - 1)])
            ratio = dif[vmax > 0] / vmax[vmax > 0]
            xshift = bin_size * np.mean(ratio)
            dif = np.diff(h, axis=1) / 2.
            vmax = np.vstack([np.max(h[:, i: i + 2], axis=1) for i in range(h.shape[1] - 1)]).T
            ratio = dif[vmax > 0] / vmax[vmax > 0]
            yshift = bin_size * np.mean(ratio)
            move = np.array([xshift, yshift])
        elif pts.ndim == 1:
            h = np.histogram(pts, bins=bins[0])[0]
            dif = np.diff(h) / 2.
            vmax = np.hstack([np.max(h[i: i + 2]) for i in range(len(h) - 1)])
            ratio = dif[vmax > 0] / vmax[vmax > 0]
            xshift = bin_size * np.mean(ratio)
            move = np.array([xshift])
        else:
            raise ValueError("pts should be ndim = 1 or 2.")

        states.append(shift.copy())
        states_std.append(np.std(h[h > 0]))

        if np.linalg.norm(move) < bin_size * eps:
            break

        shift += move
    if states_std:
        i = np.argmin(states_std)
        return states[i] % bin_size

    return shift

def rotate_2d(arr, phi):
    """Rotate 2D vector counter-clockwise.

    Parameters
    ----------
    arr : array-like
        Vector coordinates.
    phi : radians
        Rotation angle.

    Returns
    -------
    arr : array-like
        Rotated vector.
    """
    c, s = np.cos(phi), np.sin(phi)
    rotm = np.array([[c, -s], [s, c]])
    return np.dot(rotm, arr.T).T

def make_1d_bin_index(dfr, dfs, dfx, bin_size, origin=None, phi=None,
                      opt='gradient', **kwargs):
    """Get bins for 1d seismic geometry.

    Parameters
    ----------
    dfr : pandas.DataFrame
        SPS R file data.
    dfs : pandas.DataFrame
        SPS S file data.
    dfx : pandas.DataFrame
        SPS X file data.
    bin_size : scalar
        Grid bin size.
    origin : dict
        Grid origin for each line.
    phi : dict
        Grid orientation for each line.
    opt : str
        Grid location optimizer.
    kwargs : dict
        Named argumets for optimizer.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with bins indexing.
    """
    rids = np.hstack([np.arange(s, e + 1) for s, e in
                      list(zip(*[dfx['from_receiver'], dfx['to_receiver']]))])
    channels = np.hstack([np.arange(s, e + 1) for s, e in
                          list(zip(*[dfx['from_channel'], dfx['to_channel']]))])
    n_reps = dfx['to_receiver'] - dfx['from_receiver'] + 1

    dtypes = dfx.dtypes.values
    dfx = pd.DataFrame(dfx.values.repeat(n_reps, axis=0), columns=dfx.columns)
    for i, c in enumerate(dfx.columns):
        dfx[c] = dfx[c].astype(dtypes[i])

    dfx['rid'] = rids
    dfx['trace_number'] = channels
    dfm = (dfx
           .merge(dfs, on=['sline', 'sid'])
           .merge(dfr, on=['rline', 'rid'], suffixes=('_s', '_r')))
    dfm['CDP_X'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['CDP_Y'] = (dfm['y_s'] + dfm['y_r']) / 2.
    dfm['azimuth'] = np.arctan2(dfm['y_r'] - dfm['y_s'], dfm['x_r'] - dfm['x_s'])

    dfm['x_index'] = None
    meta = {}

    for rline, group in dfm.groupby('rline'):
        pts = group[['CDP_X', 'CDP_Y']].values
        if phi is None:
            if np.std(pts[:, 0]) > np.std(pts[:, 1]):
                reg = LinearRegression().fit(pts[:, :1], pts[:, 1])
                phi_ = np.arctan(reg.coef_)[0]
            else:
                reg = LinearRegression().fit(pts[:, 1:], pts[:, 0])
                phi_ = np.arctan(1. / reg.coef_)[0]
        else:
            phi_ = np.radians(phi[rline]) # pylint: disable=assignment-from-no-return

        pts = rotate_2d(pts, - phi_)
        ppx, y = pts[:, 0], np.mean(pts[:, 1])

        if origin is None:
            if opt == 'gradient':
                shift = gradient_bins_shift(ppx, bin_size, **kwargs)
            elif opt == 'monte-carlo':
                shift = random_bins_shift(ppx, bin_size, **kwargs)
            else:
                raise ValueError('Unknown grid optimizer.')

            s = shift + bin_size * ((np.min(ppx) - shift) // bin_size)
            origin_ = rotate_2d(np.array([[s, y]]), phi_)[0]
        else:
            origin_ = origin[rline]
            s = rotate_2d(origin_.reshape((-1, 2)), - phi_)[0, 0]

        t = np.max(ppx)
        bins = np.arange(s, t + bin_size, bin_size)

        index = np.digitize(ppx, bins)

        dfm.loc[dfm['rline'] == rline, 'x_index'] = index
        meta.update({rline: dict(origin=origin_,
                                 phi=np.rad2deg(phi_),
                                 bin_size=bin_size)})

    dfm['bin_id'] = (dfm['rline'].astype(str) + '/' + dfm['x_index'].astype(str)).values
    dfm.set_index('bin_id', inplace=True)

    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2) / 2.

    dfm.drop(labels=['from_channel', 'to_channel',
                     'from_receiver', 'to_receiver',
                     'x_index'], axis=1, inplace=True)
    dfm.rename(columns={'x_s': 'SourceX', 'y_s': 'SourceY'}, inplace=True)

    return dfm, meta

def make_2d_bin_index(dfr, dfs, dfx, bin_size, origin=None, phi=None,
                      opt='gradient', **kwargs):
    """Get bins for 2d seismic geometry.

    Parameters
    ----------
    dfr : pandas.DataFrame
        SPS R file data.
    dfs : pandas.DataFrame
        SPS S file data.
    dfx : pandas.DataFrame
        SPS X file data.
    bin_size : tuple
        Grid bin size.
    origin : dict
        Grid origin for each line.
    phi : dict
        Grid orientation for each line.
    opt : str
        Grid location optimizer.
    kwargs : dict
        Named argumets for optimizer.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with bins indexing.
    """
    if bin_size[0] != bin_size[1]:
        raise ValueError('Bins are not square')

    bin_size = bin_size[0]

    rids = np.hstack([np.arange(s, e + 1) for s, e in
                      list(zip(*[dfx['from_receiver'], dfx['to_receiver']]))])
    channels = np.hstack([np.arange(s, e + 1) for s, e in
                          list(zip(*[dfx['from_channel'], dfx['to_channel']]))])
    n_reps = dfx['to_receiver'] - dfx['from_receiver'] + 1

    dtypes = dfx.dtypes.values
    dfx = pd.DataFrame(dfx.values.repeat(n_reps, axis=0), columns=dfx.columns)
    for i, c in enumerate(dfx.columns):
        dfx[c] = dfx[c].astype(dtypes[i])

    dfx['rid'] = rids
    dfx['TraceNumber'] = channels
    dfm = (dfx
           .merge(dfs, on=['sline', 'sid'])
           .merge(dfr, on=['rline', 'rid'], suffixes=('_s', '_r')))
    dfm['CDP_X'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['CDP_Y'] = (dfm['y_s'] + dfm['y_r']) / 2.
    dfm['azimuth'] = np.arctan2(dfm['y_r'] - dfm['y_s'], dfm['x_r'] - dfm['x_s'])

    if phi is None:
        phi = get_phi(dfr, dfs)
    else:
        phi = np.radians(phi) # pylint: disable=assignment-from-no-return

    if phi > 0:
        phi += -np.pi / 2

    pts = rotate_2d(dfm[['CDP_X', 'CDP_Y']].values, -phi) # pylint: disable=invalid-unary-operand-type

    if origin is None:
        if opt == 'gradient':
            shift = gradient_bins_shift(pts, bin_size, **kwargs)
        elif opt == 'monte-carlo':
            shift = random_bins_shift(pts, bin_size, **kwargs)
        else:
            raise ValueError('Unknown grid optimizer.')

        s = shift + bin_size * ((np.min(pts, axis=0) - shift) // bin_size)
        origin = rotate_2d(s.reshape((1, 2)), phi)[0]
    else:
        s = rotate_2d(origin.reshape((1, 2)), -phi)[0] # pylint: disable=invalid-unary-operand-type

    t = np.max(pts, axis=0)
    xbins, ybins = np.array([np.arange(a, b + bin_size, bin_size) for a, b in zip(s, t)])

    x_index = np.digitize(pts[:, 0], xbins)
    y_index = np.digitize(pts[:, 1], ybins)

    dfm['bin_id'] = np.array([ix + '/' + iy for ix, iy in zip(x_index.astype(str), y_index.astype(str))])
    dfm.set_index('bin_id', inplace=True)

    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2) / 2.

    dfm = dfm.drop(labels=['from_channel', 'to_channel',
                           'from_receiver', 'to_receiver'], axis=1)
    dfm.rename(columns={'x_s': 'SourceX', 'y_s': 'SourceY'}, inplace=True)
    meta = dict(origin=origin, phi=np.rad2deg(phi), bin_size=(bin_size, bin_size))
    return dfm, meta

def make_bin_index(dfr, dfs, dfx, bin_size, origin=None, phi=None, **kwargs):
    """Get bins for seismic geometry.

    Parameters
    ----------
    dfr : pandas.DataFrame
        SPS R file data.
    dfs : pandas.DataFrame
        SPS S file data.
    dfx : pandas.DataFrame
        SPS X file data.
    bin_size : scalar or tuple of scalars
        Grid bin size.
    origin : dict
        Grid origin for each line.
    phi : dict
        Grid orientation for each line.
    opt : str
        Grid location optimizer.
    kwargs : dict
        Named argumets for optimizer.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with bins indexing.
    """
    if isinstance(bin_size, (list, tuple, np.ndarray)):
        df, meta = make_2d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, **kwargs)
    else:
        df, meta = make_1d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, **kwargs)

    df.columns = pd.MultiIndex.from_arrays([df.columns, [''] * len(df.columns)])
    return df, meta

def build_sps_df(dfr, dfs, dfx):
    """Index traces according to SPS data.

    Parameters
    ----------
    dfr : pandas.DataFrame
        SPS R file data.
    dfs : pandas.DataFrame
        SPS S file data.
    dfx : pandas.DataFrame
        SPS X file data.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with trace indexing.
    """
    rids = np.hstack([np.arange(s, e + 1) for s, e in
                      zip(*[dfx['from_receiver'], dfx['to_receiver']])])
    channels = np.hstack([np.arange(s, e + 1) for s, e in
                          zip(*[dfx['from_channel'], dfx['to_channel']])])
    n_reps = dfx['to_receiver'] - dfx['from_receiver'] + 1

    dfx.drop(labels=['from_channel', 'to_channel', 'from_receiver', 'to_receiver'],
             axis=1, inplace=True)

    dtypes = dfx.dtypes.values
    dfx = pd.DataFrame(dfx.values.repeat(n_reps, axis=0), columns=dfx.columns)
    for i, c in enumerate(dfx.columns):
        dfx[c] = dfx[c].astype(dtypes[i])

    dfx['rid'] = rids
    dfx['TraceNumber'] = channels
    dfm = (dfx
           .merge(dfs, on=['sline', 'sid'])
           .merge(dfr, on=['rline', 'rid'], suffixes=('_s', '_r')))
    dfm['CDP_X'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['CDP_Y'] = (dfm['y_s'] + dfm['y_r']) / 2.
    dfm['azimuth'] = np.arctan2(dfm['y_r'] - dfm['y_s'], dfm['x_r'] - dfm['x_s'])
    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2) / 2.
    dfm.rename(columns={'x_s': 'SourceX', 'y_s': 'SourceY'}, inplace=True)
    dfm.columns = pd.MultiIndex.from_arrays([dfm.columns, [''] * len(dfm.columns)])

    return dfm

def make_segy_index(filename, extra_headers=None, limits=None):
    """Index traces in a single SEGY file.

    Parameters
    ----------
    filename : str
        Path to SEGY file.
    extra_headers : array-like or str
        Additional headers to put unto DataFrme. If 'all', all headers are included.
    limits : slice or int, default to None
        If int, index only first ```limits``` traces. If slice, index only traces
        within given range. If None, index all traces.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with trace indexing.
    """
    if not isinstance(limits, slice):
        limits = slice(limits)

    with segyio.open(filename, strict=False) as segyfile:
        segyfile.mmap()
        if extra_headers == 'all':
            headers = [h.__str__() for h in segyio.TraceField.enums()]
        elif extra_headers is None:
            headers = DEFAULT_SEGY_HEADERS + SUPPORT_SEGY_HEADERS
        else:
            extra_headers = [extra_headers] if isinstance(extra_headers, str) else list(extra_headers)
            headers = set(DEFAULT_SEGY_HEADERS + extra_headers + SUPPORT_SEGY_HEADERS)

        meta = dict()

        for k in headers:
            meta[k] = segyfile.attributes(getattr(segyio.TraceField, k))[limits]

        meta['file_id'] = np.repeat(filename, segyfile.tracecount)[limits]
        meta['RecieverID'] = np.array([hash(pair) for pair in zip(meta['GroupX'], meta['GroupY'])])

    df = pd.DataFrame(meta)
    return df

def build_segy_df(extra_headers=None, name=None, limits=None, **kwargs):
    """Index traces in multiple SEGY files.

    Parameters
    ----------
    extra_headers : array-like or str
        Additional headers to put into DataFrame. If 'all', all headers are included.
    name : str
        Name that will be associated with indexed traces.
    limits : slice or int, default to None
        If int, index only first ```limits``` traces. If slice, index only traces
        within given range. If None, index all traces.
    kwargs : dict
        Named argumets for ```batchflow.FilesIndex```.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with trace indexing.
    """
    markup_path = kwargs.pop('markup_path', None)
    index = FilesIndex(**kwargs)
    df = pd.concat([make_segy_index(index.get_fullpath(i), extra_headers, limits) for
                    i in sorted(index.indices)])
    if len(index) > 1:
        for colname in GATHER_HEADERS:
            if np.any(df[[colname, 'file_id']].groupby(colname).nunique()[('file_id')] > 1):
                raise ValueError((f'Non-unique values in {colname} among provided files!',
                                  'Resulting index may not be unique.'))
    if markup_path is not None:
        markup = pd.read_csv(markup_path)
        df = df.merge(markup, how='inner')
    common_cols = list(set(df.columns) - set(FILE_DEPENDEND_COLUMNS))
    df = df[common_cols + FILE_DEPENDEND_COLUMNS]
    df.columns = pd.MultiIndex.from_arrays([common_cols + FILE_DEPENDEND_COLUMNS,
                                            [''] * len(common_cols) + [name] * len(FILE_DEPENDEND_COLUMNS)])
    return df

def calc_sdc(ix, time, speed, v_pow, t_pow):
    r""" Calculate spherical divergence correction (SDC).
    This value has the following formula:
    $$ g(t) = \frac{V_{rms}^{v_{pow}} * t^{t_{pow}}}{V_0} $$

    Here parameters $v_{pow} and t_{pow} is a hyperparameters.
    The quality of the correction depends on them.

    $$ V_{rms} = \left(\frac{\sum_0^t V^2}{|V|} \right)^{1/2} $$
    Where $|V|$ is a number of elements in V.

    Parameters
    ----------
    time : array
        Trace time values.
        Time measured in either in samples or in milliseconds.
    speed : array
        Wave propagation speed depending on the depth.
        Speed is measured in samples.
    v_pow : float or int
        Speed's power.
    t_pow : float or int
        Time's power.

    Returns
    -------
        : float
        Correction value to suppress the spherical divergence.
    """
    correction = (((np.mean(speed[: ix+1]**2))**.5) ** v_pow * time[ix] ** t_pow)/speed[0]
    if correction == 0:
        return 1.
    return correction

def calculate_sdc_for_field(field, time, speed, v_pow=2, t_pow=1):
    """ Correction of spherical divergence.

    Parameters
    ----------
    field : array or arrays
        Field for correction.
    time : array
        Trace time values.
        Time measured in either in samples or in milliseconds.
    speed : array
        Wave propagation speed depending on the depth.
        Speed is measured in samples.
    v_pow : float or int
        Speed's power.
    t_pow : float or int
        Time's power.

    Returns
        : array of arrays
        Corrected field.
    """
    new_field = np.zeros_like(field)
    for ix in range(field.shape[1]):
        timestamp = field[:, ix]
        correction_coef = (calc_sdc(ix, time, speed, v_pow=v_pow, t_pow=t_pow)
                           / calc_sdc(np.argmax(time), time, speed, v_pow=v_pow, t_pow=t_pow))
        new_field[:, ix] = timestamp * correction_coef
    return new_field


def measure_gain_amplitude(field, window):
    """Calculate the gain amplitude.

    Parameters
    ----------
    field : array or arrays
        Field for amplitude measuring.

    Returns
    -------
        : array
        amplitude values in each moment t
        after transformations.
    """
    h_sample = []
    for trace in field:
        hilb = hilbert(trace).real
        env = (trace**2 + hilb**2)**.5
        h_sample.append(env)

    h_sample = np.array(h_sample)
    mean_sample = np.mean(h_sample, axis=0)
    max_val = np.max(mean_sample)
    dt_val = (-1) * (max_val / mean_sample)
    result = medfilt(dt_val, window)
    return result

def calculate_sdc_quality(parameters, field, time, speed, window=51):
    """Calculate the quality of estimated parameters.

    The quality caluclated as the median of absolute value of the first order derivative.

    Parameters
    ----------
    parameters : list of 2
        Power values for speed and time.
    field : array or arrays
        Field for compensation.
    time : array
        Trace time values.
        Time measured in either in samples or in milliseconds.
    speed : array
        Wave propagation speed depending on the depth.
        Speed is measured in samples.
    window : int, default 51
        Size of smoothing window of the median filter.

    Returns
    -------
        : float
        Error with given parameters.
    """

    v_pow, t_pow = parameters
    new_field = calculate_sdc_for_field(field, time=time, speed=speed,
                                        v_pow=v_pow, t_pow=t_pow)

    result = measure_gain_amplitude(new_field, window)
    return np.median(np.abs(np.gradient(result)))

def massive_block(data):
    """ Function that takes 2d array and returns the indices of the
    beginning of the longest block of ones in each row.

    Parameters
    ----------
    data : np.array
        Array with masks.

    Returns
    -------
    ind : list
        Indices of the beginning of the longest blocks for each row.
    """
    arr = np.append(data, np.zeros((data.shape[0], 1)), axis=1)
    arr = np.insert(arr, 0, 0, axis=1)

    plus_one = np.argwhere((np.diff(arr)) == 1)
    minus_one = np.argwhere((np.diff(arr)) == -1)

    if len(plus_one) == 0:
        return [[0]] * data.shape[0]

    distance = minus_one[:, 1] - plus_one[:, 1]
    mask = minus_one[:, 0]

    idxs = np.argsort(distance, kind="stable")
    sort = idxs[np.argsort(mask[idxs], kind="stable")]
    ind = [0] * mask[0]
    for i in range(len(sort[:-1])):
        diff = mask[i +1] - mask[i]
        if diff > 1:
            ind.append(plus_one[:, 1][sort[i]])
            ind.extend([0] * (diff - 1))
        elif diff == 1:
            ind.append(plus_one[:, 1][sort[i]])
    ind.append(plus_one[:, 1][sort[-1]])
    ind.extend([0] * (arr.shape[0] - mask[-1] - 1))
    return ind

def transform_to_fixed_width_columns(path, path_save=None, n_spaces=8, max_len=(6, 4)):
    """ Transforms the format of the csv file with dumped picking so all the columns are separated
    by `n_spaces` spaces exactly. To make such transform possible you must provide the maximum number
    of digits each column, except the last one, contains. In case, for example, traces are identified
    by the 'FieldRecord' and 'TraceNumber' headers, and their maximum values are 999999 and 9999 respectively,
    `max_len` is `(6, 4)`. Such transform makes it compatible with specific seismic processing software.


    Parameters
    ----------
    path : str
        Path to the file with picking.
    path_save : str, optional
        Path where the result would be stored. By default the file would be overwritten.
    n_spaces : int, default is 8
        The number of spaces separating columns.
    max_len : tuple, default is (6, 4)
        Width of each column except last one.
    """
    if path_save is not None:
        write_object = open(path_save, 'w')
    # in case you want to overwrite the existing file, temporary file would be created.
    # the intermediate results would be saved to this temp file, in the end original file
    # would be replaced with temporary one, afterwards temp file deleted
    else:
        write_object = tempfile.NamedTemporaryFile(mode='w', delete=True)

    with open(path, 'r', newline='') as read_file:
        reader = csv.reader(read_file)
        with write_object as write_file:
            for row in reader:
                for i, item in enumerate(row[:-1]):
                    write_file.write(str(item).ljust(max_len[i] + n_spaces))
                write_file.write(str(row[-1]) + '\n')

            if path_save:
                return
            shutil.copyfile(write_file.name, path)

def collect_components_data(batch, src, pos):
    """ Collect data from the batch. 
    Packs it into 2d array 'O' dtype, where each elent is a component object. """
    if src[0] is None:
        return None        
    data = np.empty((len(pos), len(src)), 'O')
    for i, ipos in enumerate(pos):
        for j, isrc in enumerate(src):
            data[i, j] = getattr(batch, isrc)[ipos]
    return data

def is_2d_array(array):
    return np.isscalar(array[0][0])

def is_1d_array(array):
    return np.isscalar(array[0])

def infer_array(data, cond):
    """ Given the nested(max depth 2) structure of objects,  where the object is defined 
    by the condition (for example: for gather - is_2d_array, for horizon - is_1d_array) rearange it into
    np.2darray with 'O' dtype. Each element of resulted array is a requsted object.
    """
    if data is None:
        return None 

    if cond(data): # data is a desired object, wrap it into 'O' array with shape 1x1 
        blank = np.empty((1, 1), 'O')
        blank[0, 0] = data
    elif cond(data[0]): # data is a iterable of objects, wrap it into array with the shape 1xN
        blank = np.empty((1, len(data)),  'O')
        for i, _ in enumerate(data):
            blank[0, i] = data[i]
    elif cond(data[0][0]):  # data is a nested(depth 2) structure of objects, wrap it into array with the shape MxN
        blank = np.empty((len(data), len(data[0])), 'O')
        for i, _ in enumerate(data):
            for j, _ in enumerate(data[0]):
                blank[i, j] = data[i][j]

    return blank

def to_list(obj, n=1):
    """Cast an object to a list and repeat it n times. Almost identical to `list(obj)` for 1-D
    objects, except for `str`, which won't be split into separate letters but
    transformed into a list of a single element.
    """
    return np.ravel(obj).tolist() * n
     