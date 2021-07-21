"""Implements functions to load and dump data in various formats"""

import os
import glob
from collections import namedtuple

import segyio
import numpy as np
from tqdm.auto import tqdm

from .general_utils import to_list


def aggregate_segys(in_paths, out_path, recursive=False, mmap=True, keep_exts=("sgy", "segy"), bar=True):
    """Merge several SEG-Y files into a single one.

    Parameters
    ----------
    in_paths : str or list of str
        Glob mask or masks to search for source files to merge.
    out_path : str
        A path to the resulting merged file.
    recursive : bool, optional, defaults to False
        Whether to treat '**' pattern in `in_paths` as zero or more directories to perfrom a recursive file search.
    mmap : bool, optional, defaults to True
        Whether to perform memory mapping of input files. Setting this flag to `True` may result in faster reads.
    keep_exts : None, array-like, optional, defaults to ("sgy", "segy")
        Extensions of files to use for merging. If `None`, no filtering is performed.
    bar : bool, optional, defaults to True
        Whether to show the progres bar.

    Raises
    ------
    ValueError
        If no files match the given pattern.
        If source files contain inconsistent samples.
    """
    in_paths = sum([glob.glob(path, recursive=recursive) for path in to_list(in_paths)], [])
    if keep_exts is not None:
        in_paths = [path for path in in_paths if os.path.splitext(path)[1][1:] in keep_exts]
    if not in_paths:
        raise ValueError("No files match the given pattern")

    # Check whether all files have the same trace length and sample rate
    source_handlers = [segyio.open(path, ignore_geometry=True) for path in in_paths]
    samples = source_handlers[0].samples
    if not all(np.array_equal(samples, handler.samples) for handler in source_handlers[1:]):
        raise ValueError("Source files contain inconsistent samples")

    if mmap:
        for source_handler in source_handlers:
            source_handler.mmap()

    # Create segyio spec for the new file
    spec = segyio.spec()
    spec.samples = samples
    spec.ext_headers = source_handlers[0].ext_headers
    spec.format = source_handlers[0].format
    spec.tracecount = sum(handler.tracecount for handler in source_handlers)

    # Write traces and their headers from source files into the new one
    os.makedirs(os.path.abspath(os.path.dirname(out_path)), exist_ok=True)
    with segyio.create(out_path, spec) as out_handler:
        trace_pos = 0
        for source_handler in tqdm(source_handlers, disable=not bar):
            out_handler.trace[trace_pos : trace_pos + source_handler.tracecount] = source_handler.trace
            out_handler.header[trace_pos : trace_pos + source_handler.tracecount] = source_handler.header
            trace_pos += source_handler.tracecount
        for i in range(out_handler.tracecount):
            out_handler.header[i].update({segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1})

    # Close source SEG-Y file handlers
    for source_handler in source_handlers:
        source_handler.close()


def read_vfunc(path):
    """Read a file with vertical functions in Paradigm Echos VFUNC format.

    The file may have one or more records with the following structure:
    VFUNC [inline] [crossline]
    [x1] [y1] [x2] [y2] ... [xn] [yn]

    Parameters
    ----------
    path : str
        A path to the file.

    Returns
    -------
    vfunc_list : list of namedtuples
        List of loaded vertical functions. Each vfunc is a `namedtuple` with the following fields: `inline`,
        `crossline`, `x` and `y`, where `x` and `y` are 1d `np.ndarray`s with the same length.

    Raises
    ------
    ValueError
        If data length for any VFUNC record is odd.
    """
    vfunc_list = []
    VFUNC = namedtuple("VFUNC", ["inline", "crossline", "x", "y"])
    with open(path) as file:
        for data in file.read().split("VFUNC")[1:]:
            data = data.split()
            inline, crossline = int(data[0]), int(data[1])
            data = np.array(data[2:], dtype=np.float64)
            if len(data) % 2 != 0:
                raise ValueError("Data length for each VFUNC record must be even")
            vfunc_list.append(VFUNC(inline, crossline, data[::2], data[1::2]))
    return vfunc_list


def read_single_vfunc(path):
    """Read a single vertical function from a file in Paradigm Echos VFUNC format.

    The file must have exactly one record with the following structure:
    VFUNC [inline] [crossline]
    [x1] [y1] [x2] [y2] ... [xn] [yn]

    Parameters
    ----------
    path : str
        A path to the file.

    Returns
    -------
    vfunc : namedtuple
        Vertical function with the following fields: `inline`, `crossline`, `x` and `y`, where `x` and `y` are 1d
        `np.ndarray`s with the same length.

    Raises
    ------
    ValueError
        If data length for any VFUNC record is odd.
        If the file does not contain a single vfunc.
    """
    file_data = read_vfunc(path)
    if len(file_data) != 1:
        raise ValueError(f"Input file must contain a single vfunc, but {len(file_data)} were found in {path}")
    return file_data[0]


def dump_vfunc(path, vfunc_list):
    """Dump vertical functions in Paradigm Echos VFUNC format to the file.

    For each vfunc there is corresponding block in the resulted file with the following strcture:
    The first row contains 3 values: VFUNC [inline] [crossline].
    All other rows represent pairs: [time] [velocity]. The single row contains 4 pairs(8 fields),
    except the last one, which may contain less. Each field is left aligned with the width 8.

    Block example
    :::
    VFUNC   22      33
    17      1546    150     1530    294     1672    536     1812
    760     1933    960     2000    1202    2148    1374    2251
    1574    2409    1732    2517    1942    2675
    :::

    Parameters
    ----------
    path : str
        A path to the file.
    vfunc_list : iterable of tuples
        Each tuple corresponds to a vertical function and consists of the following values: `inline`, `crossline`,
        `x` and `y`, where `x` and `y` are 1d np.ndarrays with the same length.
    """
    with open(path, 'w') as f:
        for inline, crossline, times, offsets in vfunc_list:
            f.write('{:8}{:<8}{:<8}\n'.format('VFUNC', inline, crossline))
    
            data = np.dstack([times, offsets])
            rows = np.split(data.flatten(), np.arange(8, data.size, 8))
            for row in rows:
                f.write(''.join(['{:<8.0f}'.format(i) for i in row] + ['\n']))


# pylint: disable=too-many-arguments, invalid-name
def make_prestack_segy(path, survey_size=(1000, 1000), origin=(0, 0), sources_step=(50, 300), recievers_step=(100, 25),
                       bin_size=(50, 50), activation_dist=(500, 500), n_samples=1500, sample_rate=2000, delay=0,
                       trace_gen=None, **kwargs):
    """Make a prestack SEG-Y file with rectangular geometry. Its headers are filled with values inferred from survey
    geometry parameters, traces are filled with data generated by `trace_gen`.

    All tuples in method arguments indicate either coordinate in (`x`, `y`) or distance in (`x_dist`, `y_dist`) format.

    Parameters
    ----------
    path : str
        Path to store the generated SEG-Y file.
    survey_size : tuple of ints, defaults to (1000, 1000)
        Survey dimensions measured in meters.
    origin : tuple of ints, defaults to (0, 0)
        Coordinates of bottom left corner of the survey.
    sources_step : tuple of ints, defaults to (50, 300)
        Distances between sources. (50, 300) are standard values indicating that source lines are positioned along `y`
        axis with 300 meters step, while sources in each line are located every 50 meters along `x` axis.
    recievers_step : tuple of ints, defaults to (100, 25)
        Distances between recievers. It is supposed that reciever lines span along `x` axis. By default distance
        between reciever lines is 100 meters along `x` axis, and distance between recievers in lines is 25 meters
        along `y` axis.
    bin_size : tuple of ints, defaults to (50, 50)
        Size of a CDP bin in meters.
    activation_dist : tuple of ints, defaults to (500, 500)
        Maximum distance from source to active reciever along each axis. Each source activates a rectanglar field of
        recievers with source at its center and shape (2 * activation_dist[0], 2 * activation_dist[1])
    n_samples : int, defaults to 1500
        Number of samples in traces.
    sample_rate : int, defaults to 2000
        Sampling interval in microseconds.
    delay : int, defaults to 0
        Delay time of the seismic trace in milliseconds.
    trace_gen : callable, defaults to None.
        Callable to generate trace data. It recieves a dict of trace headers along with everything passed in `kwargs`.
        If `None`, traces are filled with gaussian noise.
        Passed headers: FieldRecord, TraceNumber, SourceX, SourceY, Group_X, Group_Y, offset, CDP_X, CDP_Y,
                        INLINE_3D, CROSSLINE_3D, TRACE_SAMPLE_COUNT, TRACE_SAMPLE_INTERVAL, DelayRecordingTime
    kwargs : misc, optional
        Additional keyword arguments to `trace_gen`.
    """
    # By default traces are filled with random noise
    if trace_gen is None:
        def trace_gen(TRACE_SAMPLE_COUNT, **kwargs):
            _ = kwargs
            return np.random.normal(size=TRACE_SAMPLE_COUNT).astype(np.float32)

    def generate_coordinates(origin, survey_size, step):
        """ Support function to create coordinates of sources / recievers """
        x, y = np.mgrid[[slice(start, start+size, step) for start, size, step in zip(origin, survey_size, step)]]
        return np.vstack([x.ravel(), y.ravel()]).T

    # Create coordinate points for sources and recievers
    source_coords = generate_coordinates(origin, survey_size, sources_step)
    reciever_coords = generate_coordinates(origin, survey_size, recievers_step)

    # Create and fill up a SEG-Y spec
    spec = segyio.spec()
    spec.format = 5 # 5 stands for IEEE-floating point, which is the standard -
    spec.samples = np.arange(n_samples) * sample_rate / 1000

    # Calculate matrix of active recievers for each source and get overall number of traces
    activation_dist = np.array(activation_dist)
    active_recievers_mask = np.all(np.abs(source_coords[:, None, :] - reciever_coords) <= activation_dist, axis=-1)
    spec.tracecount = np.sum(active_recievers_mask)

    with segyio.create(path, spec) as dst_file:
        # Loop over the survey and put all the data into the new SEG-Y file
        TRACE_SEQUENCE_FILE = 0

        for FieldRecord, source_location in enumerate(tqdm(source_coords)):
            active_recievers_coords = reciever_coords[active_recievers_mask[FieldRecord]]

            # TODO: maybe add trace with zero offset
            for TraceNumber, reciever_location in enumerate(active_recievers_coords):
                TRACE_SEQUENCE_FILE += 1
                # Create header
                header = dst_file.header[TRACE_SEQUENCE_FILE-1]
                # Fill headers dict
                trace_header_dict = {}
                trace_header_dict['FieldRecord'] = FieldRecord
                trace_header_dict['TraceNumber'] = TraceNumber
                trace_header_dict['SourceX'], trace_header_dict['SourceY'] = source_location
                trace_header_dict['GroupX'], trace_header_dict['GroupY'] = reciever_location
                trace_header_dict['offset'] = int(np.sum((source_location - reciever_location)**2)**0.5)

                CDP = ((source_location + reciever_location)/2).astype(int)
                trace_header_dict['CDP_X'], trace_header_dict['CDP_Y'] = CDP
                trace_header_dict['INLINE_3D'], trace_header_dict['CROSSLINE_3D'] = CDP // bin_size

                # Fill depth-related fields in header
                trace_header_dict['TRACE_SAMPLE_COUNT'] = n_samples
                trace_header_dict['TRACE_SAMPLE_INTERVAL'] = sample_rate
                trace_header_dict['DelayRecordingTime'] = delay

                # Generate trace and write it to file
                trace = trace_gen(**trace_header_dict, **kwargs)
                dst_file.trace[TRACE_SEQUENCE_FILE-1] = trace

                # Rename keys in trace_header_dict and update SEG-Y files' header
                trace_header_dict = {segyio.tracefield.keys[k]: v for k, v in trace_header_dict.items()}
                header.update(trace_header_dict)

        dst_file.bin = {segyio.BinField.Traces: TRACE_SEQUENCE_FILE,
                        segyio.BinField.Samples: n_samples,
                        segyio.BinField.Interval: sample_rate}
