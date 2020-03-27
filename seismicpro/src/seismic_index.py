"""Index for SeismicBatch."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import segyio

from ..batchflow import DatasetIndex

from .utils import make_bin_index, build_sps_df, build_segy_df
from .plot_utils import show_2d_heatmap, show_1d_heatmap


class TraceIndex(DatasetIndex):
    """Index for individual seismic traces.

    Parameters
    ----------
    kwargs : dict
        Named arguments for ```build_df``` method.
        Can be either a set of ```dfr```, ```dfs```, ```dfx``` arguments for
        building index from SPS files, or named arguments for ```batchflow.FilesIndex```
        for building index from SEGY files.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    _idf : DataFrame
        DataFrame with rows corresponding to seismic traces and columns with metadata about
        traces. Set of columns includes FieldRecord, TraceNumber, TRACE_SEQUENCE_FILE, file_id and
        a number of extra_headers for index built from SEGY files or FieldRecord, TraceNumber and
        extra SPS file columns for index built from SPS files.
    """
    def __init__(self, *args, index_name=None, **kwargs):
        self.meta = {}
        self._idf = pd.DataFrame()
        self._idf.index.name = index_name
        super().__init__(*args, **kwargs)

    @property
    def tracecounts(self):
        """Return a number of indexed traces for each index."""
        return self._idf.groupby(self._idf.index, sort=False).size().values

    @property
    def name(self):
        """Return a number of indexed traces."""
        return self._idf.index.name

    def get_df(self, index=None, reset=True):
        """Return index DataFrame.

        Parameters
        ----------
        index : array-like, optional
            Subset of indices to loc from DataFrame. If None, get all the DataFrame.
        reset : bool, default to True
            Reset named DataFrame index.

        Returns
        -------
        df : DataFrame
            Index DataFrame.
        """
        if index is None:
            df = self._idf
        else:
            df = self._idf.loc[index]

        if reset:
            return df.reset_index(drop=self.name is None)

        return df

    def head(self, *args, **kwargs):
        """Return the first n rows of the index DataFrame.

        Parameters
        ----------
        args : misc
            Positional arguments to ```DataFrame.head```.
        kwargs : dict
            Named arguments to ```DataFrame.head```.

        Returns
        -------
        First n rows of the index DataFrame.
        """
        return self._idf.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        """Return the last n rows of the index DataFrame.

        Parameters
        ----------
        args : misc
            Positional arguments to ```DataFrame.tail```.
        kwargs : dict
            Named arguments to ```pDataFrame.tail```.

        Returns
        -------
        Last n rows of the index DataFrame.
        """
        return self._idf.tail(*args, **kwargs)

    def filter(self, columns, cond):
        """Filter DataFrame by condition. Only rows that meet the condition will kept.

        Parameters
        ----------
        columns : str, tuple or list
            Group of columns that should meet the condition.
        cond : callable
            Condition to be evaluated.

        Returns
        -------
        index : type(self)
            Filtered index.
        """
        df = self.get_df()
        if isinstance(df[columns], pd.Series):
            df = df.loc[df[columns].apply(cond)]
        else:
            df = df.loc[df[columns].apply(cond, axis='columns')]

        if self.name is not None:
            df.set_index(self.name, inplace=True)

        indices = df.index.unique().sort_values()
        return type(self).from_index(index=indices, idf=df, index_name=self.name)

    def duplicated(self, keep='first'):
        """Get mask of duplicated ('FieldRecord', 'TraceNumber') pairs.

        Parameters
        ----------
        keep :  {‘first’, ‘last’, False}, default ‘first’
            ‘first’ : Mark duplicates as True except for the first occurrence.
            ‘last’ : Mark duplicates as True except for the last occurrence.
            False : Mark all duplicates as True.

        Returns
        -------
        mask : Series
            Mask of duplicated items.
        """
        subset = [('FieldRecord', ''), ('TraceNumber', '')]
        return self.get_df().duplicated(subset=subset, keep=keep)

    def drop_duplicates(self, keep='first'):
        """Drop duplicated ('FieldRecord', 'TraceNumber') pairs."""
        subset = [('FieldRecord', ''), ('TraceNumber', '')]
        df = self.get_df().drop_duplicates(subset=subset, keep=keep)
        if self.name is not None:
            df.set_index(self.name, inplace=True)
        indices = df.index.unique().sort_values()
        return type(self).from_index(index=indices, idf=df, index_name=self.name)

    def merge(self, x, **kwargs):
        """Merge two DataFrameIndex on common columns.

        Parameters
        ----------
        x : DataFrame
            DataFrame to merge with.
        kwargs : dict
            Named arguments to ```DataFrame.merge```.

        Returns
        -------
        df : DataFrame
            Merged DataFrame
        """
        idf = self.get_df()
        xdf = x.get_df()
        df = idf.merge(xdf, **kwargs)
        if self.name is not None:
            df.set_index(self.name, inplace=True)
        indices = df.index.unique().sort_values()
        return type(self).from_index(index=indices, idf=df, index_name=self.name)

    def concat(self, other):
        """ Concatenate vertically current and `other` indices DataFrames.

        Parameters
        ----------
        other : Index or sequence of Index
            Indices which dataframes are being concatenated.

        Returns
        -------
            : TraceIndex
            Concatenated index.
        """
        if not isinstance(other, (list, tuple)):
            other = [other]

        df = [self.get_df()]
        other_dfs = [index.get_df() for index in other]
        concat_dfs = pd.concat(df + other_dfs, ignore_index=True)

        if self.name is not None:
            concat_dfs.set_index(self.name, inplace=True)
        indices = concat_dfs.index.unique().sort_values()
        return type(self).from_index(index=indices, idf=concat_dfs, index_name=self.name)

    def __add__(self, other):
        return self.concat(other)

    def build_index(self, index=None, idf=None, **kwargs):
        """Build index."""
        if index is not None:
            if idf is not None:
                return self.build_from_index(index, idf)
            idf = index.get_df()
            if self.name is not None:
                idf.set_index(self.name, inplace=True)

            self._idf = idf.sort_index()
            return self._idf.index.unique()

        df = self.build_df(**kwargs)
        df.reset_index(drop=df.index.name is None, inplace=True)
        if self.name is not None:
            df.set_index(self.name, inplace=True)

        self._idf = df.sort_index()
        return self._idf.index.unique()

    def build_df(self, **kwargs):
        """Build DataFrame."""
        if 'dfx' in kwargs.keys():
            return build_sps_df(**kwargs)

        return build_segy_df(**kwargs)

    def build_from_index(self, index, idf):
        """Build index from another index for indices given."""
        self._idf = idf.loc[index]
        return index

    def create_subset(self, index):
        """Return a new Index based on the subset of indices given."""
        return type(self).from_index(index=index, idf=self._idf, index_name=self.name)


class SegyFilesIndex(TraceIndex):
    """Index for SEGY files.

    Parameters
    ----------
    name : str
        Name that will be associated with traces of SEGY files.
    kwargs : dict
        Named arguments for ```batchflow.FilesIndex```.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    _idf : DataFrame
        DataFrame with rows corresponding to seismic traces and columns with metadata about
        traces. Columns include FieldRecord, TraceNumber, TRACE_SEQUENCE_FILE, file_id and
        a number of extra_headers if specified.
    """
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = ('file_id', kwargs.get('name'))
        super().__init__(*args, **kwargs)


class CustomIndex(TraceIndex):
    """Index for any SEGY header.

    Parameters
    ----------
    name : str
        Any segyio.TraceField keyword that will be set as index.
    kwargs : dict
        Named arguments for ```batchflow.FilesIndex````.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    _idf : DataFrame
        DataFrame with rows corresponding to seismic traces and columns with metadata about
        traces. Columns include FieldRecord, TraceNumber, TRACE_SEQUENCE_FILE, file_id and
        a number of extra_headers if specified.
    """
    def __init__(self, *args, **kwargs):
        index_name = kwargs['index_name']
        if index_name is not None:
            extra_headers = kwargs.get('extra_headers', [])
            if extra_headers == 'all':
                extra_headers = [h.__str__() for h in segyio.TraceField.enums()]

            kwargs['extra_headers'] = list(set(extra_headers + [index_name]))
        super().__init__(*args, **kwargs)


class KNNIndex(TraceIndex):
    """Index for groups of k nearest located seismic traces.

    Parameters
    ----------
    n_neighbors : int
        Group size parameter.
    raise_error: bool
        Wheather to raise error in case 2 recievers with the same coordinates found on the same shot, default True
    kwargs : dict
        Named arguments for ```batchflow.FilesIndex````.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    _idf : DataFrame
        DataFrame with rows corresponding to seismic traces and columns with metadata about
        traces. Columns include FieldRecord, TraceNumber, TRACE_SEQUENCE_FILE, file_id and
        a number of extra_headers if specified.
    """
    def __init__(self, *args, raise_error=True, **kwargs):
        self.raise_error = raise_error
        kwargs['index_name'] = 'KNN'
        super().__init__(*args, **kwargs)

    def build_df(self, n_neighbors, **kwargs):
        """Build DataFrame."""
        extra_headers = kwargs.get('extra_headers', [])
        if extra_headers == 'all':
            extra_headers = [h.__str__() for h in segyio.TraceField.enums()]

        kwargs['extra_headers'] = list(set(extra_headers + ['CDP_X', 'CDP_Y']))
        field_index = FieldIndex(**kwargs)
        dfs = []
        for fid in field_index.indices:
            df = field_index.get_df([fid])
            data = np.stack([df['CDP_X'], df['CDP_Y']]).T
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
            _, indices = nbrs.fit(data).kneighbors(data)
            if not np.all(indices[:, 0] == np.arange(len(data))):
                if self.raise_error:
                    raise ValueError("Faild to build KNNIndex. Duplicated CDP.")

            dfs.append(df.iloc[np.hstack(indices)])
        df = pd.concat(dfs).reset_index(drop=True)
        indices = np.repeat(np.arange(field_index.tracecounts.sum()), n_neighbors)
        df['KNN'] = indices
        return df


class FieldIndex(TraceIndex):
    """Index for field records.

    Parameters
    ----------
    kwargs : dict
        Named arguments for ```build_df```` method.
        Can be either a set of ```dfr```, ```dfs```, ```dfx``` arguments for
        building index from SPS files, or named arguments for ```batchflow.FilesIndex```
        for building index from SEGY files.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    _idf : DataFrame
        DataFrame with rows corresponding to seismic traces and columns with metadata about
        traces. Set of columns includes FieldRecord, TraceNumber, TRACE_SEQUENCE_FILE, file_id and
        a number of extra_headers for index built from SEGY files or SPS file columns for index
        built from SPS files.
    """
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = 'FieldRecord'
        super().__init__(*args, **kwargs)


class BinsIndex(TraceIndex):
    """Index for bins of CDP.

    Parameters
    ----------
    dfr : DataFrame
        SPS R file data.
    dfs : DataFrame
        SPS S file data.
    dfx : DataFrame
        SPS X file data.
    bin_size : scalar or tuple of scalars
        Grid bin size.
    origin : array-like
        Grid origin coordinates.
    phi : scalar or array-like
         Grid orientation.
    iters : int
        Maxiimal number of iterations for grid optimization algorithm.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    _idf : DataFrame
        DataFrame with rows corresponding to seismic traces and columns with metadata about
        traces. Set of columns includes FieldRecord, TraceNumber and extra SPS file columns.
    """
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = 'bin_id'
        super().__init__(*args, **kwargs)

    def build_df(self, **kwargs):
        """Build DataFrame."""
        df, meta = make_bin_index(**kwargs)
        self.meta.update(meta)
        return df

    def show_heatmap(self, **kwargs):
        """2d histogram of CDP distribution between bins."""
        bin_size = self.meta['bin_size']
        if isinstance(bin_size, (list, tuple, np.ndarray)):
            show_2d_heatmap(self._idf, **kwargs)
        else:
            show_1d_heatmap(self._idf, **kwargs)


class RecIndex(TraceIndex):
    """ Index for recievers.
    It uses `Group` columns of the dataframe as index. `Group` is built by concatenation
    of `GroupX` and `GroupY` columns. If any of `GroupX` and `GroupY` is not present,
    `Group` is not built and `RecIndex` will raise an error.

    Parameters
    ----------
    kwargs : dict
        Named arguments for ```build_df```` method.
        Can be either a set of ```dfr```, ```dfs```, ```dfx``` arguments for
        building index from SPS files, or named arguments for ```batchflow.FilesIndex```
        for building index from SEGY files.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    _idf : DataFrame
        DataFrame with rows corresponding to seismic traces and columns with metadata about
        traces. Set of columns includes FieldRecord, TraceNumber, TRACE_SEQUENCE_FILE, file_id and
        a number of extra_headers for index built from SEGY files or SPS file columns for index
        built from SPS files.
    """
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = 'Group'
        super().__init__(*args, **kwargs)
