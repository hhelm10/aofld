import numpy as np

from sklearn.preprocessing import StandardScaler as SS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as bas

import mne
import os
from scipy import signal

def concatenate_features(ts, axes=[-2, -1]):
    if isinstance(ts, list):
        return [concatenate_features(ts_, axes=axes) for ts_ in ts]

    shape = np.array(ts.shape)

    for i, ax in enumerate(axes):
        if ax < 0:
            axes[i] = len(shape) + ax

    reshape_inds = [i for i in range(len(shape)) if i not in axes]

    ts = ts.reshape((*shape[reshape_inds], np.prod(shape[axes])))
    return ts
    

def get_raw_edfs(file_path):
    datafiles = []

    for f in np.sort(os.listdir(file_path)):
        if "." in f:
            if f[-3:] == 'edf':
                if f[-6:] == 'md.edf':
                    continue
                datafiles.append(file_path + f)
        else:
            try:
                for f_ in os.listdir(file_path + f):
                    if f_[-3:] == 'edf':
                        if f_[-6:] == 'md.edf':
                            continue
                        datafiles.append(file_path + f + "/" + f_)
            except:
                pass

    return [mne.io.read_raw_edf(f, verbose=False) for f in datafiles]

def _normalize_dimensions(ts):
    """ 
    Normalizes the format of the time series to be (n_channels, n_timesteps).
    
    Input
    ts - 1-d or 2-d array or list of lists of the same size.
    If 1-d then it is assumed that len(ts) == n_timesteps. 
    If 2-d then ts is returned.

    Output
    ts reshaped to be (n_channels, n_timesteps)
    """

    if not isinstance(ts, np.ndarray):
        if isinstance(ts, list):
            if isinstance(ts[0], ARRAYLIKE):
                if not isinstance(ts[0][0], NUMERIC):
                    raise ValueError("ts must be a 1-d or 2-d array or a list of non-nested lists")
            try:
                ts = np.array(ts)
            except:
                raise ValueError("ts must be a 1-d or 2-d array or a list of lists of the same size.")
        else:
            raise ValueError("ts must be a 1-d or 2-d array or a list of lists of the same size.")

    if ts.ndim == 1:
        ts = ts.reshape((1, len(ts)))
    elif ts.ndim > 2:
        raise ValueError("ts must be a 1-d or 2-d array or a list of lists of the same size.")

    return ts


def get_bandwidths(ts, order=10, sfreq=500, bands=[[1,4], [4,7], [8,12], [12, 25]], band_axis=0):
    """
    Apply butterworth band filters.

    Input
    ts - 2d array or list of 1d arrays or list of non-nested lists
    Time series to apply band filters to.

    order - int
    Order of the butterworth filter.

    sfreq - int
    Sampling frequency of the time series.

    bands - list or list of lists
    Set of bands to extract from each time series in ts.

    band_axis - int
    Axis of array to put the band dimension.


    Output
    banded_ts - array-like
    Time series after the bands have been applied.

    """

    if isinstance(ts, list):
        return [get_bandwidths(ts_, order=order, sfreq=sfreq, bands=bands, band_axis=band_axis) for ts_ in ts]

    if not isinstance(ts, np.ndarray):
        raise ValueError("ts must be an numpy array")

    ts_shape = ts.shape

    if ts.ndim == 1:
        ts = _normalize_dimensions(ts)

    n_bands = len(bands)

    ts_banded = np.zeros((n_bands, *ts_shape))

    cart_prod = list(itertools.product(*[np.arange(shape) for shape in ts_shape[:-1]]))

    for i, band in enumerate(bands):
        sos_ = signal.butter(order, band, btype='bandpass', fs=sfreq, output='sos')

        for prod in cart_prod:
            prod = tuple(prod)
            indices = (i, *prod)
            ts_banded[indices] = signal.sosfilt(sos_, ts[prod])


    tranpose_axes = np.insert(np.arange(1, ts.ndim+1), band_axis, 0)

    ts_banded = ts_banded.transpose((*tranpose_axes))
              
    return ts_banded


def get_power_spectrum_ratio(ts, sfreq=500, bands = [[3.5,7.5], [7.5,12.5], [12.5, 30.5]]):
    """
    """

    if isinstance(ts, list):
        return [get_power_spectrum_ratio(ts_, sfreq=sfreq, bands=bands) for ts_ in ts]

    if ts.ndim == 1:
        ts = _normalize_dimensions(ts)
    
    f, Pxx = signal.periodogram(ts, fs=sfreq, axis=-1)

    Pxx_shape = Pxx.shape
    Pxx = Pxx.transpose((-1, *np.arange(len(Pxx_shape)-1)))

    masses = np.zeros((len(bands), *Pxx_shape[:-1]))
    
    for i, band in enumerate(bands):
        min_, max_ = band
        
        larger = f >= min_
        smaller = f <= max_
        both_larger_and_smaller = larger * smaller

        masses[i] = np.sum(Pxx[both_larger_and_smaller], axis=0)

    masses = masses.transpose((*np.arange(1, len(Pxx_shape)), 0))
        
    return masses / np.sum(masses, axis=-1, keepdims=True)


def get_time_windows(ts, sfreq=500, n_seconds_per_window=5, n_seconds_overlap=0):
    """
    Window the time series.

    Input
    ts - array-like
    Array of times series to be windowed.

    sfreq - int
    Sampling frequency of the time series.

    n_seconds_per_window - float
    Number of seconds to include in each window.

    n_seconds_overlap - float
    Number of seconds each window should overlap.


    Output
    ts_windowed - array-like
    Windowed time series.

    """

    if isinstance(ts, list):
        return [get_time_windows(ts_, sfreq=sfreq, n_seconds_per_window=n_seconds_per_window, n_seconds_overlap=n_seconds_overlap) for ts_ in ts]

    if not isinstance(ts, np.ndarray):
        raise ValueError("ts must be an array")

    if ts.ndim == 1:
        ts = _normalize_dimensions(ts)


    if not isinstance(sfreq, int):
        if sfreq < 0:
            raise ValueError("sfreq must be a positive integer")


    if not isinstance(n_seconds_per_window, (int, float, np.int64)):
        raise ValueError("n_seconds_per_window must be numeric")

    if n_seconds_per_window <= 0:
        raise ValueError("n_seconds_per_window be positive")


    if not isinstance(n_seconds_overlap, (int, float, np.int64)):
        raise ValueError("n_seconds_overlap must be numeric")

    if n_seconds_per_window < 0:
        raise ValueError("n_seconds_overlap be positive")

    n_per_window = int(sfreq * n_seconds_per_window)
    n_overlap = int(sfreq * n_seconds_overlap)

    return _get_time_windows(ts, n_per_window, n_overlap)


def _get_time_windows(ts, n_per_window, n_overlap):
    """
    Internal get_time_windows function in the frequency domain.
    """

    shape = ts.shape

    if len(shape) == 1:
        ts = _normalize_dimensions(ts)

    n_timesteps = shape[-1]

    n_slide = n_per_window - n_overlap
    n_windows = int(np.math.ceil((n_timesteps - n_per_window) / n_slide) + 1)

    ts = ts.transpose((len(shape) -1, *np.arange(len(shape) - 1)))
    
    ts_windowed = np.zeros((n_per_window, n_windows, *shape[:-1]))

    for i in range(n_windows-1):
        ts_windowed[:, i] = ts[i*n_slide: n_per_window + i*n_slide]
            
    ts_windowed[:, n_windows-1] = ts[n_timesteps - n_per_window:]
    
    return ts_windowed.transpose((*np.arange(1, len(shape)+1), 0))

def even_sample_inds(y, p=0.8):
    unique, counts = np.unique(y, return_counts=True)
    by_unique = []
    min_ = min(counts)
    
    for c in unique:
        inds = np.random.choice(np.where(y == c)[0], size=min_, replace=False)
        by_unique.append(inds)
                
    return np.concatenate(by_unique).astype(int)

def get_stratified_train_test_inds(y, p_train=0.8):
    unique, counts = np.unique(y, return_counts=True)
    
    n_samples_by_class = [int(np.max([1, np.math.floor(p_train * c)])) for c in counts]
    
    train = []
    
    for i, c in enumerate(unique):
        train.append(np.random.choice(np.where(y == c)[0], size=n_samples_by_class[i], replace=False))
        
    train = np.concatenate(train)    
    test = [i for i in range(len(y)) if i not in train]
    
    return train, test

def train_scalers_and_ldas(X, y, p=1, return_inds=False):
    lda_list = []
    ss_list = []
    inds = []

    n_subjs=len(X)
    
    for j in range(n_subjs):
    
        in_bag_inds = even_sample_inds(y[j], p=1)
        inds.append(in_bag_inds)
        
        ss = SS()
        ss.fit(X[j][in_bag_inds])
        ss_list.append(ss)

        lda = LDA(priors=[0.5, 0.5])
        lda.fit(ss.transform(X[j][in_bag_inds]), y[j][in_bag_inds])
        lda_list.append(lda)
        
    if return_inds:
        return ss_list, lda_list, inds

    return ss_list, lda_list


def get_oracle_alpha(X, y, in_task, combined, h=0.01):    
    oracle_bacc = 0
    alpha_grid = np.arange(0,1+h, h)
    for i, alpha in enumerate(alpha_grid):
        temp_proj = alpha * in_task + (1 - alpha) * combined
        # temp_proj /= np.linalg.norm(temp_proj)

        t = X @ temp_proj

        preds = (t >= 0).astype(int)
        temp_bacc = bas(y, preds)

        if temp_bacc >= oracle_bacc:
            oracle_bacc = temp_bacc
            alpha_oracle = alpha
            
    return alpha_oracle


def experiment(X, y, p=0.8):
    """
    Input
    X - list of features of length n_sessions which has elements of shape (n_samples, n_features). 
            The number of samples across sessions can differ but n_features must be the same.
    y - list of labels of length n_sessions which has elements of length n_samples. 
    p - proportion of data to use as training

    Return
    accuracies - accuracies of three models for each subject.

    """
    n_subjs = len(X)
    n_features = X[0].shape[1]
    
    accuracies = np.zeros((n_subjs, 3))
    
    ss_list, lda_list, inds = train_scalers_and_ldas(X, y, return_inds=True)
    
    for i in range(n_subjs):
        subj_inds = inds[i]
        
        train, test = get_stratified_train_test_inds(y[i][subj_inds], p_train=p)
        
        y_train = y[i][subj_inds][train]
        y_test = y[i][subj_inds][test]
        
        new_ss = SS().fit(X[i][train])
        new_lda = LDA(priors=[0.5, 0.5]).fit(new_ss.transform(X[i][train]), y_train)
        
        in_proba = new_lda.predict_proba(new_ss.transform(X[i][test]))
        out_proba = [lda.predict_proba(X[i][test]) for j, lda in enumerate(lda_list) if j != i]
        
        out_proba = np.mean(out_proba, axis=0)
        
        accuracies[i, 0] = bas(y_test, np.argmax(in_proba, axis=1))
        accuracies[i, 1] = bas(y_test, np.argmax(out_proba, axis=1))
        
        alpha_grid = np.arange(0, 1.001, step=0.001)
        
        accuracies[i,2] = accuracies[i,1]
        for alpha in alpha_grid[1:]:
            temp_proba = np.average([in_proba, out_proba], weights=[alpha, 1-alpha], axis=0)
            acc = bas(y_test, np.argmax(temp_proba, axis=1))
            
            if acc > accuracies[i,2]:
                accuracies[i,2] = acc
                                        
    return accuracies