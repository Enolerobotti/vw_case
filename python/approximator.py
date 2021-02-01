import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema

def get_path_coords(xy: np.ndarray):
    """:returns curve natural parameter"""
    return np.insert(np.cumsum(((xy[0, 1:]-xy[0, :-1]) ** 2 + (xy[1, 1:]-xy[1, :-1]) ** 2) ** .5), 0, 0)

def smooth(xyz: np.ndarray, window_size=51, polyorder=7, mode='interp', max_grad=2e-5):
    """
    Apply a Savitzky-Golay filter to streamline coordinates
    :param xyz: (np.ndarray) streamline coordinates
    :param window_size: (int) odd int
    :param polyorder: (int) order of poly
    :param mode: (str) mode for the savgol_filter
    :param max_grad: (int) maximum gradient to filter data
    :return: (tuple) natural parameter s, velocity v, smoothed velocity v_hat
    """
    xy = xyz[:2, 1:]
    v = xyz[2, 1:]
    mask = np.abs(np.gradient(v)) <= max_grad
    masked_v = v[mask]
    s = get_path_coords(xy)
    if window_size > masked_v.shape[0]:
        window_size = masked_v.shape[0]
        window_size = window_size-1 if window_size%2 ==0 else window_size
    if window_size > polyorder:
        v_hat = savgol_filter(masked_v, window_size, polyorder, mode=mode)
    else:
        v_hat = np.zeros_like(masked_v)
    return s[mask], masked_v, v_hat

def extr(x: np.ndarray, y: np.ndarray, aggr_func='max', order=3):
    assert aggr_func in 'maxmin'
    comparator = {
        'max': np.greater,
        'min': np.less
    }
    idxs = argrelextrema(y, comparator[aggr_func], order=order)
    return x[idxs], y[idxs]




if __name__ == '__main__':
    res=np.load('data/line3.npy')
    _s, _v, _v_hat = smooth(res)
    first_der = np.gradient(_v)
    fig, axs = plt.subplots(2)
    axs[0].plot(_s, _v)
    axs[0].plot(_s, _v_hat)
    s_max, v_max = extr(_s[_s<6], _v_hat[_s<6], 'max')
    s_min, v_min = extr(_s[_s<6], _v_hat[_s<6], 'min')
    a= np.vstack([s_max, v_max])
    axs[0].scatter(s_min, v_min)
    axs[0].scatter(s_max, v_max)
    axs[1].plot(_s, first_der)
    axs[1].plot(_s, np.zeros(_s.shape))
    # axs[2].plot(_s, np.gradient(first_der))
    for ax in axs:
        ax.set_xlim(0,8)

    plt.show()