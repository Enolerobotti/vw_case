import logging
import numpy as np
import matplotlib.pyplot as plt
from python.utils import ViscousWaveLines
from python.approximator import smooth, extr

time = np.arange(10)*0.001 + 0.02

if __name__ == '__main__':
    co = 'data/thin_p_coordinates.pkl'
    fi = 'data/thin_p_fields.pkl'
    logger = logging.getLogger("vw_case")
    num_of_lines = 5
    for t in time:
        str_time = str(t)[:5]
        logger.info(f"Time is {str_time}")
        try:
            vwl = ViscousWaveLines(co, fi, str_time)
            # -20, -14, -6, 0
            x0 = np.linspace(-11, -8, num_of_lines)
            y0 = 1e-4
            xl, yl, ml = vwl(x0, y0)
            for i, (xx, yy, mm) in enumerate(zip(xl, yl, ml)):
                logger.info(f"line is {i}")
                try:
                    res = np.vstack([xx, yy, mm])
                    _s, _v, _v_hat = smooth(res)
                    s_max, v_max = extr(_s[_s < 6], _v_hat[_s < 6], 'max')
                    s_min, v_min = extr(_s[_s < 6], _v_hat[_s < 6], 'min')
                    np.save(f'data/time_{str_time}_line{i}_data.npy', np.vstack([_s, _v, _v_hat]))
                    np.save(f'data/time_{str_time}_line{i}_max.npy', np.vstack([s_max, v_max]))
                    np.save(f'data/time_{str_time}_line{i}_min.npy', np.vstack([s_min, v_min]))
                except Exception as ex:
                    logger.error(f"Step is failed. Time = {str_time}, Line = {i}\n{ex}")
        except Exception as ex:
            logger.error(f"Step is failed. Time = {str_time} For all lines\n{ex}")

    for line in range(num_of_lines):
        res_max = []
        res_min = []
        for t in time:
            try:
                str_time = str(t)[:5]
                # data =  np.load(f'data/time_{str_time}_line{line}_data.npy')
                _max = np.load(f'data/time_{str_time}_line{line}_max.npy')
                _min = np.load(f'data/time_{str_time}_line{line}_min.npy')
                if _max.shape[1] > 0:
                    rem_max = _max[0, 0]
                    res_max.append(rem_max)
                if _min.shape[1] > 0:
                    rem_min = _min[0, 0]
                    res_min.append(rem_min)
            except FileNotFoundError:
                logger.error(f"Propagation velocity is not found failed. Time = {str_time}, Line = {line}\n{ex}")

        delta = 0.002753
        res_max=np.array(res_max)
        res_min=np.array(res_min)
        c_max = np.gradient(res_max * delta) * 1000
        c_min = np.gradient(res_min * delta) * 1000
        np.save(f"data/Propagation velocity max line={line}.npy", np.vstack([c_max, res_max]))
        np.save(f"data/Propagation velocity max line={line}.npy", np.vstack([c_min, res_min]))

    axes = plt.gca()
    axes.set_xlim([0, 6])
    axes.set_ylim([-2, 2])
    plt.scatter(res_max, c_max)
    plt.scatter(res_min, c_min)
    plt.show()