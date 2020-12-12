from typing import Union, List, Tuple

import joblib
import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

class DataClean:
    def __init__(self, criterion=1e-20):
        self.criterion = criterion

    def check(self, p, n):
        assert np.max(np.abs(p - n)) < self.criterion, f"Too strong criterion = {self.criterion}"

    def filter_2d(self, coords, field):
        mask_p = coords[:, 2] >= 0
        mask_n = coords[:, 2] < 0
        coo_p = coords[mask_p][:, :2]
        coo_n = coords[mask_n][:, :2]
        self.check(coo_p, coo_n)
        fi_p = field[mask_p][:, :2]
        fi_n = field[mask_n][:, :2]
        self.check(fi_p, fi_n)
        return (coo_p + coo_n) / 2, (fi_p + fi_n) / 2


class ContinuousField:
    flag = 1
    def __init__(self, xy, uv, _range=1e-3, resolution = 100, streamlines=False):
        assert _range > 0
        self.field = np.hstack([xy, uv])
        self._range = _range
        self.resolution = resolution
        self.streamlines = streamlines
        ContinuousField.limit.terminal = True

    def get_neighbourhood(self, x, y):
        x_min = x - self._range
        y_min = y - self._range
        x_max = x + self._range
        y_max = y + self._range
        return self.field[(self.field[:, 0] >= x_min) &
                          (self.field[:, 0] <= x_max) &
                          (self.field[:, 1] >= y_min) &
                          (self.field[:, 1] <= y_max)]

    @staticmethod
    def get_bounds(area: np.ndarray):
        return area[:, 0].min(), area[:, 0].max(), area[:, 1].min(), area[:, 1].max()

    @staticmethod
    def adjust(xi, x):
        dx = xi-x
        idx = np.argmin(dx**2)
        return xi - dx[idx], idx

    def velocity_function(self, x, y):
        area = self.get_neighbourhood(x, y)
        x_min, x_max, y_min, y_max = self.get_bounds(area)
        xi = np.linspace(x_min, x_max, self.resolution)
        yi = np.linspace(y_min, y_max, self.resolution)
        xi, x_idx = self.adjust(xi, x)
        yi, y_idx = self.adjust(yi, y)
        triang = tri.Triangulation(area[:, 0], area[:, 1])
        interpolatorU = tri.LinearTriInterpolator(triang, area[:, 2])
        interpolatorV = tri.LinearTriInterpolator(triang, area[:, 3])
        Xi, Yi = np.meshgrid(xi, yi)
        Ui = interpolatorU(Xi, Yi).filled()
        Vi = interpolatorV(Xi, Yi).filled()
        u = Ui[x_idx, y_idx]
        v = Vi[x_idx, y_idx]
        return u, v

    def along_y(self, y, x):
        try:
            u, v = self.velocity_function(x[0], y)
            return u/v if self.streamlines else -v/u  # orthogonal lines to the streamlines
        except Exception as e:
            self.flag = 0

    def along_x(self, x, y):
        try:
            u, v = self.velocity_function(x, y[0])
            return v/u if self.streamlines else -u/v  # orthogonal lines to the streamlines
        except Exception as e:
            self.flag = 0

    def limit(self, t, y):
        return self.flag

    def __call__(self, x_line, y_line):
        magnitude = []
        for x, y in zip(x_line, y_line):
            u, v = self.velocity_function(x, y)
            magnitude.append((u ** 2 + v ** 2) ** .5)
        return np.array(magnitude)


class ViscousWaveLines:
    def __init__(self, xy_fn, uv_fn, time, delta=0.002752963, dx=6, max_step=0.05, continious_field_range=1):
        coordinates = joblib.load(xy_fn)[time]
        velocity = joblib.load(uv_fn)[time]['U']
        self.xy, self.uv = DataClean().filter_2d(coordinates, velocity)
        self.time = time
        self.delta = delta
        self.dx = dx * delta
        self.max_step = max_step * delta
        self.continious_field_range = continious_field_range * delta

    def get_line(self, x, y):
        x = x * self.delta
        y = y * self.delta
        cf_forward = ContinuousField(self.xy, self.uv, _range=self.continious_field_range)
        sol_forward = solve_ivp(cf_forward.along_y, (y, y + self.dx), [x], max_step=self.max_step, events=cf_forward.limit)
        if 'termination' in sol_forward.message:
            x0 = sol_forward.y_events[0][0][0]
            y0 = sol_forward.t_events[0][0]
            cf_backward = ContinuousField(self.xy, self.uv, _range=self.continious_field_range)
            sign = 1 if x0 > x else -1
            sol_backward = solve_ivp(cf_backward.along_x, (x0, x0 + sign * self.dx), [y0], max_step=self.max_step, events=cf_backward.limit)
            x_line = np.hstack([sol_forward.y[0], sol_backward.t])
            y_line = np.hstack([sol_forward.t, sol_backward.y[0]])
        else:
            x_line = sol_forward.y[0]
            y_line = sol_forward.t
        magnitude = cf_forward(x_line, y_line)
        return x_line/ self.delta, y_line/ self.delta, magnitude

    def __call__(self,
                 points_x: Union[List[float], Tuple[float], np.ndarray],
                 points_y: Union[List[float], Tuple[float], np.ndarray, float]):
        x_lines, y_lines, magnitude = [], [], []
        if isinstance(points_y, list) or isinstance(points_y, tuple) or isinstance(points_y, np.ndarray):
            assert len(points_x) == len(points_y), "Expected lists of same length"
            for x, y in zip(points_x, points_y):
                x_line, y_line, mag = self.get_line(x, y)
                x_lines.append(x_line)
                y_lines.append(y_line)
                magnitude.append(mag)
        else:
            for x in points_x:
                x_line, y_line, mag = self.get_line(x, points_y)
                x_lines.append(x_line)
                y_lines.append(y_line)
                magnitude.append(mag)
        return x_lines, y_lines, magnitude

if __name__ == '__main__':
    co = 'data/thin_p_coordinates.pkl'
    fi = 'data/thin_p_fields.pkl'

    vwl = ViscousWaveLines(co, fi, '0.02')
    # -20, -14, -6, 0
    x0 = np.linspace(-11, -8, 5)
    y0 = 1e-4
    xl, yl, ml = vwl(x0, y0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for xx, yy, mm in zip(xl, yl, ml):
        mm[mm >= 1e10] = np.nan
        ax.plot(xs=xx, ys=yy, zs=mm)
    plt.show()