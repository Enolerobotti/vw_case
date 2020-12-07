import joblib
import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
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
    def __init__(self, xy, uv, _range=1e-3, resolution = 100):
        assert _range > 0
        self.field = np.hstack([xy, uv])
        self._range = _range
        self.resolution = resolution
        self.limit.terminal = False

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
        Ui = interpolatorU(Xi, Yi)
        Vi = interpolatorV(Xi, Yi)
        return Ui[x_idx, y_idx], Vi[x_idx, y_idx]

    def inverse(self, x, y):
        try:
            u, v = self.velocity_function(x, y[0])
            return -u/v
        except Exception:
            self.limit.terminal = True

    def __call__(self, y, x):
        try:
            u, v = self.velocity_function(x[0], y)
            return -v / u
        except Exception as e:
            self.limit.terminal = True

    @staticmethod
    def limit(t, y):
        return 0


if __name__ == '__main__':
    co = joblib.load('data/thin_p_coordinates.pkl')
    fi = joblib.load('data/thin_p_fields.pkl')
    cl = DataClean()

    scale = 0.002752963
    time = '0.02'

    coords = co[time]
    field = fi[time]['U']
    xy_, field = cl.filter_2d(coords, field)
    # -20, -14, -6, 0
    # x0 = []
    # y0 = []
    # for i in range(-15, -5):
    #     print(i)
    #     sol=solve_ivp(ContinuousField(xy_, field), [0, 0.014], [i * scale], max_step=1e-4, events=limit)
    #     y0.append(sol.t)
    #     x0.append(sol.y[0])
    # plt.plot(np.hstack(x0), np.hstack(y0))
    # plt.show()
    x = -8*scale

    sol_inverse = solve_ivp(ContinuousField(xy_, field).inverse, [x, x+1e-2], [1e-4], max_step=1e-4, events=ContinuousField.limit)
    sol_direct = solve_ivp(ContinuousField(xy_, field), [1e-4, 6e-3], [x], max_step=1e-4, events=ContinuousField.limit)
    plt.plot(sol_inverse.t, sol_inverse.y[0])
    plt.plot(sol_direct.y[0], sol_direct.t)
    plt.show()