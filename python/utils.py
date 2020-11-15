import numpy as np

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