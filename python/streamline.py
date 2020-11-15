import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from python.utils import DataClean

co = joblib.load('data/thin_p_coordinates.pkl')
fi = joblib.load('data/thin_p_fields.pkl')
cl = DataClean()

time = '0.02'

coords = co[time]
field = fi[time]['U']
xy, field = cl.filter_2d(coords, field)



xi = np.linspace(coords[:,0].min(), coords[:,0].max(), 100)
yi = np.linspace(coords[:,1].min(), coords[:,1].max(), 100)
triang = tri.Triangulation(xy[:, 0], xy[:, 1])
interpolatorU = tri.LinearTriInterpolator(triang, field[:,0])
interpolatorV = tri.LinearTriInterpolator(triang, field[:,1])
Xi, Yi = np.meshgrid(xi, yi)
Ui = interpolatorU(Xi, Yi)
Vi = interpolatorV(Xi, Yi)
# magnitude for color
magnitude = (Ui ** 2 + Vi ** 2) ** .5

xi_range = xi[25:75]
bias = 0.0009
# start points
start_points=np.vstack([xi_range, np.zeros_like(xi_range)+bias]).T

fig, ax = plt.subplots()
ax.set_xlim(coords[:,0].min(), coords[:,0].max())
ax.set_ylim(coords[:,1].min(), coords[:,1].max())
# lines which are orthogonal to the velocity streamlines
s=ax.streamplot(Xi, Yi, Vi, -Ui, start_points=start_points, color=magnitude, integration_direction='forward')
plt.show()
paths = s.lines.get_paths()
segments = s.lines.get_segments()