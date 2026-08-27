import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

u = np.linspace(0, 2 * np.pi, 200)
v = np.linspace(0, np.pi, 200)
u, v = np.meshgrid(u, v)

radius = 1.0
x = radius * np.sin(v) * np.cos(u)
y = radius * np.sin(v) * np.sin(u)
z = radius * np.cos(v)

slice_thickness = 0.03

mask_xy = np.abs(z) < slice_thickness
mask_xz = np.abs(y) < slice_thickness

x_xy = np.ma.masked_where(~mask_xy, x)
y_xy = np.ma.masked_where(~mask_xy, y)
z_xy = np.ma.masked_where(~mask_xy, z)

x_xz = np.ma.masked_where(~mask_xz, x)
y_xz = np.ma.masked_where(~mask_xz, y)
z_xz = np.ma.masked_where(~mask_xz, z)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_xy, y_xy, z_xy, color='crimson', alpha=0.9, rstride=1, cstride=1)
ax.plot_surface(x_xz, y_xz, z_xz, color='deepskyblue', alpha=0.9, rstride=1, cstride=1)

ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sphere: X-Y Slice (Red) and X-Z Slice (Blue)')

ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()
