import numpy as np
import matplotlib.pyplot as plt
from walker import *
from rk4 import *

dt = float(1)
tFinal = 10000.0
t = np.arange(0.0, tFinal, dt)


walkers = []
walkers.append(['walker_1', 53, 50, 32, 1, 500, 360, 0])
walkers.append(['walker_2', 85, 50, 8,  1, 720, 360, 30])

# print(walkers)

constellation = Constellation()
constellation.addWalkers(walkers)

satellites = []
satellites.append(['sat_1', 650, 64, 60, 35])
satellites.append(['sat_2', 650, 50, 60, 35])
satellites.append(['sat_3', 700, 64, 25, 35])
satellites.append(['sat_4', 650, 64, 60, 60])
constellation.addBackupSatellites(satellites)

# stateRVarr = propagateOrbit(a0, ecc0, trueAnomaly0, raan0, inc0, aop0, t, dt)

stateRVarr = Satellite(*satellites[0][1:]).propagateOrbit(t, dt, 1)

theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)

r = 6378135 # m

# Convert to Cartesian coordinates
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)
print(f"Shape of x, y, z: {x.shape}")
print(f"Min and max values - x: ({x.min():.2f}, {x.max():.2f}), y: ({y.min():.2f}, {y.max():.2f}), z: ({z.min():.2f}, {z.max():.2f})")

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='white',  alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Sphere')
ax.set_box_aspect((1, 1, 1))

x = stateRVarr[:, 0]
y = stateRVarr[:, 1]
z = stateRVarr[:, 2]

plt.plot(x, y, z, label='parametric curve')

# plt.savefig('3dOrbitGraph.png')
plt.show()
plt.close()

