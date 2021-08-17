import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_arr = np.loadtxt('a.txt',skiprows=1,dtype=float)

#X=np.asarray(plot_arr[:,0])
#Y=np.asarray(plot_arr[:,1])

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
#Z = np.arange(-10, 10, 0.25)

#X, Y = np.meshgrid(X, Y)
#Z=np.asarray(plot_arr[:,2])

#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)

ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

plt.xlabel('Time (ns)')
plt.ylabel('Pressure (MPa)')
plt.show()
