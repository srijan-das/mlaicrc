import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,100, step=1)
y = x*2
z = x**2
'''
fig0 = plt.figure()
axis = fig0.add_axes([0.1,0.1,0.8,0.8])
axis.plot(x,y)
axis.set_xlabel('x')
axis.set_ylabel('y')
axis.set_title('title')
plt.show()
'''
'''
fig1 = plt.figure()
ax1 = fig1.add_axes([0.06,0.01,0.85,0.9])
ax2 = fig1.add_axes([0.16,0.5,0.4,0.3])
ax1.plot(x,y)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.plot(x,y)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.show()
'''
'''
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = fig.add_axes([0.16,0.5,0.4,0.5])
ax1.plot(x,z)
ax2.plot(x,z)
ax2.set_xlim([20,22])
ax2.set_ylim([30,50])
ax2.set_title('Zoom')
plt.show()
'''
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes[0].plot(x, y, lw = 3, ls = '-')
axes[1].plot(x, z, lw = 3, ls = '--')
plt.show()