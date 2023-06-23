import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


df = pd.DataFrame(columns = ['radius', 'sr'])
radius = [0.2,0.25,0.3,0.35,0.4, 0.41, 0.42, 0.43, 0.44,0.45,0.5,0.55]
sr = [100,100,100,100,100,100,100,0,35,5,0,0]
df['radius'] = radius
df['sr'] = sr

fig = plt.figure(figsize = [5,5])
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('success rate (%)')
axes.set_xlabel('radius (m)')

axes.plot(np.array(df['radius']),np.array(df['sr']), color = 'red', marker = '.')

name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/success_rate.png'
title = 'Success rate for different radius values'
axes.grid()
axes.set_title(title)
plt.show()
plt.savefig(name, bbox_inches='tight')


df = pd.DataFrame(columns = ['number of robots', 'sr'])
radius = [5,10,12,15,18,20]
sr = [0,0,0,0,0,0]
df['number_robots'] = radius
df['sr'] = sr

fig = plt.figure(figsize = [5,5])
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('success rate (%)')
axes.set_xlabel('number of robots')

axes.plot(np.array(df['number_robots']),np.array(df['sr']), color = 'red', marker = '.')

name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/success_rate_number_robots.png'
title = 'Success rate for different number of robots'
axes.grid()
axes.set_title(title)
plt.show()
plt.savefig(name, bbox_inches='tight')


df = pd.DataFrame(columns = ['linear and angular speed (m/s, rad/s)', 'sr'])
radius = [0.5,0.55,0.60,0.65,0.75,1.0,1.5]
sr = [0,0,0,0,100,100,100]
df['speed'] = radius
df['sr'] = sr

fig = plt.figure(figsize = [5,5])
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('success rate (%)')
axes.set_xlabel('linear and angular speed (m/s, rad/s)')

axes.plot(np.array(df['speed']),np.array(df['sr']), color = 'red', marker = '.')

name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/success_rate_speed.png'
title = 'Success rate for different speed values'
axes.grid()
axes.set_title(title)
plt.show()
plt.savefig(name, bbox_inches='tight')
