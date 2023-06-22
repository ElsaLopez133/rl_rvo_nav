import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.gridspec as gridspec


df = pd.DataFrame(columns = ['number of robots', 'sr'])
radius = [5,10,12,15,18,20]
sr = [0,0,0,0,0,0]
df['number_robots'] = radius
df['sr'] = sr

fig = plt.figure(figsize = [5,5])
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('success rate (%)')
axes.set_xlabel('number of robots (m)')

axes.plot(np.array(df['number_robots']),np.array(df['sr']), color = 'red', marker = '.')

name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/success_rate_number_robots.png'
title = 'Success rate for different number of robots and speed 0.5m/s - 0.5rad/s'
axes.grid()
axes.set_title(title)
plt.show()
plt.savefig(name)
