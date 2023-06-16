import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.gridspec as gridspec


df = pd.DataFrame(columns = ['radius', 'sr'])
radius = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55]
sr = [100,100,100,100,100,5,0,0]
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
plt.savefig(name)
