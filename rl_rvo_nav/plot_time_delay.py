import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.gridspec as gridspec


df = pd.read_csv('/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/Experiments/rl_rvo_1_dis0/time_delay_episode_0_step_0-1_vmax_linear1-5_vmax_angular1-5.csv')
df_zoomed = df[:10]
#################################################
## omni x
#################################################
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('m/s')
axes.set_xlabel('time_stamp')

axes.plot(df['time_stamp_init'],df['action_x'], color = 'red', marker = '.')
axes.plot(df['time_stamp_final'], df['cur_vel_x_init'],  color = 'blue',marker = '.')
#axes.plot(df['time_stamp_final'], df['cur_vel_x_final'],  color = 'magenta',marker = '.')

axes.legend(['cmd_vel_diff_x','odom_diff_x_init', 'odom_diff_x_final'])
name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/time_delay_omni_x.png'
title = 'Time delay'
axes.grid()
axes.set_title(title)
plt.savefig(name)


fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('m/s')
axes.set_xlabel('time_stamp')

axes.plot(df_zoomed['time_stamp_init'],df_zoomed['action_x'], color = 'red', marker = '.')
axes.plot(df_zoomed['time_stamp_final'], df_zoomed['cur_vel_x_init'],  color = 'blue',marker = '.')
#axes.plot(df_zoomed['time_stamp_final'], df_zoomed['cur_vel_x_final'],  color = 'magenta',marker = '.')

name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/time_delay_zoom_omni_x.png'
title = 'Time delay. Zoomed'
axes.grid()
axes.set_title(title)
axes.legend(['cmd_vel_diff_x','odom_diff_x_init', 'odom_diff_x_final'])
plt.savefig(name)


#################################################
## omni y
#################################################
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('m/s')
axes.set_xlabel('time_stamp')

axes.plot(df['time_stamp_init'],df['action_y'],  color = 'red', linestyle = '-',marker = '.')
axes.plot(df['time_stamp_final'],df['cur_vel_y_init'],  color = 'blue', linestyle = '-',marker = '.')
#axes.plot(df['time_stamp_final'],df['cur_vel_y_final'],  color = 'magenta', linestyle = '-',marker = '.')

axes.legend(['cmd_vel_diff', 'odom_diff_y_init','odom_diff_y_final'])
name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/time_delay_omni_y.png'
title = 'Time delay'
axes.grid()
axes.set_title(title)
plt.savefig(name)


fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('m/s')
axes.set_xlabel('time_stamp')

axes.plot(df_zoomed['time_stamp_init'],df_zoomed['action_y'],  color = 'red', linestyle = '-',marker = '.')
axes.plot(df_zoomed['time_stamp_final'],df_zoomed['cur_vel_y_init'],  color = 'blue', linestyle = '-',marker = '.')
#axes.plot(df_zoomed['time_stamp_final'],df_zoomed['cur_vel_y_final'],  color = 'magenta', linestyle = '-',marker = '.')

name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/time_delay_zoom_omni_y.png'
title = 'Time delay. Zoomed'
axes.grid()
axes.set_title(title)
axes.legend(['cmd_vel_diff_x','odom_diff_x_init', 'odom_diff_x_final'])
plt.savefig(name)

#################################################
## linear x
#################################################
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('m/s')
axes.set_xlabel('time_stamp')

axes.plot(df['time_stamp_init'],df['action_linear_x'],  color = 'red', linestyle = '-',marker = '.')
axes.plot(df['time_stamp_init'],df['cur_vel_linear_x_init'],  color = 'blue', linestyle = '-',marker = '.')
#axes.plot(df['time_stamp_final'],df['cur_vel_linear_x_final'],  color = 'magenta', linestyle = '-',marker = '.')

axes.legend(['cmd_vel_linear_x', 'odom_linear_x_init', 'odom_linear_x_final'])
name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/time_delay_linear_x.png'
title = 'Time delay'
axes.grid()
axes.set_title(title)
plt.savefig(name)

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('m/s')
axes.set_xlabel('time_stamp')

axes.plot(df_zoomed['time_stamp_init'],df_zoomed['action_linear_x'],  color = 'red', linestyle = '-',marker = '.')
axes.plot(df_zoomed['time_stamp_init'],df_zoomed['cur_vel_linear_x_init'],  color = 'blue', linestyle = '-',marker = '.')
#axes.plot(df_zoomed['time_stamp_final'],df_zoomed['cur_vel_linear_x_final'],  color = 'magenta', linestyle = '-',marker = '.')

name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/time_delay_zoom_linear_x.png'
title = 'Time delay. Zoomed'
axes.grid()
axes.set_title(title)
axes.legend(['cmd_vel_linear_x', 'odom_linear_x_init', 'odom_linear_x_final'])
plt.savefig(name)

#################################################
## angular z
#################################################
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('m/s')
axes.set_xlabel('time_stamp')

axes.plot(df['time_stamp_init'],df['action_angular_z'],  color = 'red', linestyle = '-',marker = '.')
axes.plot(df['time_stamp_final'],df['cur_vel_angular_z_init'],  color = 'blue', linestyle = '-',marker = '.')
#axes.plot(df['time_stamp_final'],df['cur_vel_angular_z_final'],  color = 'magenta', linestyle = '-',marker = '.')

axes.legend(['cmd_vel_angular_z', 'odom_angular_z_init','odom_angular_z_final'])
name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/time_delay_angular_z.png'
title = 'Time delay'
axes.grid()
axes.set_title(title)
plt.savefig(name)

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.set_ylabel('m/s')
axes.set_xlabel('time_stamp')

axes.plot(df_zoomed['time_stamp_init'],df_zoomed['action_angular_z'],  color = 'red', linestyle = '-',marker = '.')
axes.plot(df_zoomed['time_stamp_final'],df_zoomed['cur_vel_angular_z_init'],  color = 'blue', linestyle = '-',marker = '.')
#axes.plot(df_zoomed['time_stamp_final'],df_zoomed['cur_vel_angular_z_final'],  color = 'magenta', linestyle = '-',marker = '.')

name = '/home/rosfr/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/time_delay_zoom_angular_z.png'
title = 'Time delay. Zoomed'
axes.grid()
axes.set_title(title)
axes.legend(['cmd_vel_angular_z', 'odom_angular_z_init','odom_angular_z_final'])
plt.savefig(name)


