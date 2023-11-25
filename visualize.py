# -*- coding:utf-8 -*-

# -----------------------------------
# 3D Skeleton Display
# Author: souljaboy764
# Date: 2023/08/26
# -----------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from read_nuisi import *
import os

# Thanks to https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7/45734500#45734500
def mypause(interval):
	backend = plt.rcParams['backend']
	if backend in matplotlib.rcsetup.interactive_bk:
		figManager = matplotlib._pylab_helpers.Gcf.get_active()
		if figManager is not None:
			canvas = figManager.canvas
			if canvas.figure.stale:
				canvas.draw()
			canvas.start_event_loop(interval)
			return

if __name__=="__main__":
	for a in range(len(actions)):
		index_p1, times_p1, num_joints_p1, data_p1 = readfile(os.path.join('data', f'{actions[a]}_p1.txt'))
		index_p2, times_p2, num_joints_p2, data_p2 = readfile(os.path.join('data', f'{actions[a]}_p2.txt'))

		data_p1[:,:,[-3,-2,-1]] = data_p1[:,:,[-1,-3,-2]]*0.001
		data_p1[:,:,-2] *= -1

		data_p2[:,:,[-3,-2,-1]] = data_p2[:,:,[-1,-3,-2]]*0.001
		data_p2[:,:,-2] *= -1

		trajs_a = []
		for s in trajectory_idx[actions[a]]:
			T1 = rotation_normalization(data_p1[s[0], :, -3:])
			T2 = rotation_normalization(data_p2[s[0], :, -3:])
			p1 = []
			p2 = []
			for i in range(s[0], s[1]):
				p1.append(T1[:3,:3].dot(data_p1[i, active_joints_idx, -3:].T).T + T1[:3,3])
				p2.append(T2[:3,:3].dot(data_p2[i, active_joints_idx, -3:].T).T + T2[:3,3])

			p1 = np.array(p1)
			p2 = np.array(p2)

			seq_len = s[1] - s[0]
			trajs_a.append(np.concatenate([p1.reshape((seq_len, -1)), p2.reshape((seq_len, -1))], axis=-1))


		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		plt.ion()

		ax.view_init(0, -90)
		# ax.grid(False)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		# ax.set_axis_bgcolor('white')w
		for i in range(5):
			seq_len, D = trajs_a[i].shape 
			data = np.concatenate([trajs_a[i][:,:D//2].reshape(seq_len, D//6, 3), trajs_a[i][:,D//2:].reshape(seq_len, D//6, 3)], axis=-1)
			for frame_idx in range(seq_len):
				ax.cla()
				ax.set_xlabel('X')
				ax.set_ylabel('Y')
				ax.set_zlabel('Z')
				ax.set_facecolor('none')
				ax.set_xlim3d([-0.1, 0.9])
				ax.set_ylim3d([-0.35, 0.65])
				ax.set_zlim3d([-0.75, 0.25])
				# ax.axis('off')

				x = data[frame_idx, :, 0]
				y = data[frame_idx, :, 1]
				z = data[frame_idx, :, 2]
				ax.scatter3D(x, y, z, color='r', marker='o')

				x = 1 - data[frame_idx, :, 3]
				y = 0.3 - data[frame_idx, :, 4]
				z = data[frame_idx, :, 5]
				ax.scatter3D(x, y, z, color='b', marker='o')

				plt.pause(0.05)
				if not plt.fignum_exists(1):
					break
			if not plt.fignum_exists(1):
				break
