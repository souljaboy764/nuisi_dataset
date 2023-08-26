# -*- coding:utf-8 -*-

# -----------------------------------
# 3D Skeleton Display
# Author: souljaboy764
# Date: 2023/08/26
# -----------------------------------

import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

data = np.load('nuisi_dataset.npz', allow_pickle=True)
test_data = data['test_data']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()

ax.view_init(0, -90)
# ax.grid(False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_axis_bgcolor('white')w
for i in range(len(test_data)):
    seq_len, D = test_data[i].shape 
    data = np.concatenate([test_data[i][:,:D//2].reshape(seq_len, D//6, 3), test_data[i][:,D//2:].reshape(seq_len, D//6, 3)], axis=-1)
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
plt.ioff()
plt.show()

