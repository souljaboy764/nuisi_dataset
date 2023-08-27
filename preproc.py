import numpy as np
from visualize import mypause

def rotation_normalization(skeleton):
	leftShoulder = skeleton[6]
	rightShoulder = skeleton[12]
	waist = skeleton[4]

	xAxisHelper = waist - rightShoulder
	yAxis = leftShoulder - rightShoulder	# y axis leftward
	xAxis = np.cross(xAxisHelper, yAxis)	# x axis forward
	zAxis = np.cross(xAxis, yAxis) 			# z axis upward

	xAxis /= np.linalg.norm(xAxis)
	yAxis /= np.linalg.norm(yAxis)
	zAxis /= np.linalg.norm(zAxis)

	rotmat = np.array([
						[xAxis[0], xAxis[1], xAxis[2]],
						[yAxis[0], yAxis[1], yAxis[2]],
						[zAxis[0], zAxis[1], zAxis[2]]
					])

	origin = -rotmat.dot(skeleton[11][:,None])[:, 0]

	T = np.eye(4)
	T[:3,:3] = rotmat
	T[:3,3] = origin

	return T

def readfile(path):
	data = []
	f = open(path)
	for l in f:
		data_l = np.array(list(map(float, l.split(','))))
		if len(data_l)==3:
			data_l = np.concatenate([data_l, np.zeros(200)])
			data_l[3:] = np.nan
		else:
			assert len(data_l) == 203

		data.append(data_l)
	data = np.array(data)
	index = data[:, 0].astype(int)
	times = data[:, 1].astype(np.uint64)
	num_joints = data[:, 2]
	data = data[:, 3:].reshape((-1, 25, 8))

	return index, times, num_joints, data

actions = ['Waving', 'Handshake', 'Rocket', 'Parachute']
trajectory_idx = {
	"Waving":[
				[170,600],
	   			[1220,1600],
				[1650,2030],
				[2190,2630],
				[3380,3830],
				[3900,4330],
				[4400,4840],
				[4880,5280],
				[5760,6190],
				[6260,6670],
				[6710,7150],
				[7320,7710]
			],
	"Handshake":[
				[100,420],
	   			[460,780],
				[800,1111],
				[1145,1440],
				# [1460,1760],
				[1800,2095],
				[2140,2420],
				[2440,2690],
				# [2720,2970],
				[3030,3240],
				[3260,3490],
				[3510,3720],
				[3750,4000],
			],
	"Rocket":[
				[110,310],
	   			[340,550],
				[570,750],
				[770,920],
				[940,1120],
				[1140,1300],
				[1310,1470],
				[1500,1640],
				[1680,1815],
				[1840,2000],
				[2030,2240],
				[2290,2500]
			],
	"Parachute":[
				[230,490],
	   			[530,770],
				[780,1045],
				# [1070,1320],
				[1320,1590],
				# [1610,1850],
				# [1870,2100],
				[2110,2350],
				[2370,2580],
				[2590,2830],
				# [2860,3150],
				[3200,3490],
				[3500,3780],
			],

}

# Further Info for Indices: https://download.3divi.com/Nuitrack/doc/group__SkeletonTracker__group__csharp.html#ga659db18c8af0cb3d660930d7116709ae
active_joints_idx = (1,2,3,4,5,6,7,8,11,12,13,14)
body_idx = (1,2,3,4)
larm_idx = (5,6,7,8)
rarm_idx = (11,12,13,14)


train_data = []
test_data = []
train_labels = []
test_labels =[]
for a in range(len(actions)):
	index_p1, times_p1, num_joints_p1, data_p1 = readfile(f"data/{actions[a]}_p1.txt")
	index_p2, times_p2, num_joints_p2, data_p2 = readfile(f"data/{actions[a]}_p2.txt")

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


	train_split = int(0.8*len(trajs_a))
	train_data += trajs_a[:train_split]
	test_data += trajs_a[train_split:]
	train_labels += (np.ones(train_split)*a).tolist()
	test_labels += (np.ones(len(trajs_a)-train_split)*a).tolist()
	print(len(train_data), len(test_data))

train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data, dtype=object)

np.savez_compressed('nuisi_dataset.npz', train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
