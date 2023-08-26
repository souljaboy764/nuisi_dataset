import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn. functional import grid_sample, affine_grid

# Used to subsample both trajectories to the same length
theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device)
def downsample(traj, downsample_len):
	seq_len, n_joint, dims = traj.shape
	with torch.no_grad():
		traj = traj.transpose(1,2,0) # N, D, seq_len
		traj = torch.Tensor(traj).to(device).unsqueeze(2) # N, D, 1 seq_len
		traj = torch.concat([traj, torch.zeros_like(traj)], dim=2) # N, D, 2 seq_len
		
		grid = affine_grid(theta.repeat(n_joint,1,1), torch.Size([n_joint, dims, 2, downsample_len]), align_corners=True)
		traj = grid_sample(traj.type(torch.float32), grid, align_corners=True) # N, D, 2 new_length
		traj = traj[:, :, 0].cpu().numpy() # N, D, new_length
		traj = traj.transpose(2,0,1) # new_length, N, D
		return traj

def rotation_normalization(skeleton):
	leftShoulder = skeleton[4]
	rightShoulder = skeleton[7]
	waist = skeleton[3]
	
	xAxisHelper = waist - rightShoulder
	yAxis = leftShoulder - rightShoulder # Should be right to left but something is wrong with the data/preproc
	xAxis = np.cross(xAxisHelper, yAxis) # out of the human(like an arrow in the back)
	zAxis = np.cross(xAxis, yAxis) # like spine, but straight
	
	xAxis /= np.linalg.norm(xAxis)
	yAxis /= np.linalg.norm(yAxis)
	zAxis /= np.linalg.norm(zAxis)

	return np.array([[xAxis[0], xAxis[1], xAxis[2]],
					 [yAxis[0], yAxis[1], yAxis[2]],
					 [zAxis[0], zAxis[1], zAxis[2]]])

colors_10 = get_cmap('tab10')

# v1
joints = ['head', 'neck', 'torso', 'waist', 'left_shoulder', 'left_elbow', 'left_hand', 'right_shoulder', 'right_elbow', 'right_hand']
joints_idx_map = {}
for i in np.arange(len(joints)):
	joints_idx_map[joints[i]] = i

# joints_idx = np.arange(len(joints)) # For all the joints
joints_idx = [joints_idx_map[j] for j in ['neck', 'right_shoulder', 'right_elbow', 'right_hand']] # Currently only need right arm joints

# Index values of the trajectories after leaving intial and final stationary poses
starts1 = [[50,630], [20,721], [41,677], [53,602], [35,880], [45,916]]
starts2 = [[8,465], [3,386], [7,418], [36,477], [12,717], [10,507]]
num_demos = [25, 24, 19, 30, 24, 27]
dims = ['x','y','z']
actions = ['clapfist2', 'fistbump2', 'handshake2', 'highfive1', 'rocket1', 'wave1']

train_data = []
train_labels = []
test_data = []
test_labels = []

for a in range(len(actions)):
# for a in [2,4]: # hanshake and rocket
	action = actions[a]
	s1 = starts1[a]
	s2 = starts2[a]
	n_demos = num_demos[a]
	data1 = np.load('data/'+action+'_agent1.bag.npz',allow_pickle=True)
	trajs_1 = data1['arr_0'] # Nx10x7 : x, y, z, qx, qy, qz, qw
	T1 = np.eye(4)
	T1[:3,3] = trajs_1[0,7,:3]
	T1[:3,:3] = rotation_normalization(trajs_1[0,:,:3])# - trajs_1[0,3,:3])

	times_1 = np.array([i.to_sec() for i in  data1['arr_1']])
	len1 = len(times_1)
	data2 = np.load('data/'+action+'_agent2.bag.npz',allow_pickle=True)
	trajs_2 = data2['arr_0'] # Nx10x7 : x, y, z, qx, qy, qz, qw
	T2 = np.eye(4)
	T2[:3,3] = trajs_2[0,7,:3]
	T2[:3,:3] = rotation_normalization(trajs_2[0,:,:3])# - trajs_2[0,3,:3])
	times_2 = np.array([i.to_sec() for i in  data2['arr_1']])
	len2 = len(times_2)

	trajs = [trajs_1, trajs_2]
	starts = [s1, s2]
	T = [T1,T2]
	fig, ax = plt.subplots(nrows=2, ncols=1)
	plt.subplots_adjust(top=0.989,
						bottom=0.029,
						left=0.004,
						right=0.996,
						hspace=0.062,
						wspace=0.2)
	# print(action)
	cropped_trajs_ = []
	for i in range(2):
		cropped_trajs_.append([])
		color_counter = 0
		start = max(0, starts[i][0]-2)
		end = min(trajs[i].shape[0], starts[i][1]+2)
		ax[i].set_xlim(0, end-start)
		for dim in range(3):
			mean = trajs[i][start:end, -1, dim].mean()
			ax[i].plot(trajs[i][start:end, -1, dim] - mean, color=colors_10(color_counter%10),label=dims[dim])
			color_counter += 1
		ax[i].legend()
		max_ = trajs[i][start:end, -1].max()
		min_ = trajs[i][start:end, -1].min()
		
		mean_y = trajs[i][start:end, -1, 1].mean()
		y_ = trajs[i][start:end, -1, 1] - mean_y
		mean_z = trajs[i][start:end, -1, 2].mean()
		z_ = trajs[i][start:end, -1, 2] - mean_z 
		idx_filter = ((y_ > 0.1) & (z_ > 0.1)).astype(int)
		idx_filter = np.diff(idx_filter, prepend=1)
		idx_s = np.where(idx_filter>0)[0]-1
		idx_e = np.where(idx_filter<0)[0]+1
		# idx = np.where(y_ > 0.1)[0]
		idx_ = (0.5*(idx_e[1:] + idx_s[:-1])).astype(int)

		assert(len(idx_)==n_demos-1)
		for i_ in idx_:
			ax[i].plot([i_,i_], [y_.min(),y_.max()],'r-')
		
		traj = trajs[i][start:end,:,:3] - T[i][:3,3]
		traj = T[i][:3,:3].dot(traj.transpose((0,2,1))).transpose((1,2,0))
		crop_idx = np.concatenate([np.array([0]), idx_, np.array([end-start-1])])
		for idx in range(len(crop_idx)-1):
			cropped_trajs_[-1].append(traj[crop_idx[idx]:crop_idx[idx+1],joints_idx]) # remove the :3 if the quaternions are needed (unlikely)

		assert(len(cropped_trajs_[-1])==n_demos)
	# plt.show()
	cropped_trajs = []
	cropped_trajs_downsampled = []
	for n in range(n_demos):
		cropped_trajs.append([cropped_trajs_[0][n],cropped_trajs_[1][n]])
		cropped_trajs_downsampled.append(np.concatenate([cropped_trajs_[0][n].reshape(-1,len(joints_idx)*3),downsample(cropped_trajs_[1][n], cropped_trajs_[0][n].shape[0]).reshape(-1,len(joints_idx)*3)],axis=-1))
		# print(cropped_trajs_downsampled[-1].shape)

	for n in range(15):
		# print(cropped_trajs_downsampled[n].shape, cropped_trajs[n][0].shape, cropped_trajs[n][1].shape)
		train_data.append(cropped_trajs_downsampled[n])
		train_labels.append(a)
	for n in range(15,19):
		test_data.append(cropped_trajs_downsampled[n])
		test_labels.append(a)
	# print('')

train_data=np.array(train_data,dtype=object)
test_data=np.array(test_data,dtype=object)

# np.savez_compressed('nuisi_dataset.npz', train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
# np.savez_compressed('nuisi_dataset_rarm.npz', train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)