import tensorflow_graphics.geometry.transformation.euler as tfe

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import kinematic_utils as K


subject = 5
trial = 2
# the initial frame thats visualized in qualitative evaluation
fragment_start = 300

eval_ms = [80, 160, 320, 400]
eval_actions = ['walking', 'eating', 'smoking', 'discussion', 'directions',
				'greeting', 'phoning', 'posing', 'purchases', 'sitting', 'sittingdown', 
				'takingphoto', 'waiting', 'walkingdog', 'walkingtogether']

def ms2frame(ms, fps=25):
	return int(ms/1000. * fps - 1)

def frame2sec(frame, fps=25):
	return frame / float(fps)

def create_eval_data(dataset, protocol=None):
	'''
	If protocol == 'std':
	get evaluation sequences using the standard protocol as used in:
    SRNN (https://arxiv.org/pdf/1511.05298.pdf)
    QuaterNet (https://arxiv.org/abs/1805.06485)
    Martinez et. al. (https://arxiv.org/pdf/1705.02445.pdf)
    if protocol == 'quaternet':
	get evaluation sequences using the extended protocol as proposed in:
    QuaterNet (https://arxiv.org/abs/1805.06485)
    '''

	if protocol == 'std' or protocol is None:
		n_samples = 4*2
	elif protocol == 'quaternet': 
	    n_samples = 128*2
	else:
		raise NotImplementedError(f'The evaluation protocol {protocol} is not implemented.')

	eval_sequences = {}

	for action in dataset.action_names:
		eval_sequences[action] = dataset.get_eval_sequences(
			action, n_samples=n_samples)

	return eval_sequences


def run_quantitative_eval(dataset, model=None, max_len=32, metric='srnn', protocol='std', batch_size=16):
	'''
	Run quantitative evaluation under the standard protocol (protocol = 'std') or the protocol proposed by
	Pavllo et. al. (protocol = 'quaternet').

    Setting protocol = 'std' uses a common evaluation scheme for H3.6M.
    This scheme has its origin in SRNN by Jain et. al. (https://arxiv.org/pdf/1511.05298.pdf)
    It evaluates the error of the euler angles between prediction and target:
        mean(sqrt(sum(square(prediction - target)))
        The sum is taken over one state of the skeleton, i.e. 32 joints x 3 dimensions

    Pavllo et. al. noted that the standard protocol is showing high variance, as it uses only 4 samples per motion sequence
    and instead performed evaluation using 128 samples per sequence.
	'''
	# gathers quantitative reconstruction errors
	results = {}

	srnn_metric = lambda x, y: tf.sqrt(tf.reduce_sum((y - x)**2, axis=(-1,-2)))
	mse_metric = lambda x, y: tf.reduce_mean((y - x)**2, axis=(-1,-2))

	if metric == 'srnn':
		metric_fn = srnn_metric
	elif metric == 'mse':
		metric_fn = mse_metric
	else:
		raise NotImplementedError('The metric ' + metric + 'is not implemented.')

	eval_data = create_eval_data(dataset, protocol)
	
	for action in eval_actions:

		test_data = eval_data[action]

		batch_x = [ prefix[-max_len:] for prefix, target in test_data ]
		future_steps = len(test_data[0][-1])

		zero_velocity = [ prefix[-1:] for prefix, target in test_data ]
		batch_y = [ target for prefix, target in test_data ]

		batch_x = np.stack(batch_x)
		batch_y = np.stack(batch_y)
		batch_zero_vel = np.stack(zero_velocity)

		if model is None:
			prediction = batch_zero_vel
		else:
			batch_x = batch_x[:,:max_len]

			# split in batches of 16
			n_batches = int(np.ceil(len(batch_x)/batch_size))
			preds = []

			for b in range(n_batches):
				batch = batch_x[b*batch_size:(b+1)*batch_size]
				a_mat = tf.repeat([dataset.adj], len(batch), axis=0)
				p_mat = tf.repeat([dataset.parent_mat], len(batch), axis=0)
				k_mat = tf.repeat([dataset.kinematic_mat], len(batch), axis=0)
				prediction = model((batch, a_mat, p_mat, k_mat), future_steps=future_steps, training=False)
				preds.append(prediction)

			prediction = np.concatenate(preds)
			batch_y = batch_y[:,:prediction.shape[1]]

		target_euler = tfe.from_quaternion(batch_y).numpy()
		prediction = tfe.from_quaternion(prediction).numpy()
		errors = metric_fn(prediction[:,:,1:], target_euler[:,:,1:])
		results[action] = np.mean(errors, axis=0).round(2)

	average_results = np.mean(list(results.values()), axis=0)
	results['average'] = average_results.round(2)

	return results

def plot_joint_trajectories(dataset, model, n_seconds=1.28, future_steps=0, figsize=None): # 1.28 secs = 32 frames
	plt.style.use('seaborn')

	d_labels = ['x', 'y', 'z', 'w']
	d_colors = ['g', 'r', 'b', 'k']
	actions  = ['walking', 'walkingdog', 'walkingtogether', 'greeting', 'discussion', 'waiting', 'smoking']

	# these actions are used for the ESANN publication due to the 6-page constraint
	#actions  = ['walking', 'walkingdog', 'discussion', 'smoking']

	joints = { 
		2  : 'Right Knee', 
		7  : 'Left Knee', 
		15 : 'Right Elbow', 
		29 : 'Thorax'
	}

	fps = dataset.fps
	fragment = [fragment_start, int(fragment_start + fps * n_seconds) + future_steps]

	figsize = (10.38,  3.65) if figsize is None else figsize
	fig, ax = plt.subplots(len(actions), len(joints), sharex=True, sharey=True, figsize=figsize)
	
	xs = np.linspace(0, n_seconds + frame2sec(future_steps), fragment[1] - fragment[0])

	for a, action in enumerate(actions):
		
		# load trajectory
		quats = dataset.dataset[subject][action][trial][fragment[0]:fragment[1]]

		if future_steps == 0:
			model_input = quats
		else:
			model_input = quats[:-future_steps]
		# predict trajectory
		pred = model((
			model_input[tf.newaxis], 
			dataset.adj[tf.newaxis],
			dataset.parent_mat[tf.newaxis],
			dataset.kinematic_mat[tf.newaxis]), 
			future_steps=future_steps, 
			training=False)[0]

		pred = tf.concat((model_input, pred), axis=0)
		
		for j, joint in enumerate(joints):

			dim = quats.shape[-1]
			# plot predicted
			for d in range(dim):
				seq = pred[:, joint, d]
				ax[a][j].plot(xs, seq, 
							  label=d_labels[d], 
							  #linewidth=1.5,
							  alpha=0.6,
							  color=d_colors[d])#colormap_pred(0.3+(dim+d)/(dim*4)))

			# plot ground truth
			for d in range(dim):
				seq = quats[:, joint, d]
				ax[a][j].plot(xs, seq, 
							  label='_nolegend_', 
							  linestyle='dotted', 
							  #linewidth=1.5,
							  color=d_colors[d])#colormap_gt(0.3+d/(dim*4)))
				
			ax[0][j].set_title(joints[joint])
			
			ax[-1][j].set_xlabel('time [sec]')

			if future_steps > 0:
				ax[a][j].axvline(x=xs[-future_steps], linestyle='--', linewidth=1., color='grey')


		# plot y labels
		ax[a][0].set_ylabel(action)

		# adjust y ticks
		ax[a][0].set_yticks(np.arange(-1, 1, 0.25))
		ax[a][0].relim()
		ax[a][0].autoscale_view()

	# plot legend on first plot
	legend = ax[0][0].legend(frameon = 1)
	frame = legend.get_frame()
	frame.set_color('white')

	plt.tight_layout()
	plt.subplots_adjust(
		top=0.936,
		bottom=0.146,
		left=0.074,
		right=0.981,
		hspace=0.068,
		wspace=0.043)

def plot_skeleton_trajectories(dataset, model, viz_steps=10, viz_every=3, 
	y_offset=300, y_init=-300, action='walking', n_seconds=1.28):
	y_init = y_init * viz_every
	fps = dataset.fps

	init_offset = np.array([[0., 0., y_init]])
	step_offset = np.array([[0., 0., y_offset]])

	fragment = [fragment_start, int(fragment_start + fps * n_seconds)]

	left_color = 'b'
	right_color = 'g'
	center_color = 'k'

	dir_colors = ['g', 'g', 'g', 'g', 'g', 'b', 'b', 'b', 'b', 'b', 
				  'b', 'k', 'k', 'g', 'g', 'g', 'g', 'g', 'g', 'b', 
				  'b', 'b', 'b', 'b', 'k', 'k', 'k', 'y', 'y', 'k', 'k', 'k']

	trace_joints = [3, 8, 14, 22]


	joint_colors = np.empty([32], dtype=np.unicode)
	joint_colors[dataset.left_joints] = left_color
	joint_colors[dataset.right_joints] = right_color
	lr_joints = np.concatenate((dataset.left_joints, dataset.right_joints))
	center_joints = np.setdiff1d(np.arange(32), lr_joints)
	joint_colors[center_joints] = center_color

	fig = plt.figure(figsize=(14, 6))
	fig.tight_layout()
	fig.subplots_adjust(top=1, bottom=0.03, left=0, right=1, wspace=0, hspace=0)
	plt.style.use('default')

	cmap = plt.cm.gnuplot

	ground_truth = dataset.dataset[subject][action][trial][fragment[0]:fragment[1]]

	prediction = model((
		ground_truth[tf.newaxis], 
		dataset.adj[tf.newaxis],
		dataset.parent_mat[tf.newaxis],
		dataset.kinematic_mat[tf.newaxis]), training=False)[0]

	plot_sequences = { 
		'ground truth' : ground_truth, 
		'prediction'   : prediction
	}

	for a, quats in enumerate(plot_sequences.values()):

		trajectory = K.forward_kinematics(
			dataset.parent_mat[tf.newaxis], 
			dataset.skeleton, 
			quats[tf.newaxis])[0]
		
		visualize_frames = trajectory[:viz_steps*viz_every:viz_every]
		
		ax = fig.add_subplot(len(plot_sequences), 1, a+1, projection='3d')
		
		max_y = 0
		min_y = 9e4
		
		for i, pos in enumerate(visualize_frames):		
			pos = pos + init_offset + i * step_offset * viz_every
			alpha = min(1, 0.2+(i+1)/(viz_steps+1))

			positions = np.transpose(pos)[[0,2,1]]
			ax.scatter(*positions, marker='.', alpha=0.7, c=joint_colors)

			parents = dataset.parent_mat @ pos[tf.newaxis]
			has_no_parent = 1-dataset.parent_mat.sum(-1, keepdims=True)
			parents = parents + pos[tf.newaxis] * has_no_parent
			directions = (pos - parents)[0]
			
			limb_start = np.transpose(parents[0])[[0,2,1]]
			limb_dir = np.transpose(directions)[[0,2,1]]
			
			ax.quiver(*limb_start, *limb_dir, arrow_length_ratio=0, alpha=0.7, color=dir_colors)
			max_y = np.maximum(max_y, pos.numpy().max())
			min_y = np.minimum(min_y, pos.numpy().min())
			#ax.axis('off')

			# add x-axis label
			label_interval = 1./(len(visualize_frames))
			fig.text(label_interval*0.7 + i * label_interval, 0.01, '%.2f' % frame2sec(i * viz_every), 
				fontsize=11, horizontalalignment='center', verticalalignment='center')

		ax.set_xlim(-1900, 1900)
		ax.set_ylim(
			min_y + 0.24*viz_steps*y_offset*viz_every, 
			max_y - 0.23*viz_steps*y_offset*viz_every
		)
		ax.set_zlim(-540, 540)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')

		# set camera perspective
		ax.view_init(elev=0., azim=25)
		# add x-axis describtion
		fig.text(0, 0.01, 'time [s]', fontsize=11, horizontalalignment='left', verticalalignment='center')
		
		for j, joint in enumerate(trace_joints):
			last_vis = i * viz_every
			j_traj = trajectory[:(viz_steps-1)*viz_every+1, joint].numpy()
			j_traj = j_traj + np.arange(len(j_traj))[:,np.newaxis] * step_offset + init_offset
			ax.plot(*j_traj.T[[0,2,1]], linestyle='dotted', color=joint_colors[joint])
			
	# add describtion
	interval = 1/viz_steps
	fraction = 1./len(plot_sequences)

	for i, s in enumerate(plot_sequences):
		fig.text(0.02, 1 - fraction/2 - i * fraction, s, fontsize=11, rotation=90, 
					   horizontalalignment='center', verticalalignment='center')

# ----------------------------------------------------------------

def dict_to_latex_table_lines(d, output='table.tex', actions_per_line=4):
	line_format = (' & %.2f'*len(eval_ms))
	eval_frames = np.array([ ms2frame(ms) for ms in eval_ms ])
	out = ''

	with open(output, 'w') as file:
		for l, (action, values) in enumerate(d.items()):
			out = out + line_format % tuple(values[eval_frames])
			if (l+1) % actions_per_line == 0:
				out = out + ' \\\ \n'
				file.write(out)
				out = ''
			else:
				out = out + ' &'
		file.write(out)

	return out
