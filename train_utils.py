import tensorflow_graphics.geometry.transformation.euler as tfe
import tensorflow as tf
import os

from datetime import datetime
from models import encoders


def build_encoder(config):

	name = config['model_name']

	if name == 'LayeredGatedGCN':
		encoder = encoders.LayeredGatedGCN(gnn, **config)
	elif name == 'LayeredGCNGated':
		encoder = encoders.LayeredGCNGated(gnn, **config)
	elif name == 'GatedGCN':
		cell = encoders.GatedGCNCell(gnn, **config)
		encoder = encoders.RecurrentGCN(cell)
	elif name == 'GraphWavenet':
		encoder = encoders.GraphWavenet(**config)
	elif name == 'Transformer':
		encoder = encoders.AREncoder(**config)
	else:
		raise NotImplementedError('A model with name ' +
			name + ' is not implemented.')
	return encoder

def dict_log(writer, topic, values_dict, step):
	# do tensorboard logs
	with writer.as_default():
		for k, v in values_dict.items():
			tf.summary.scalar(topic + '/' + str(k), v, step=step)

def get_tb_writer(log_dir, name, writer_name):
	current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
	#current_time = '20201022-153447'
	train_log_dir = os.path.join(log_dir, name, current_time, writer_name)
	return tf.summary.create_file_writer(train_log_dir)

def get_forecasting_metrics(y_true, y_pred, loss):
	mae = tf.keras.metrics.MAE(y_true, y_pred)
	mae = tf.reduce_mean(mae)
	return {
		'mae' : mae,
		'loss' : loss
	}

def get_rotational_metrics(y_true, y_pred, zero_vel_pred):
	true_euler = tfe.from_quaternion(y_true)
	pred_euler = tfe.from_quaternion(y_pred)
	zero_vel_euler = tfe.from_quaternion(zero_vel_pred)

	difference = pred_euler - true_euler

	abs_diff = tf.abs(difference)
	square_diff = tf.square(difference)

	mae = tf.reduce_mean(abs_diff)
	mse = tf.reduce_mean(square_diff)

	joint_mae = tf.reduce_sum(abs_diff, axis=-1)
	joint_mean_mae = tf.reduce_mean(joint_mae)

	skeleton_mae = tf.reduce_sum(joint_mae, axis=-1)
	skeleton_mean_mae = tf.reduce_mean(skeleton_mae)

	joint_mse = tf.reduce_sum(square_diff, axis=-1)
	joint_mean_mse = tf.reduce_mean(joint_mse)

	skeleton_mse = tf.reduce_sum(joint_mse, axis=-1)
	skeleton_mean_mse = tf.reduce_mean(skeleton_mse)

	joint_rsse = tf.reduce_sum(square_diff, axis=(-1, -2))
	joint_rsse = tf.sqrt(joint_rsse)
	joint_rsse = tf.reduce_mean(joint_rsse)

	zero_vel_mae = tf.square(true_euler - zero_vel_euler)
	zero_vel_joint_mae = tf.reduce_sum(zero_vel_mae, axis=(-1, -2))
	zero_vel_joint_mae = tf.sqrt(zero_vel_joint_mae)
	zero_vel_joint_mae = tf.reduce_mean(zero_vel_joint_mae)

	return {
		'mae' : mae,
		'joint_rsse' : joint_rsse,
		'joint_mae' : joint_mean_mae,
		'skeleton_mae' : skeleton_mean_mae,
		'mse' : mse,
		'joint_mse' : joint_mean_mse,
		'skeleton_mse' : skeleton_mean_mse,
		'0-velocity_joint_rsse' : zero_vel_joint_mae
	}