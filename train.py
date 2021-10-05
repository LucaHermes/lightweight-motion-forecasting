import os
from datetime import datetime

from models.forecasting_model import ForecastingModel
from models import encoders
from data import h36m, h36m_utils
import tensorflow as tf


def train(batch_size, epochs, condition_length=32, forecasting_length=10, from_velocities=True, 
	      data_dir=None, output_dir=None, init_learning_rate=None, checkpoint=None, **config):
	N_PARALLEL = tf.data.experimental.AUTOTUNE
	tb_logs_dir  = os.path.join(output_dir, 'train_logs')
	ckpt_dir  = os.path.join(output_dir, 'checkpoints')
	figures_out  = os.path.join(output_dir, 'figures')
	numerics_out = os.path.join(output_dir, 'numeric_results')

	name = '%s_forecasting_%s' % (config['block_type'], 'vel' if from_velocities else 'abs')
	current_time = datetime.now().strftime("%Y%m%d-%H%M%S")


	def prepare_forecast(quats, subject, action, adj, parent_mat, kin_mat):
	    condition = quats[:condition_length]
	    target = quats[condition_length:][:forecasting_length]
	    return (condition, adj, parent_mat, kin_mat), target

	# get the dataset
	motion_data = h36m.H36MDataset(data_dir)
	motion_data = motion_data.downsample(2)
	#motion_data = motion_data.mirror()
	motion_data = motion_data.reorder_to_rlc()
	motion_data = motion_data.mirror()

	time_horizon = condition_length + forecasting_length
	train = h36m_utils.as_tf_dataset(motion_data, 'training', time_horizon, sampling=True)
	val = h36m_utils.as_tf_dataset(motion_data, 'validation', time_horizon, sampling=True)
	#test = h36m_utils.as_tf_dataset(motion_data, 'testing',   time_horizon, sampling=True)

	train_fc = train.map(prepare_forecast, num_parallel_calls=N_PARALLEL)
	val_fc = val.map(prepare_forecast, num_parallel_calls=N_PARALLEL)

	train_fc = train_fc.shuffle(100)
	train_fc = train_fc.batch(batch_size, drop_remainder=True)
	train_fc = train_fc.prefetch(N_PARALLEL)
	val_fc = val_fc.shuffle(100)
	val_fc = val_fc.batch(batch_size, drop_remainder=True)
	val_fc = val_fc.prefetch(N_PARALLEL)

	forecasting_module = encoders.GraphWavenet(**config)
	model = ForecastingModel(forecasting_module, motion_data.skeleton, predict_velocities=config['vel'])

	learning_rate = lambda: init_learning_rate * 0.999**tf.cast(model.training_epoch, tf.float32)
	optimizer = tf.keras.optimizers.Adam(learning_rate)

	if checkpoint is not None:
		model.load(checkpoint, optimizer=optimizer)
	
	model.train(train_fc, optimizer, epochs, val_fc, 
	            tb_logs_dir + '/' + name + '/' + current_time,
	            future_steps=forecasting_length,
	            ckpt_name=name + '/' + current_time,
	            max_grad_norm=.5)