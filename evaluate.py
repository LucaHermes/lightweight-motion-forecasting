from models.forecasting_model import ForecastingModel
from models import encoders
from data import h36m, h36m_utils
import evaluation.evaluation as E

from datetime import datetime
import os
import pprint
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

def evaluate(batch_size, condition_length=32, forecasting_length=10, from_velocities=True, eval_protocol=None,
	         data_dir=None, output_dir=None, init_learning_rate=None, checkpoint=None, **config):
	tb_logs_dir  = os.path.join(output_dir, 'train_logs')
	ckpt_dir  = os.path.join(output_dir, 'checkpoints')
	figures_out  = os.path.join(output_dir, 'figures')
	numerics_out = os.path.join(output_dir, 'numeric_results')

	name = '%s_forecasting_%s' % (config['block_type'], 'vel' if from_velocities else 'abs')
	current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

	tex_table_file = os.path.join(numerics_out, name + '_table_%s.tex' % eval_protocol)
	results_dict_file = os.path.join(numerics_out, name + '_all_quantitative_results.txt')

	# get the dataset
	print('Loading H3.6M dataset ...')
	motion_data = h36m.H36MDataset(data_dir)
	print(' * Downsampling H3.6M ...')
	motion_data = motion_data.downsample(2)
	motion_data = motion_data.reorder_to_rlc()

	print('Building the model ...')
	forecasting_module = encoders.GraphWavenet(**config)
	model = ForecastingModel(forecasting_module, motion_data.skeleton, predict_velocities=config['vel'])

	if checkpoint is not None:
		model.load(checkpoint)

	# ---------------- Quantitative evaluation ----------------
	quantitative_res = E.run_quantitative_eval(motion_data, model=model, max_len=32, 
		protocol=eval_protocol, batch_size=batch_size)
	if not os.path.isdir(numerics_out):
		os.makedirs(numerics_out)
	_ = E.dict_to_latex_table_lines(quantitative_res, output=tex_table_file)
	with open(results_dict_file, 'w') as f:
		pp = pprint.PrettyPrinter(indent=4)
		config_str = pp.pformat(quantitative_res)
		f.write(config_str)

	# ---------------- Qualitative evaluation ----------------
	E.plot_joint_trajectories(motion_data, model=model, future_steps=32, figsize=(28, 20))
	plt.show()
	#print('Plotting skeletal trajectory ...')
	#E.plot_skeleton_trajectories(motion_data, model=model)
	#plt.show()