import tensorflow as tf
import tensorflow.keras.layers as layers

from models import graph_utils as G
from models import gcn as graph_conv

class TEConvolution(layers.Layer):

	def __init__(self, filters, kernel_size, dilation_rate=1, activation=None):
		super(TEConvolution, self).__init__()
		self.filters = filters
		self.kernel_size = kernel_size
		self.dilation_rate = dilation_rate
		self.conv = layers.Conv2D(
			filters, 
			(3, kernel_size), 
			dilation_rate=(1, dilation_rate), 
			activation=activation)

	def build(self, input_shape):
		time, n_joints, features = input_shape[-3:]
		x_ndims = input_shape.rank
		perm = tf.range(x_ndims + 1)
		# create permutation order for the temporal convolution
		self.perm1 = [*perm[:-4], *perm[-3:-1], perm[-4], perm[-1]]
		self.perm2 = [*perm[:-4], perm[-2], perm[-4], perm[-3], perm[-1]]
		# create 'causal'-padding for the temporal axis
		temp_padding = (self.kernel_size-1) * self.dilation_rate
		self.paddings = [[0, 0]] * (x_ndims + 1 - 2) + [[temp_padding, 0]] + [[0, 0]]

	def call(self, x, parent_mat):
		'''
		Expected shape x:          [..., time, joints, channels]
		Expected shape parent_mat: [..., 1, joints, joints]
		'''
		parents = parent_mat[0] @ x
		grandparents = parent_mat[0] @ parents
		# stack to shape: [..., time, joints, 3, channels]
		x_ext = tf.stack((x, parents, grandparents), axis=-2)
		# batch, joints, 3, time, channels
		# transpose to shape: [..., joints, 3, time, channels]
		x_ext = tf.transpose(x_ext, self.perm1)
		# this is not a causal convolution, apply causal padding
		x_ext = tf.pad(x_ext, self.paddings)

		result = self.conv(x_ext)

		# throw away the extended dimension
		# transpose to shape: [..., time, joints, 1, channels]
		result = tf.transpose(result, self.perm2)
		result = tf.squeeze(result, axis=-2)
		return result

class EncoderBase(layers.Layer):

	def __init__(self, n_blocks, **kwargs):
		super(EncoderBase, self).__init__()
		self.n_blocks = n_blocks
		self.kwargs = kwargs
		self.stats = {}

	def call(self, x, adj, parent_mat, kin_mat, training=True):
		batch_size, time, n_joints, n_featrues = x.shape
		skip = None
		x = self.embedding(x, training=training)

		for block in self.blocks:
			x, skip = block(x, adj, parent_mat, kin_mat, skip, training=training)

		return self.out_model(skip, training=training)

class GraphWavenet(EncoderBase):

	def __init__(self, block_type, n_blocks, skip_channels, filters, kernel_sizes, dilations, dropout_rate, **kwargs):
		super(GraphWavenet, self).__init__(n_blocks, **kwargs)
		self.blocks = []
		self.n_blocks = n_blocks
		self.skip_channels = skip_channels
		self.filters = filters
		self.kernel_sizes = kernel_sizes
		self.dilations = dilations
		self.kwargs = kwargs
		self.dropout_rate = dropout_rate
		self.block_type = block_type

	def build(self, input_shape):
		self.embedding = tf.keras.Sequential([
			layers.Dense(self.filters[0]),
			layers.Dropout(self.dropout_rate)
		])

		if self.block_type == 'graph_wavenet':
			BlockCLS = GraphWavenetBlock
		elif self.block_type == 'graph_wavenet_original':
			BlockCLS = GraphWavenetBlockOrig
		elif self.block_type == 'graph_wavenet_original_te':
			BlockCLS = GraphWavenetBlockOrigTE
		else:
			raise NotImplementedError(f'A block with type {self.blockType}'
				'is not implemented.')

		for b in range(self.n_blocks):
			gw_block = BlockCLS(
				self.skip_channels,
				self.kernel_sizes[b],
				self.filters[b], 
				self.dilations[b],
				b == (self.n_blocks - 1),
				n_clusters=max(8 - 2 * b, 2),
				dropout_rate=self.dropout_rate,
				#alpha=5./2**b,
				**self.kwargs)

			self.blocks.append(gw_block)

		out_channels = self.kwargs.get('out_channels', self.skip_channels)

		#self.out_model = tf.keras.Sequential([
		#	layers.Dropout(self.dropout_rate),
		#	layers.ReLU(),
		#	layers.Dense(self.skip_channels, activation=None),
		#	layers.Dropout(self.dropout_rate),
		#	layers.ReLU(),
		#	layers.Dense(out_channels, activation=None)
		#])
		self.out_model = tf.keras.Sequential([
			layers.ReLU(),
			layers.Dense(self.skip_channels, activation=None),
			layers.ReLU(),
			layers.Dropout(self.dropout_rate),
			layers.Dense(out_channels, activation=None),
			layers.Dropout(self.dropout_rate)])
		
class GraphWavenetBlock(layers.Layer):

	def __init__(self, skip_channels, kernel_size, filters, dilation, is_last, mlp_depth, dropout_rate, selfloops, **kwargs):
		super(GraphWavenetBlock, self).__init__()
		self.skip_channels = skip_channels
		self.kernel_size = kernel_size
		self.dilation = dilation
		self.filters = filters
		self.mlp_depth = mlp_depth
		self.selfloops = selfloops
		self.is_last = is_last
		self.dropout_rate = dropout_rate
		self.kwargs = kwargs

	def build(self, input_shape):
		batch, time, joints, in_features = input_shape

		self.causal_conv = TEConvolution(
			self.filters*2, 
			self.kernel_size, 
			dilation_rate=self.dilation)

		self.skip_conv = layers.Conv1D(
			self.skip_channels, 
			kernel_size=1, 
			activation=None)

		self.gcn = graph_conv.KinematicGCN(
			self.filters, 
			depth=1, 
			layer_type='n2n',
			activation=None,
			add_identity=True, 
			use_bias=False)

		self.dropout = layers.Dropout(self.dropout_rate)

		if 'single_temp_out' not in self.kwargs:
			self.steps_return = -1
		elif self.kwargs['single_temp_out']:
			self.steps_return = -1
		else:
			self.steps_return = None

		if self.is_last:
			self.gcn = lambda x, t: tf.identity(x)

		if self.filters != in_features:
			self.res_conv = layers.Conv1D(self.filters, 1, activation=None)
		else:
			self.res_conv = tf.identity

	def call(self, x, adj, parent_mat, kin_mat, x_skip=None, training=True):
		batch_size, time, n_joints, n_featrues = x.shape
		res = x

		x = self.causal_conv(x, parent_mat)

		# gated linear activation
		act, gate = tf.split(x, 2, -1)
		activation = tf.nn.tanh(act)
		gate = tf.nn.sigmoid(gate)
		x = activation * gate

		if x_skip is None:
			x_skip = x[:, self.steps_return:]
			x_skip = self.skip_conv(x_skip)
		else:
			x_skip += self.skip_conv(x[:, self.steps_return:])

		x = self.gcn(x, parent_mat)
		x = self.dropout(x, training=training)
		x += self.res_conv(res)

		return x, x_skip

class GraphWavenetBlockOrig(layers.Layer):

	def __init__(self, skip_channels, kernel_size, filters, dilation, is_last, mlp_depth, dropout_rate, selfloops, **kwargs):
		super(GraphWavenetBlockOrig, self).__init__()
		self.skip_channels = skip_channels
		self.kernel_size = kernel_size
		self.dilation = dilation
		self.filters = filters
		self.mlp_depth = mlp_depth
		self.selfloops = selfloops
		self.is_last = is_last
		self.dropout_rate = dropout_rate
		self.kwargs = kwargs

	def build(self, input_shape):
		batch, time, joints, in_features = input_shape

		self.causal_conv = layers.Conv1D(
			self.filters*2, 
			self.kernel_size, 
			dilation_rate=self.dilation, 
			padding='causal')

		self.skip_conv = layers.Conv1D(
			self.skip_channels, 
			kernel_size=1, 
			activation=None)
		
		self.gcn = graph_conv.GraphConv(
			self.filters, 
			depth=2, 
			layer_type='n2n_diffusion', 
			activation=None)

		if 'single_temp_out' not in self.kwargs:
			self.steps_return = -1
		elif self.kwargs['single_temp_out']:
			self.steps_return = -1
		else:
			self.steps_return = None

		if self.is_last:
			self.gcn = lambda x, t: tf.identity(x)

		if self.filters != in_features:
			self.res_conv = layers.Conv1D(self.filters, 1, activation=None)
		else:
			self.res_conv = tf.identity

	def call(self, x, adj, parent_mat, kin_mat, x_skip=None, training=True):
		batch_size, time, n_joints, n_featrues = x.shape
		res = x

		x = self.causal_conv(x)
		x = gated_activation(x)

		if x_skip is None:
			x_skip = x[:, self.steps_return:]
			x_skip = self.skip_conv(x_skip)
		else:
			x_skip += self.skip_conv(x[:, self.steps_return:])

		x = self.gcn(x, adj)
		x += self.res_conv(res)

		return x, x_skip

class GraphWavenetBlockOrigTE(layers.Layer):

	def __init__(self, skip_channels, kernel_size, filters, dilation, is_last, mlp_depth, dropout_rate, selfloops, **kwargs):
		super(GraphWavenetBlockOrigTE, self).__init__()
		self.skip_channels = skip_channels
		self.kernel_size = kernel_size
		self.dilation = dilation
		self.filters = filters
		self.mlp_depth = mlp_depth
		self.selfloops = selfloops
		self.is_last = is_last
		self.dropout_rate = dropout_rate
		self.kwargs = kwargs

	def build(self, input_shape):
		batch, time, joints, in_features = input_shape

		self.causal_conv = TEConvolution(
			self.filters*2, 
			self.kernel_size, 
			dilation_rate=self.dilation)

		self.skip_conv = layers.Conv1D(
			self.skip_channels, 1, 
			activation=None, 
			input_shape=[time, self.filters])
		
		self.gcn = graph_conv.GraphConv(
			self.filters, 
			depth=2, 
			layer_type='n2n_diffusion', 
			activation=None)

		if 'single_temp_out' not in self.kwargs:
			self.steps_return = -1
		elif self.kwargs['single_temp_out']:
			self.steps_return = -1
		else:
			self.steps_return = None

		if self.is_last:
			self.gcn = lambda x, t: tf.identity(x)

		if self.filters != in_features:
			self.res_conv = layers.Conv1D(self.filters, 1, activation=None)
		else:
			self.res_conv = tf.identity

	def call(self, x, adj, parent_mat, kin_mat, x_skip=None, training=True):
		batch_size, time, n_joints, n_featrues = x.shape
		res = x

		x = self.causal_conv(x, parent_mat)
		x = gated_activation(x)

		if x_skip is None:
			x_skip = x[:, self.steps_return:]
			x_skip = self.skip_conv(x_skip)
		else:
			x_skip += self.skip_conv(x[:, self.steps_return:])
		
		x = self.gcn(x, adj)
		x += self.res_conv(res)

		return x, x_skip

def gated_activation(x):
	act, gate = tf.split(x, 2, -1)
	gate = tf.nn.sigmoid(gate)
	activation = tf.nn.tanh(act)
	return activation * gate