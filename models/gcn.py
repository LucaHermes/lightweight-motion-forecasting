import tensorflow as tf
import tensorflow.keras.layers as layers

from models import model_utils as M
from models import graph_utils as G


class KinematicGCN(layers.Layer):
	
	def __init__(self, hidden, depth, layer_type='n2n', activation=None, add_identity=True, use_bias=False):
		'''
		Applies different graph convolutions to inner joints (closer to root) and outer joints.
		If add_identity is True, a 'graph convolution' with the identity matrix
		will be added, i.e. MLP(x).
		'''
		super(KinematicGCN, self).__init__()
		self.forward_gcn = GraphConv(hidden, depth, layer_type, 
			activation=activation, use_bias=use_bias)
		self.backward_gcn = GraphConv(hidden, depth, layer_type, 
			activation=activation, use_bias=use_bias)

		if add_identity:
			hidden = [hidden] * depth
			self.identity_gcn = M.mlp(hidden, activation=activation, use_bias=use_bias)
		else:
			self.identity_gcn = tf.zeros_like

		self.add_identity = add_identity

	def _transpose(self, mat):
		return tf.transpose(mat, [0, 2, 1])
	
	def call(self, x, mats):
		'''
		Returns the sum of the graph convolutions:
		MLP(mats @ x) + MLP(mats^T @ x)
		If add_identity is True, a 'graph convolution' with the identity matrix
		will be added, i.e. MLP(x).
		'''
		output = self.identity_gcn(x)
		output += self.forward_gcn(x, mats, transpose_mat=False)
		output += self.backward_gcn(x, mats, transpose_mat=True)
		
		return output

class GraphConv(layers.Layer):

	def __init__(self, hidden, depth, layer_type='n2n', att_heads=0, activation='elu', use_bias=True):
		super(GraphConv, self).__init__()
		# mlp gets applied after msg passing, except for n2n_diffusion
		if tf.constant(hidden).ndim == 0:
			hidden = [hidden] * depth

		self.mlp = M.mlp(hidden, activation=activation, use_bias=use_bias)
		# set the correct msg passing operation
		if layer_type == 'n2n':
			self.msg_passing = lambda x, m, t: G.node2node(x, m[0], transpose_mat=t)
		elif layer_type == 'n2n_diffusion':
			self.msg_passing = self._n2n_diffusion
			act = [activation] * (depth-1) + [None]
			self.layers = [ layers.Dense(h, a) for h, a in zip(hidden, act) ]
			self.mlp = tf.identity

	def _n2n_diffusion(self, x, mat, transpose_mat=False):
		a_pow = mat[0]

		z = tf.zeros_like(x)

		for layer in self.layers:
			z_diff = G.node2node(x, a_pow, transpose_mat=transpose_mat)
			z += layer(z_diff)
			a_pow = tf.minimum(a_pow @ a_pow, 1.)

		return z

	def call(self, x, mats, transpose_mat=False):
		# mats is a tuple: (adj, rec, snd)
		graph_stream = self.msg_passing(x, mats, transpose_mat)
		return self.mlp(graph_stream)
