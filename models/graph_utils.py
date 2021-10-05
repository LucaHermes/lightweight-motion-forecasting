import tensorflow as tf
import numpy as np

def naive_normalize(mat, is_transposed=False):
	# if the matrix will be transposed in gcn, normalize the second last dimension
	num_inputs = tf.reduce_sum(mat, -1 - is_transposed, keepdims=True)
	return mat / (num_inputs + 1e-9)

def normalize(mat, is_transposed=False):
	# if the matrix will be transposed in gcn, normalize the second last dimension
	# check if the matrix is symmetric
	is_symmetric = tf.reduce_all(tf.equal(mat, tf.linalg.matrix_transpose(mat)))
	if not is_symmetric:
		return naive_normalize(mat, is_transposed=is_transposed)
	num_inputs = tf.reduce_sum(mat, axis=-1 - is_transposed)
	num_inputs_norm = (num_inputs + 1e-8)**-0.5
	d_norm = tf.linalg.diag(num_inputs_norm)
	return d_norm @ mat @ d_norm

def node2edge(x, rec, send):
	'''
	Message passing from the nodes to the edges undirected.
	'''
	rec = normalize(rec)
	send = normalize(send)
	receivers = tf.matmul(rec, x)
	senders = tf.matmul(send, x)
	return tf.concat((senders, receivers), -1)

def edge2node(x, rec):
	'''
	Message passing from the edges to the nodes undirected.
	'''
	rec_mat = tf.transpose(rec, [0, 2, 1])
	rec_mat = normalize(rec_mat)
	receive = tf.matmul(rec_mat, x)
	return receive

def node2node(x, adj, transpose_mat=False):
	'''
	Message passing from the edges to the nodes undirected.
	'''
	adj = normalize(adj, is_transposed=transpose_mat)
	msg = tf.matmul(adj, x, transpose_a=transpose_mat)
	return msg

def remove_diag(tensor):
	'''
	Removes the diagonal from a batch of n x n matrices.
	'''
	# tensor has shape [batch, n, n]
	n = tf.shape(tensor)[1]
	mask = tf.linalg.diag(tf.zeros(n), padding_value=1)
	return tf.boolean_mask(tensor, mask, axis=1)

def self_stack_graph(adj, direction=None):
	'''
	Stack nodes of two graphs by concatenating them along the node axis: concat(nodesA, nodesB)
	Apply GCN with the output matrix to fuse the graphs.
	The resulting matrix will have the original edges of the graphs together with
	edges connecting the nodes of graph A with the nodes of graph B.
	This is an identity mapping, so the graphs must have an equal number of nodes.

	-> nodesA, nodesB, adj
	AB_mat = self_stack_graph(adj)
	AB_nodes = concat(nodesA, nodesB)

	direction: 
		None        - connections: graph A <-> graph B (default)
		'bottom_up' - connections: graph A  -> graph B
		'top_down'  - connections: graph B  -> graph A
	'''
	n_nodes = adj.shape[-1]

	identity = tf.ones_like(adj) * tf.eye(n_nodes)

	if direction is None:
		identityA = identity
		identityB = tf.identity(identity)
	elif direction == 'bottom_up':
		identityA = tf.zeros_like(identity)
		identityB = identity
	elif direction == 'top_down':
		identityA = identity
		identityB = tf.zeros_like(identity)

	td_mat = tf.concat((adj, identityA), axis=-1)
	bu_mat = tf.concat((identityB, adj), axis=-1)

	return tf.concat((td_mat, bu_mat), axis=-2)

#def get_msg_matrices(n_joints, adj=None):
	'''
	Calculates the message passing matrices for n_joints and
	the given adjacency matrix. If none is given, a fully 
	connected graph is assumed.
	'''
#	if adj is None:
#		adj = tf.ones([n_joints, n_joints]) - tf.eye(n_joints)
	
	#batch_size = adj.shape[0]
	# receivers of messages
	#adj = tf.cast(adj, tf.float32)
	#receivers = tf.one_hot(tf.transpose(tf.where(adj > .5))[1], n_joints)
	#senders = tf.one_hot(tf.transpose(tf.where(adj > .5))[2], n_joints)
	#receivers = tf.stack(tf.split(receivers, batch_size))
	#senders = tf.stack(tf.split(senders, batch_size))
	
	#batch_size = adj.shape[0]
	# receivers of messages
#	adj = tf.cast(adj, tf.float32)
#	receivers = tf.one_hot(tf.transpose(tf.where(adj > .5))[0], n_joints)
#	senders = tf.one_hot(tf.transpose(tf.where(adj > .5))[1], n_joints)
#
#	return adj, receivers, senders

def print_matrix(matrix):
	mat_shape = np.shape(matrix)

	print('  \\ ', end='')
	for i in range(mat_shape[-1]):
		print('%3d' % i, end='')
	print()
	print('  |  ' + '---'*mat_shape[-1])

	for i in range(mat_shape[-2]):
		print('%2d|' % i, end=' ')
		for j in range(mat_shape[-1]):
			print('%3d' % int(matrix[...,i,j]), end='')
		print()

def get_msg_matrices(n_joints, adj=None, selfloops=False):
	'''
	Calculates the message passing matrices for n_joints and
	the given adjacency matrix. If none is given, a fully 
	connected graph is assumed.
	'''
	if adj is None:
		adj = tf.ones([1, n_joints, n_joints])

	batch = adj.shape[0]
	
	if not selfloops:
		rec_joints = n_joints - 1
		# remove the diagonal as self loops are not included
		adjmod = tf.reshape(remove_diag(adj), [batch, n_joints, rec_joints])
	else:
		rec_joints = n_joints
		adjmod = adj + tf.eye(n_joints)

	adjmod = tf.clip_by_value(adjmod, 0., 1.)
	
	shape_n2e = [batch, n_joints, n_joints * rec_joints]
	shape_e2n = [batch, n_joints * rec_joints, n_joints]


	# create message passing matrices from nodes to edges
	idx = tf.where(adjmod > 0.5)
	scale = idx[:,1] * rec_joints
	idx = idx + tf.pad(scale[:,tf.newaxis], [[0,0],[2,0]])
	vals = tf.ones(idx.shape[0])
	sparse_n2e = tf.sparse.SparseTensor(idx, vals, shape_n2e)
	n2e = tf.sparse.to_dense(sparse_n2e)

	# create message passing matrices from edges to nodes
	node_ids = tf.range(n_joints, dtype=tf.int64)[tf.newaxis]
	receives = tf.tile(node_ids, [n_joints, 1])
	
	if not selfloops:
		receives = remove_diag([receives])[0]
	else:
		receives = tf.reshape(receives, [-1])

	receives = tf.gather(receives, tf.where(n2e)[:,-1:])
	senders = tf.where(n2e)[:,::2]
	idx = tf.concat((senders, receives), -1)
	sparse_e2n = tf.sparse.SparseTensor(idx, vals, shape_e2n)
	e2n = tf.sparse.to_dense(sparse_e2n)

	n2e = tf.transpose(n2e, [0,2,1])

	return adj, n2e, e2n