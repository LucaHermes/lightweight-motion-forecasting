import tensorflow as tf
import tensorflow_graphics.geometry.transformation.quaternion as tfq
import tensorflow_graphics.geometry.transformation.euler as tfe
import numpy as np

# ---------- WORKAROUND to get tfg support for tf2 graph execution -----------
# https://github.com/tensorflow/graphics/issues/15#issuecomment-533034402
import sys

module = sys.modules['tensorflow_graphics.util.shape']
def _get_dim(tensor, axis):
    """Returns dimensionality of a tensor for a given axis."""
    return tf.compat.v1.dimension_value(tensor.shape[axis])
module._get_dim = _get_dim

sys.modules['tensorflow_graphics.util.shape'] = module

# -----------------------------------------------------------------------------


GLOBAL_REF = tf.constant([1.,0.,0.])
NULL_QUATERNION = tf.constant([0., 0., 0., 1.])
PI = tf.constant(np.pi)

def expmap_to_quaternion(x):
    # convert to quaternions
    angle = tf.linalg.norm(x, axis=-1, keepdims=True)
    quaternion = from_axis_angle(x, angle)
    return quaternion

def expmap_to_quaternion_np(x):
    # convert to quaternions
    angle = np.linalg.norm(x, axis=-1, keepdims=True)
    #axis = x / (angle + eps)
    half_angle = 0.5 * angle

    w = np.cos(half_angle)
    xyz = 0.5 * np.sinc(half_angle/np.pi) * x
    quat = np.concatenate((xyz, w), axis=-1)

    return quat

def from_axis_angle(axis, angle):
    axis = tf.nn.l2_normalize(axis, axis=-1)
    half_angle = 0.5 * angle
    w = tf.cos(half_angle)
    xyz = tf.sin(half_angle) * axis
    quat = tf.concat((xyz, w), axis=-1)
    return tf.nn.l2_normalize(quat, axis=-1)

def quat_to_euler(quaternions, eps=0.):
    shape = tf.shape(quaternions)
    quats = tf.reshape(quaternions, [-1, 4])
    euler = tfe.from_quaternion(quats)
    euler = tf.reshape(euler, tf.concat((shape[:-1], [3]), -1))
    return euler

def euler_to_quat(euler):
    return tfq.from_euler(euler)

def euler_to_quat_II(euler):
    shape = tf.shape(euler)
    euler = tf.reshape(euler, [-1, 3])
    quats = tfq.from_euler(euler)
    quats = tf.reshape(euler, tf.concat((shape[:-1], [4]), -1))
    return quats

def select_hemisphere(quaternions):
    # Selects the closest quaternion between consecutive timeframes.
    # Inputs: [batch, time, joints, 4]
    quats = tf.identity(quaternions)
    prod = quats[:,:-1] * quats[:,1:]
    # get the dot product to determine relative orientations
    dot_products = tf.reduce_sum(prod, axis=-1, keepdims=True)
    # negative value means the vectors of two timeframes have an
    # angle > 90 [deg] and therefore the hemisphere needs to be switched.
    switch_ids = tf.where(tf.less(dot_products, 0), 1., 0.)
    switch_ids = tf.math.mod(tf.cumsum(switch_ids, axis=1), 2)

    switch_ids = tf.pad(switch_ids == 1, [[0,0], [1,0], [0,0], [0,0]])
    quats = tf.where(switch_ids, quats * -1, quats)
    return quats

def accumulate_temporal_quaternions(quaternions):
    # quaternions: [batch, time, ACC_STEPS, joints, features]
    # the quaternions are multiplied along axis acc_steps.
    quaternions = tf.transpose(quaternions, [2,0,1,3,4])
    return tf.transpose(tf.scan(tfq.multiply, quaternions), [1,2,0,3,4])

def get_difference_quaternions(q1, q2):
    return quaternion_multiply(q2, conjugate_quaternion(q1))

def quaternion_mae_wrap(true_quaternions, pred_quaternions):
    #return tf.reduce_mean(tfq.relative_angle(q1, q2))
    true_euler = quat_to_euler(true_quaternions)
    pred_euler = quat_to_euler(pred_quaternions)
    error = tf.math.mod(pred_euler - true_euler + PI, 2 * PI) - PI
    return tf.reduce_mean(tf.abs(error))

LEFT_JOINTS =  [1, 2, 3, 4, 5,  24, 25, 26, 27, 28, 29, 30, 31]
RIGHT_JOINTS = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]

left_joint_hot = tf.one_hot(LEFT_JOINTS, depth=32)
right_joint_hot = tf.one_hot(RIGHT_JOINTS, depth=32)
left_right_hot = tf.reduce_sum(left_joint_hot + right_joint_hot, axis=-2)

mirror_mat = tf.matmul(left_joint_hot, right_joint_hot, transpose_a=True)
mirror_mat = mirror_mat + tf.linalg.matrix_transpose(mirror_mat)
# add selfloops for joints that are in the center, i.e. not a left/right joint
mirror_mat = mirror_mat + left_right_hot

def quaternion_mae_wrap_mirror_inv(true_quaternions, pred_quaternions):
    #return tf.reduce_mean(tfq.relative_angle(q1, q2))

    # original loss
    true_euler = quat_to_euler(true_quaternions)
    pred_euler = quat_to_euler(pred_quaternions)
    error = tf.math.mod(pred_euler - true_euler + PI, 2 * PI) - PI
    l1 = tf.reduce_mean(tf.abs(error))

    # mirrored loss:
    true_quaternions_mirrored = mirror_mat @ true_quaternions
    pred_quaternions_mirrored = mirror_mat @ pred_quaternions

    true_euler = quat_to_euler(true_quaternions_mirrored)
    pred_euler = quat_to_euler(pred_quaternions_mirrored)
    error = tf.math.mod(pred_euler - true_euler + PI, 2 * PI) - PI
    l2 = tf.reduce_mean(tf.abs(error))

    # take the minimum of both versions
    loss = tf.minimum(l1, l2)
    return loss

def quaternion_mse(true_quaternions, pred_quaternions):
    #return tf.reduce_mean(tfq.relative_angle(q1, q2))
    true_euler = quat_to_euler(true_quaternions)
    pred_euler = quat_to_euler(pred_quaternions)
    return tf.reduce_mean(tf.square(pred_euler - true_euler))

def quaternion_mse_wrap(true_quaternions, pred_quaternions):
    #return tf.reduce_mean(tfq.relative_angle(q1, q2))
    true_euler = quat_to_euler(true_quaternions)
    pred_euler = quat_to_euler(pred_quaternions)
    error = tf.math.mod(pred_euler - true_euler + PI, 2 * PI) - PI
    return tf.reduce_mean(tf.square(error))

def quaternion_skeleton_rsse(true_quaternions, pred_quaternions):
    # shape: [batch, time, joints, features]
    true_euler = quat_to_euler(true_quaternions)
    pred_euler = quat_to_euler(pred_quaternions)
    difference = pred_euler - true_euler
    # sum over joint dimensions
    skeleton_loss = tf.reduce_sum(tf.square(difference), axis=-1)
    # apply sqrt following https://ait.ethz.ch/projects/2019/spl/downloads/spl_iccv19.pdf
    # https://github.com/eth-ait/spl/blob/master/spl/model/base_model.py
    skeleton_loss = tf.sqrt(skeleton_loss)
    # sum over joints in skeleton
    skeleton_loss = tf.reduce_sum(skeleton_loss, axis=-1)
    # avarage over batch and time dimensions
    return tf.reduce_mean(skeleton_loss)

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4
    
    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result

def normalize_quaternions(qs):
    return tfq.normalize(qs)

def noramlization_loss(vec):
    norm_loss = (tf.reduce_sum(vec**2, axis=-1) - 1.)**2
    return 0.01 * tf.reduce_mean(norm_loss)

def conjugate_quaternion(quaternion):
    xyz, w = tf.split(quaternion, [3, 1], axis=-1)
    return tf.concat((-xyz, w), axis=-1)

def quaternion_multiply(quat1, quat2):
    x1, y1, z1, w1 = tf.split(quat1, 4, axis=-1)
    x2, y2, z2, w2 = tf.split(quat2, 4, axis=-1)
    x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    return tf.concat((x, y, z, w), axis=-1)

def rotate(point, quaternion):
    padding = tf.eye(2 * tf.rank(point), dtype=tf.int64)
    padding = tf.reshape(padding[-1], [-1, 2])
    point = tf.pad(point, padding, mode='CONSTANT')
    point = quaternion_multiply(quaternion, point)
    point = quaternion_multiply(point, conjugate_quaternion(quaternion))
    xyz, _ = tf.split(point, (3, 1), axis=-1)
    return xyz

def get_quaternion_from_vectors(vec1, vec2):
    vec1 = tf.nn.l2_normalize(vec1, axis=-1)
    vec2 = tf.nn.l2_normalize(vec2, axis=-1)
    cos_theta = tf.reduce_sum(vec1 * vec2, axis=-1, keepdims=True)

    real = 1.0 + cos_theta
    axis = tf.linalg.cross(vec1, vec2)

    x, y, z = tf.split(vec1, 3, -1)
    x_bigger = tf.abs(x) > tf.abs(z)
    x_bigger = tf.concat([x_bigger] * 3, axis=-1)
    antiparallel_ax = tf.where(
        x_bigger, 
        tf.concat((-y, x, tf.zeros_like(z)), axis=-1),
        tf.concat((tf.zeros_like(x), -z, y), axis=-1))

    is_antiparallel = real < 1e-6
    is_antiparallel = tf.concat([is_antiparallel] * 4, axis=-1)
    rotation = tf.where(is_antiparallel,
        tf.concat((antiparallel_ax, tf.zeros_like(real)), axis=-1),
        tf.concat((axis, real), axis=-1))

    return tf.nn.l2_normalize(rotation, axis=-1)
