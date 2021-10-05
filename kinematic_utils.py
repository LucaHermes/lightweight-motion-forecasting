import tensorflow as tf
import quaternion_utils as Q

def forward_kinematics(parent_mat, offsets, quaternions):
    '''
    Runs forward kinematics.
    Applies quaternions to offsets and translates offset according to their parent
    in the kinematic order.
    '''
    rotations_world = tf.ones_like(quaternions) * Q.NULL_QUATERNION
    translations_world = tf.zeros_like(offsets)

    r = kinematic_map(forward_kinematic_step, parent_mat, offsets, 
        quaternions, rotations_world, translations_world)

    return r[-1]

def forward_kinematic_step(joints_vec, parent_mat, joints, quaternions, rotations_world, translations_world):
    '''
    Performs one step of forward kinematics.

    joints_vec: one-hot encoding that marks the origin joint of the subtree.
                that joint will be rotated as well
    kinematic_mat: Matrix containing the kinematic chains of the skeleton
    joints: [batch, time, joints, D]
            each batch contains a time series of skeletons. D is the dimensionality 
            joint representation.
    '''
    time = joints.shape[1]

    # get the id of the parent joint
    parent_one_hot = tf.matmul(parent_mat, joints_vec, transpose_a=True)
    parent_one_hot = tf.minimum(parent_one_hot, 1)
    #parent_id = tf.squeeze(tf.argmax(parent_one_hot, axis=-2))

    # get the world rotation of this joints parent and apply
    parent_rotation = rotations_world * parent_one_hot
    # add identity quaternion for all inactive joints
    parent_rotation = parent_mat @ parent_rotation + (1 - joints_vec) * Q.NULL_QUATERNION
    # the root has no parent, the matmul would return zeros
    # replace zeros with identity quaternion
    replace_ids = tf.greater(tf.reduce_sum(tf.abs(parent_rotation), axis=-1, keepdims=True), 0.)
    replace_ids = 1 - tf.cast(replace_ids, tf.float32)
    parent_rotation = parent_rotation + Q.NULL_QUATERNION * replace_ids

    # get the rotation of the parent joint
    parent_translation = translations_world * parent_one_hot
    # send this rotation to the child id
    parent_translation = parent_mat @ parent_translation

    # apply the rotation and translation to the parent joint
    joints = Q.rotate(joints, parent_rotation) + parent_translation
    # add joint as offset to translations_world
    translations_world += joints * joints_vec

    # add this joints rotation to rotations world:
    # mask other rotations
    joint_rotation = quaternions * joints_vec
    # replace zeros with identity quaternion
    joint_rotation = joint_rotation + (1 - joints_vec) * Q.NULL_QUATERNION
    # accumulate parent and joint quaternions
    joint_parent_rotation = Q.quaternion_multiply(parent_rotation, joint_rotation)
    rotations_world = Q.quaternion_multiply(rotations_world, joint_parent_rotation)

    return joints, (joints, quaternions, rotations_world, translations_world)

# ----------- This forward kinematics is not working yet --------------
def _forward_kinematics(quaternions, offsets, kinematic_mat):
    '''
    Runs forward kinematics.
    Applies the quaternions 
    '''
    kin_quats = kinematic_mat[tf.newaxis,...,tf.newaxis] * quaternions[:,tf.newaxis]

    # add identity quaternions instead of zeros
    kin_quats += (1 - kinematic_mat[tf.newaxis,...,tf.newaxis]) * [0.,0.,0.,1.]
    kin_quats = tf.transpose(kin_quats, [2, 0, 1, 3])

    # propagate rotations to children
    kin_quats = tf.scan(Q.quaternion_multiply, kin_quats)[-1]
    pose = Q.rotate(offsets, kin_quats)

    # sum up the limbs to assemble the skeleton
    skeleton = kinematic_mat[tf.newaxis,...,tf.newaxis] * pose[:,tf.newaxis]
    skeleton = tf.transpose(skeleton, [2, 0, 1, 3])
    skeleton = tf.scan(tf.add, skeleton)[-1]
    return skeleton

def kinematic_map(f, parent_mat, *args, from_root=True):
    '''
    Applies the function f across the kinematic structure of the skeleton.
    The function f should accept the following parameters:
    joints_vec
        One hot vector of the joints of the current iteration.
        This is the indicator of what part of the skeleton is currently being addressed.
    parent_mat
        The parent matrix of the skeleton.
    args
        Arguments that will be passed to f every time. Changes
        to args will remain.
    '''
    n_joints = parent_mat.shape[-1]
    parent_mat = parent_mat[:,tf.newaxis]
    results = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    # get the first joint indices
    if from_root:
        # get the root joint
        joints_vec = tf.where(tf.reduce_sum(parent_mat, axis=-1) == 0, 1., 0.)
    else:
        # get the leaf joints
        joints_vec = tf.where(tf.reduce_sum(parent_mat, axis=-2) == 0, 1., 0.)
        parent_mat = tf.transpose(parent_mat, [0,1,3,2])

    joints_vec = tf.expand_dims(joints_vec, axis=-1)

    i = 0

    # start at the root, and calculate quaternions successively
    while tf.reduce_all(tf.reduce_sum(joints_vec, axis=2) != 0):
        result, args = f(joints_vec, parent_mat, *args)
        results = results.write(i, result)
        # get the children of the current joints for the next iteration

        joints_vec = parent_mat @ joints_vec
        joints_vec = tf.minimum(joints_vec, 1.)
        i += 1

    return results.stack()