import tensorflow as tf
import numpy as np

import kinematic_utils as K

# ------ evaluation utils ---------


def find_indices_srnn(data, action, prefix=50, suffix=100, subject=5, idx_offset=16, n_samples=8, seed=1234567890):
    '''
    Find the same indices as in SRNN:
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    '''
    rng = np.random.RandomState(seed)

    # each subject performs an action twice, get both trials:
    ds_starting_id_T1 = len(np.where(data.ds_ids[subject][action][1] == 0)[0])
    ds_starting_id_T2 = len(np.where(data.ds_ids[subject][action][2] == 0)[0])
    T1 = ds_starting_id_T1 - prefix - suffix
    T2 = ds_starting_id_T2 - prefix - suffix
    #T1 = data[subject][action][1].shape[0]//2 - prefix - suffix
    #T2 = data[subject][action][2].shape[0]//2 - prefix - suffix

    # generate n_samples, i.e. n_samples/2 for trial 1 and trial 2
    idx = [ rng.randint(idx_offset, [T1, T2]) for i in range(n_samples//2) ]
    idx = np.reshape(idx, [-1])

    # add dataset accessors for convenience
    n_trials = n_samples
    idx = list(zip([subject]*n_samples, [action]*n_samples, [1, 2]*n_trials, idx))
    
    return idx


def get_test_data(data, subject, action, n_samples=8):
    prefix = 50
    suffix = 100

    idx = find_indices_srnn(data, action, prefix, suffix, subject=subject, n_samples=n_samples)

    out = []

    for sub, act, trial, seq_idx in idx:
        chunk = data.dataset[sub][act][trial]
        chunk = chunk[seq_idx:][:prefix + suffix]

        out.append((
            chunk[:prefix], # add seed sequence
            chunk[prefix:]) # add target sequence
        )

    return out

# ------ training utils ------

N_PARALLEL = tf.data.experimental.AUTOTUNE

def split_traj(traj_len):
    def interleave_fn(data, subject, action, trial_id, downsample_id, mirror_id):
        # repeat the labels, trial_id is not needed
        subject = tf.repeat(subject, tf.shape(data)[0], axis=0)
        action = tf.repeat(action, tf.shape(data)[0], axis=0)

        ds = tf.data.Dataset.from_tensor_slices((data, subject, action, downsample_id, mirror_id))

        rnd_offset = tf.random.uniform([1]) * tf.cast(traj_len, tf.float32)
        rnd_offset = tf.cast(rnd_offset, tf.int64)
        ds = ds.skip(rnd_offset[0])
        ds = ds.batch(traj_len, drop_remainder=True)
        # remove all batches that contain two versions of the downsampled data or mirrored
        # i.e. one that starts at index 0 and one that start at index 1
        ds = ds.filter(lambda x, s, a, dsid, mid: tf.reduce_all(tf.equal(dsid[:1], dsid)))
        ds = ds.filter(lambda x, s, a, dsid, mid: tf.reduce_all(tf.equal(mid[:1], mid)))
        # keep only one action/subkect label per batch, discard dsid information
        ds = ds.map(lambda x, s, a, dsid, mid: (x, s[:1], a[:1]), num_parallel_calls=N_PARALLEL)
        return ds
    return interleave_fn

def sample_traj(traj_len, samples_per_trial, sampling=True):
    def sample_traj_fn(data, subject, action, trial_id, downsample_id, mirror_id):
        # find valid idis that dont overlap different downsampled/mirrored versions
        ds_segments = tf.signal.frame(downsample_id, traj_len, 1)
        mr_segments = tf.signal.frame(mirror_id, traj_len, 1)

        ds_valid = tf.reduce_all(tf.equal(ds_segments[:,:1], ds_segments), axis=-1)
        mr_valid = tf.reduce_all(tf.equal(mr_segments[:,:1], mr_segments), axis=-1)

        valid_starting_ids = tf.cast(tf.logical_and(ds_valid, mr_valid), tf.int32)
        valid_starting_ids = tf.range(tf.shape(mr_valid)[0]) * valid_starting_ids
        valid_starting_ids = tf.unique(valid_starting_ids[:-traj_len+1])[0]
        # if sampling, sample from the sequences, if not just split the sequence in chunks
        # of traj_len
        if sampling:
            starting_ids = tf.random.shuffle(valid_starting_ids)[:samples_per_trial]
            sequences = tf.signal.frame(data, traj_len, 1, axis=0)
            sequences = tf.gather(sequences, starting_ids)
        else:
            sequences = tf.signal.frame(data, traj_len, traj_len, axis=0)

        subject = tf.repeat(subject[tf.newaxis], tf.shape(sequences)[0], axis=0)
        action = tf.repeat(action[tf.newaxis], tf.shape(sequences)[0], axis=0)

        ds = tf.data.Dataset.from_tensors((sequences, subject[:,tf.newaxis], action[:,tf.newaxis]))
        ds = ds.unbatch()
        return ds
    return sample_traj_fn

def add_values(*values):
    def add_values_fn(*args):
        return (*args, *values)
    return add_values_fn

def add_joints_positions(skeleton_offsets):
    def add_forward_kinematic(quats, subject, action, adj, parent_mat, kinematic_mat):
        pose = K.forward_kinematics(parent_mat, skeleton_offsets, quats)
        return pose, quats, adj, parent_mat, kinematic_mat, subject, action
    return add_forward_kinematic

def as_tf_dataset(data, subset, traj_len, samples_per_trial=5, sampling=True):
    '''
    if sampling is True, the trajectories are sampled randomly from the sequences
    if sampling is False, the trajectories are returned over the whole dataset and in order
    also shuffling will be deactivated
    '''
    generator = data.as_generator(subset, shuffle=sampling)
    shuffle_buffer = len(np.array(list(data.iter_categories())))

    dataset = tf.data.Dataset.from_generator(generator, 
        (tf.float32, tf.int64, tf.string, tf.int64, tf.int64, tf.int64))

    # shuffle whole trajectories, i.e. shuffle subjects, actions and trials
    if sampling:
        dataset = dataset.shuffle(shuffle_buffer)

    # split the dataset into smaller sequences with length traj_len
    #interleave_fn = split_traj(traj_len)
    #dataset = dataset.interleave(interleave_fn, num_parallel_calls=N_PARALLEL)
    traj_sampler = sample_traj(traj_len, samples_per_trial, sampling=sampling)
    cycle_length = None if sampling else 1
    dataset = dataset.interleave(traj_sampler, cycle_length=cycle_length, num_parallel_calls=N_PARALLEL)

    dataset = dataset.map(add_values(
        data.adj,
        data.parent_mat, 
        data.kinematic_mat
    ), num_parallel_calls=N_PARALLEL)

    #dataset = dataset.batch(1)
    #dataset = dataset.map(add_joints_positions(tf.constant(data.skeleton)))

    return dataset