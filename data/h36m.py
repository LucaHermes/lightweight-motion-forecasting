import numpy as np
import os

import quaternion_utils as Q
from data import h36m_utils as utils

class H36MDataset:

    def __init__(self, data_path):
        '''
        This class manages the H3.6M Dataset. It loads the data from the provided txt files.
        It also implements common evaluation procedures to replicate eval-scores.
        '''
        self.data_path = data_path
        self.train_subjects = [1, 7, 8, 9, 11]
        self.val_subjects = [6]
        self.test_subjects = [5]
        self.fps = 50
        self.build()

    def build(self):
        self.subjects = os.listdir(self.data_path)
        actions = os.listdir(os.path.join(self.data_path, self.subjects[0]))
        self.action_files = np.unique([ s for s in actions if s.endswith('.txt') ])
        self.action_names = np.unique([ s.split('_')[0] for s in actions if s.endswith('.txt') ])
        self.subject_ids = [ int(s[1:]) for s in self.subjects ]
        
        self.skeleton = np.array(SKELETON_H36M, dtype=np.float32)
        self.left_joints = np.array(LEFT_JOINTS, dtype=np.int64)
        self.right_joints = np.array(RIGHT_JOINTS, dtype=np.int64)
        self.parents = np.array(PARENTS)

        self.parents = self.parents.astype(np.float32)
        self.dataset = self._to_dataset()
        self.build_matrices()

    def build_matrices(self):
        parents = self.parents.astype(np.int64)
        self.parent_mat = np.eye(len(self.skeleton))[parents]
        self.parent_mat = self.parent_mat * (parents[:, np.newaxis] >= 0)
        self.parent_mat = self.parent_mat.astype(np.float32)
        self.adj = self.parent_mat + self.parent_mat.T
        self.kinematic_mat = self.kinematic_from_parent(self.parent_mat)

    def __iter__(self):
        '''
        Iterates over the dataset sequence-wise.
        This can be used for training or to construct a different dataset, i.e. 
        a tf.data.Dataset can be constructed using its from_generator method.
        '''
        for sub, act, t_id in self.iter_categories():
            yield self.dataset[sub][act][t_id]

    def __getitem__(self, key):
        return self.dataset[key]

    def as_generator(self, subset=None, shuffle=True):
        '''
        Returns a generator to iterator across subjects, actions and trials.
        subset
            One of None, 'training', 'validation', 'testing'
            Defines which sequences will be returned
        shuffle
            If True the dataset will be shuffled
        '''
        ds_keys = np.array(list(self.iter_categories()))

        if shuffle:
            perm = np.random.permutation(np.arange(len(ds_keys)))
            ds_keys = ds_keys[perm]

        if subset == 'training':
            subset = self.train_subjects
        elif subset == 'validation':
            subset = self.val_subjects
        elif subset == 'testing':
            subset = self.test_subjects
        else:
            subset = self.subject_ids

        # check if the dataset was augmented
        mirrored = 'mirrored_ids' in dir(self)
        downsampled = 'ds_ids' in dir(self)

        def generator(): 
            for sub, act, t_id in ds_keys:
                sub = int(sub)
                t_id = int(t_id)
                seq_len = len(self.dataset[sub][act][t_id])

                if not sub in subset:
                    continue

                if downsampled:
                    downsampling_id = self.ds_ids[sub][act][t_id]
                else:
                    downsampling_id = np.zeros([seq_len])

                if mirrored:
                    mirror_id = self.mirrored_ids[sub][act][t_id]
                else:
                    mirror_id = np.zeros([seq_len])

                yield self.dataset[sub][act][t_id], sub, act, t_id, downsampling_id, mirror_id

        return generator

    def kinematic_from_parent(self, parent_mat):
        '''
        Constructs a kinematic matrix, from a parent matrix,
        which is a directed adjacency matrix.
        The kinematic matrix contains the causal chains of
        which joints affect the others.
        i.e. when the neck joint rotates, the head rotates likewise.
        '''
        joints = parent_mat.shape[0]
        kinematic_mat = np.copy(parent_mat)
        ref = np.zeros_like(kinematic_mat)

        while not np.all(kinematic_mat == ref):
            ref = np.copy(kinematic_mat)
            kinematic_mat += kinematic_mat @ kinematic_mat
            kinematic_mat = np.minimum(kinematic_mat, 1)

        # add selfloops, because not only decendents are affected,
        # by transformations, but the joints itself as well
        #kinematic_mat += np.eye(joints)

        return kinematic_mat

    def get_eval_sequences(self, action, n_samples=8):
        '''
        Returns sequences from a common evaluation protocol, used in 
        SRNN (https://arxiv.org/pdf/1511.05298.pdf)
        QuaterNet (https://arxiv.org/abs/1805.06485)
        Martinez et. al. (https://arxiv.org/pdf/1705.02445.pdf)
        ...
        '''
        if self.fps != 25:
            raise UserWarning('Common evaluation protocols use 25 fps',
                'currently this dataset is using %d fps.' % self.fps)

        subject = 5

        eval_data = utils.get_test_data(self, subject, action, n_samples=n_samples)

        return eval_data

    def iter_categories(self):
        '''
        Iterate over all keys of the dataset.
        '''
        for subject in self.dataset.keys():
            for action in self.dataset[subject].keys():
                for trial_id in [1, 2]:
                    yield subject, action, trial_id

    def reorder_to_rlc(self):
        for sub, act, t_id in self.iter_categories():
            trajectory = self.dataset[sub][act][t_id]
            self.dataset[sub][act][t_id] = trajectory[:, RLC_ORDERING]

        self.left_joints = np.array(RLC_LEFT_JOINTS)
        self.right_joints = np.array(RLC_RIGHT_JOINTS)
        self.parents = np.array(RLC_PARENTS, dtype=np.float32)
        self.skeleton = self.skeleton[RLC_ORDERING]
        self.build_matrices()
        return self

    def downsample(self, factor):
        '''
        Downsamples this dataset in memory by the specified factor.
        Does not through away samples, but concatenates the downsampled versions.
        '''
        # keeps track of the different downsampled versions
        # this is important for the evaluation.
        self.ds_ids = { s : {} for s in self.subject_ids }

        for sub, act, t_id in self.iter_categories():
            trajectory = self.dataset[sub][act][t_id]
            downsampled = []
            ds_id = []

            if act not in self.ds_ids[sub]:
                self.ds_ids[sub][act] = { i : {} for i in [1, 2] }

            for s_id in range(factor):
                ds = trajectory[s_id::factor]
                downsampled.append(ds)
                ds_id.extend([s_id] * len(ds))

            downsampled = np.concatenate(downsampled, axis=0)
            self.dataset[sub][act][t_id] = downsampled
            self.ds_ids[sub][act][t_id] = np.array(ds_id) #(range(factor), len_per_seq, axis=0)

        self.fps = self.fps // factor

        return self

    def mirror(self):
        self.mirrored_ids = { s : {} for s in self.subject_ids }

        for sub, act, t_id in self.iter_categories():
            trajectory = self.dataset[sub][act][t_id]

            if act not in self.mirrored_ids[sub]:
                self.mirrored_ids[sub][act] = { i : {} for i in [1, 2] }

            left_joints = trajectory[:,self.left_joints]
            right_joints = trajectory[:,self.right_joints]

            mirrored = np.copy(trajectory)
            mirrored[:,self.right_joints] = left_joints
            mirrored[:,self.left_joints] = right_joints
            # flip y and z values of the quaternions
            mirrored[:, :, [1, 2]] *= -1

            mirrored = Q.qfix(mirrored)

            m_ids = np.concatenate(([0] * len(trajectory), [1] * len(mirrored)))
            ds_ids = np.concatenate((self.ds_ids[sub][act][t_id], self.ds_ids[sub][act][t_id]))

            self.dataset[sub][act][t_id] = np.concatenate((trajectory, mirrored))
            self.mirrored_ids[sub][act][t_id] = m_ids
            self.ds_ids[sub][act][t_id] = ds_ids

        return self

    def get_mirror_start_idx(self, sub, act, t_id):
        return np.where(self.mirrored_ids[sub][act][t_id] == 1)[0][0]

    def _to_dataset(self):
        self.dataset = { s : {} for s in self.subject_ids }
        n_values = 0

        # load numpy arrays
        for subject in self.subjects:
            for action in self.action_files:
                file = os.path.join(self.data_path, subject, action)
                # load the txt file
                subject_action_set = np.loadtxt(file, delimiter=',')
                # reshape into shape [time, joints, exp_map_features]
                subject_action_set = np.reshape(subject_action_set, [len(subject_action_set), -1, 3])
                subject_action_set = subject_action_set.astype(np.float32)
                # remove the first transformation. According to facebook-research this is corrupted
                # https://github.com/facebookresearch/QuaterNet/blob/master/prepare_data_short_term.py
                subject_action_set = subject_action_set[:, 1:]
                # convert exponential maps to quaternions
                subject_action_set = Q.expmap_to_quaternion_np(-subject_action_set)
                subject_action_set = Q.qfix(subject_action_set)

                action_name, action_nr = action.split('_')
                action_nr = int(action_nr.split('.')[0])
                subject_nr = int(subject[1])

                if not action_name in self.dataset[subject_nr]:
                    self.dataset[subject_nr][action_name] = { i : {} for i in [1,2] }

                self.dataset[subject_nr][action_name][action_nr] = subject_action_set

        self.data_std = np.mean([ self.dataset[s][a][t].std((0,1)) for s, a, t in self.iter_categories() ], axis=0)
        self.data_mean = np.mean([ self.dataset[s][a][t].mean((0,1)) for s, a, t in self.iter_categories() ], axis=0)

        return self.dataset

# These values where taken from facebook-reseach:
# https://github.com/facebookresearch/QuaterNet/blob/master/short_term/dataset_h36m.py
SKELETON_H36M = [
       [   0.      ,    0.      ,    0.      ], # 0  - Hip - root joint
       [-132.948591,    0.      ,    0.      ], # 1  - Right Hip
       [   0.      , -442.894612,    0.      ], # 2  - Right Knee
       [   0.      , -454.206447,    0.      ], # 3  - Right Ancle
       [   0.      ,    0.      ,  162.767078], # 4  - Right Foot
       [   0.      ,    0.      ,   74.999437], # 5  - Right Toe
       [ 132.948826,    0.      ,    0.      ], # 6  - Left Hip
       [   0.      , -442.894413,    0.      ], # 7  - Left Knee
       [   0.      , -454.20659 ,    0.      ], # 8  - Left Ancle
       [   0.      ,    0.      ,  162.767426], # 9  - Left Foot
       [   0.      ,    0.      ,   74.999948], # 10 - Left Toe
       [   0.      ,    0.1     ,    0.      ], # 11 - Hip2 - controls upper body
       [   0.      ,  233.383263,    0.      ], # 12 - Spine
       [   0.      ,  257.077681,    0.      ], # 13 - Thorax
       [   0.      ,  121.134938,    0.      ], # 14 - Neck/Nose
       [   0.      ,  115.002227,    0.      ], # 15 - Head
       [   0.      ,  257.077681,    0.      ], # 16 - Thorax2
       [   0.      ,  151.034226,    0.      ], # 17 - Left Shoulder
       [   0.      ,  278.882773,    0.      ], # 18 - Left Elbow
       [   0.      ,  251.733451,    0.      ], # 19 - Left Wrist
       [   0.      ,    0.      ,    0.      ], # 20 - Left Hand1
       [   0.      ,    0.      ,   99.999627], # 21 - Left Hand2
       [   0.      ,  100.000188,    0.      ], # 22 - Left Hand3
       [   0.      ,    0.      ,    0.      ], # 23 - Left Hand4
       [   0.      ,  257.077681,    0.      ], # 24 - Thorax3
       [   0.      ,  151.031437,    0.      ], # 25 - Right Shoulder
       [   0.      ,  278.892924,    0.      ], # 26 - Right Elbow
       [   0.      ,  251.72868 ,    0.      ], # 27 - Right Wrist
       [   0.      ,    0.      ,    0.      ], # 28 - Right Hand1
       [   0.      ,    0.      ,   99.999888], # 29 - Right Hand2
       [   0.      ,  137.499922,    0.      ], # 30 - Right Hand3
       [   0.      ,    0.      ,    0.      ]  # 31 - Right Hand4
]

PARENTS = [-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0,  11, 12, 13, 14, 12,
            16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]

RIGHT_JOINTS = [1, 2, 3, 4, 5,  24, 25, 26, 27, 28, 29, 30, 31]
LEFT_JOINTS =  [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]

# an ordering like this is needed by StructureGCN, this orders the parent joints
# such that the first parent is a right side, second is left side and third is a center body part
RLC_ORDERING = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 24, 25, 26, 27, 28, 
                29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 13, 14, 15]

RLC_RIGHT_JOINTS =  [1, 2, 3, 4, 5,  13, 14, 15, 16, 17, 18, 19, 20]
RLC_LEFT_JOINTS =   [6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28]
RLC_PARENTS =       [-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0,  11, 12, 13, 14, 15,
                      16, 17, 16, 19, 12, 21, 22, 23, 24, 25, 24, 27, 12, 29, 30]