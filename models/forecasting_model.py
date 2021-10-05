import tensorflow as tf
import quaternion_utils as Q
import kinematic_utils as K
import train_utils as T

class ForecastingModel(tf.keras.Model):

    def __init__(self, encoder, skeleton, predict_velocities=True, **kwargs):
        '''
        Autoregressive forecasting model.
        '''
        super(ForecastingModel, self).__init__()
        self.encoder = encoder
        self.skeleton = skeleton
        self.predict_velocities = predict_velocities

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.training_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.stats = {}

    def call(self, x, future_steps=10, training=True):
        # x shape: [batch, time, joints, embd]
        x, adj, parent_mat, kin_mat = x
        time = tf.shape(x)[1]

        # ADD SELF LOOPS if not already present
        adj += tf.eye(32)
        adj = tf.minimum(adj, 1.)
        adj = adj[:, tf.newaxis]
        parent_mat = parent_mat[:, tf.newaxis]
        kin_mat = kin_mat[:, tf.newaxis]
        
        # calculate velocity quaternions
        diff = Q.get_difference_quaternions(x[:,:-1], x[:,1:])
        diff = tf.pad(diff, [[0,0], [1, 0], [0,0], [0,0]])
        
        # predict the next n steps autoregressively
        ta = tf.TensorArray(tf.float32, size=future_steps)
        
        for s in range(future_steps):                
            velocity = self.encoder(diff, (adj,), (parent_mat,), (kin_mat,), training=training)
            velocity = velocity[:,-1:]
            
            # compute quaternion regularization loss
            normalization_penalty = Q.noramlization_loss(velocity)
            self.add_loss(normalization_penalty)     
            
            # integrate velocity
            absolute = Q.quaternion_multiply(velocity, x[:,-1:])
            absolute = tf.nn.l2_normalize(absolute, axis=-1)
            
            # create next model input
            diff = tf.concat((diff[:,1:], velocity), axis=1)
            # add current prediction to the time series
            x = tf.concat((x[:,1:], absolute), axis=1)
            # absolute is a single timestep, remove this dim to apply stack to tensor array
            ta = ta.write(s, absolute[:,0])

        model_out = tf.transpose(ta.stack(), [1, 0, 2, 3])
        
        self.stats['enc_out'] = velocity
        
        return model_out

    def loss_fn(self, ground_truth, predictions):
        model_losses = tf.reduce_mean(self.losses)
        abs_error = tf.abs(ground_truth - predictions)
        angular_loss = tf.reduce_mean(abs_error)
        return model_losses + angular_loss

    def load(self, path, optimizer=None):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()
        ckpt = tf.train.Checkpoint(step=self.global_step, optimizer=optimizer, net=self)
        ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)

        # check if a checkpoint exists
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("No ckeckpoint found in", path)

    def save(self, path, optimizer=None):
        ckpt = tf.train.Checkpoint(step=self.global_step, optimizer=optimizer, net=self)
        ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
        ckpt_manager.save()

    @tf.function
    def train_step(self, x, y, future_steps, max_grad_norm, optimizer):
        with tf.GradientTape() as tape:
            predictions = self(x, future_steps=future_steps, training=True)

            loss = self.loss_fn(y, predictions)
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        gradients, global_grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
        optimizer.apply_gradients(zip(gradients, variables))

        step = self.global_step.assign_add(1)
        return predictions, loss, gradients, global_grad_norm


    def train(self, dataset, optimizer, epochs, val_set, log_dir, future_steps=10, max_grad_norm=100., ckpt_name='ar_model'):
        train_writer = tf.summary.create_file_writer(log_dir + '/train')

        for e in range(epochs):
            for x, y in dataset:

                one_step_zero_vel = x[0][:,-1:]

                predictions, loss, gradients, global_grad_norm = self.train_step(x, y, future_steps, max_grad_norm, optimizer)
                variables = self.trainable_variables
                step = self.global_step

                if step % 50 == 0:

                    total_stats = { 'loss' : loss }

                    with train_writer.as_default():
                        grad_norms = tf.nest.flatten(tf.nest.map_structure(tf.linalg.norm, gradients))
                        grad_norms = tf.stack(grad_norms)

                        weights = tf.nest.map_structure(lambda v: v if 'kernel' in v.name else [], variables)
                        biases = tf.nest.map_structure(lambda v: v if 'bias' in v.name else [], variables)

                        abs_weights = tf.nest.flatten(tf.nest.map_structure(
                            lambda v: tf.reshape(v, [-1]), 
                            weights))
                        abs_biases = tf.nest.flatten(tf.nest.map_structure(
                            lambda v: tf.reshape(v, [-1]), 
                            biases))

                        abs_weights = tf.concat(abs_weights, axis=0)
                        abs_biases = tf.concat(abs_biases, axis=0)
                        weights_std = tf.math.reduce_std(abs_weights)
                        biases_std = tf.math.reduce_std(abs_biases)
                        weights_mean = tf.math.reduce_mean(abs_weights)
                        biases_mean = tf.math.reduce_mean(abs_biases)
                        weights_clip = [weights_mean - 3 * weights_std, weights_mean + 3 * weights_std]
                        biases_clip = [biases_mean - 2 * biases_std, biases_mean + 2 * biases_std]

                        log_grad_norms = tf.math.log(grad_norms + 1e-8) #tf.add(grad_norms, 1.))
                        
                        tf.summary.histogram('sqrt_abs_enc_out', tf.math.log(tf.abs(self.stats['enc_out'])+1e-8), step=step)
                        tf.summary.histogram('l2_log_norm_gradients', tf.clip_by_value(log_grad_norms, 0.0, 0.8), step=step)
                        tf.summary.histogram('weights_abs', tf.clip_by_value(abs_weights, *weights_clip), step=step)
                        tf.summary.histogram('bias_abs', tf.clip_by_value(abs_biases, *biases_clip), step=step)
                        tf.summary.scalar('train_metrics/learning_rate', optimizer.learning_rate, step=step)
                        tf.summary.scalar('train_metrics/gradient_norm', global_grad_norm, step=step)
                        tf.summary.scalar('train_metrics/training_epoch', self.training_epoch, step=step)

                    #f_metrics = T.get_forecasting_metrics(true_traj, forecast, forecast_loss)					
                    r_metrics = T.get_rotational_metrics(y, predictions, one_step_zero_vel)

                    #T.dict_log(train_writer, 'forecasting_metrics', f_metrics, step)
                    T.dict_log(train_writer, 'rotational_metrics', r_metrics, step)
                    T.dict_log(train_writer, 'total_loss', total_stats, step)

            if self.training_epoch % 10 == 0:
                self.evaluate(val_set, log_dir, future_steps=future_steps)

            if self.training_epoch % 500 == 0:
                self.save('ckpts/' + 'epoch_%d_' % self.training_epoch + ckpt_name, optimizer)
            else:
                self.save('ckpts/' + ckpt_name, optimizer)

            self.training_epoch.assign_add(1)

    def evaluate(self, dataset, log_dir, future_steps=10, fps=25):
        writer = tf.summary.create_file_writer(log_dir + '/validation')
        forecast_mean_metrics = tf.keras.metrics.MeanTensor()
        rotational_mean_metrics = tf.keras.metrics.MeanTensor()
        rotational_mean_metrics80 = tf.keras.metrics.MeanTensor()
        rotational_mean_metrics160 = tf.keras.metrics.MeanTensor()
        rotational_mean_metrics320 = tf.keras.metrics.MeanTensor()
        rotational_mean_metrics400 = tf.keras.metrics.MeanTensor()

        # subtract 1 to get from frame-no. to array-index
        frame_from_ms = lambda ms: tf.cast(ms/1000 * fps - 1, tf.int64) # ms/sec_ms * fps

        for x, y in dataset:
            parent_mat = x[2]

            true_traj = K.forward_kinematics(parent_mat, self.skeleton, y)

            zero_vel_1 = x[0][:,-1:] # 2 steps used in validation to capture direction

            quats = self(x, future_steps=future_steps, training=False)

            quats_loss = self.loss_fn(y, quats)
            #forecast = self.apply_quaternions(source_traj, quats, parent_mat, kin_mat)
            forecast = K.forward_kinematics(parent_mat, self.skeleton, quats)

            forecast_loss = tf.keras.losses.mean_absolute_error(true_traj, forecast)
            forecast_loss = tf.reduce_mean(forecast_loss)

            f_metrics = T.get_forecasting_metrics(true_traj, forecast, forecast_loss)
            forecast_mean_metrics(tf.nest.flatten(f_metrics))

            #print('EVAL')
            r_metrics = T.get_rotational_metrics(y, quats, zero_vel_1)
            #print('80')
            r80_metrics = T.get_rotational_metrics(
                y[:,frame_from_ms(80)][:,tf.newaxis], 
                quats[:,frame_from_ms(80)][:,tf.newaxis], 
                zero_vel_1)
            #print('160')
            r160_metrics = T.get_rotational_metrics(
                y[:,frame_from_ms(160)][:,tf.newaxis], 
                quats[:,frame_from_ms(160)][:,tf.newaxis], 
                zero_vel_1)
            #print('320')
            r320_metrics = T.get_rotational_metrics(
                y[:,frame_from_ms(320)][:,tf.newaxis], 
                quats[:,frame_from_ms(320)][:,tf.newaxis], 
                zero_vel_1)
            #print('400')
            r400_metrics = T.get_rotational_metrics(
                y[:,frame_from_ms(400)][:,tf.newaxis], 
                quats[:,frame_from_ms(400)][:,tf.newaxis], 
                zero_vel_1)
            rotational_mean_metrics(tf.nest.flatten(r_metrics))
            rotational_mean_metrics80(tf.nest.flatten(r80_metrics))
            rotational_mean_metrics160(tf.nest.flatten(r160_metrics))
            rotational_mean_metrics320(tf.nest.flatten(r320_metrics))
            rotational_mean_metrics400(tf.nest.flatten(r400_metrics))

        f_keys = sorted(f_metrics.keys())
        r_keys = sorted(r_metrics.keys())
        forecast_mean = dict(zip(f_keys, forecast_mean_metrics.result()))
        rotational_mean = dict(zip(r_keys, rotational_mean_metrics.result()))
        rotational80_mean = dict(zip(r_keys, rotational_mean_metrics80.result()))
        rotational160_mean = dict(zip(r_keys, rotational_mean_metrics160.result()))
        rotational320_mean = dict(zip(r_keys, rotational_mean_metrics320.result()))
        rotational400_mean = dict(zip(r_keys, rotational_mean_metrics400.result()))

        total_stats = { 'loss' : quats_loss }

        T.dict_log(writer, 'forecasting_metrics', forecast_mean, self.global_step)
        T.dict_log(writer, 'total_loss', total_stats, self.global_step)
        T.dict_log(writer, 'rotational_metrics', rotational_mean, self.global_step)
        T.dict_log(writer, 'rotational_metrics_80ms', rotational80_mean, self.global_step)
        T.dict_log(writer, 'rotational_metrics_160ms', rotational160_mean, self.global_step)
        T.dict_log(writer, 'rotational_metrics_320ms', rotational320_mean, self.global_step)
        T.dict_log(writer, 'rotational_metrics_400ms', rotational400_mean, self.global_step)

        print(' * Step %d finished' % self.global_step)
        print('   - Val. forecast loss:\t%.3f' % forecast_mean['loss'])
        print('   - Val. forecast mae: \t%.3f' % forecast_mean['mae'])
