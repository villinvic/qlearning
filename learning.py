import tensorflow as tf
import numpy as np
import copy


class QNN(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(128, activation='relu', dtype=tf.float32)
        self.l2 = tf.keras.layers.Dense(128, activation='relu', dtype=tf.float32)
        self.q = tf.keras.layers.Dense(action_dim, activation='linear', dtype=tf.float32)

    def call(self, states):
        features = self.l1(states)
        features = self.l2(features)
        return self.q(features)


class Q(tf.keras.Model):
    def __init__(self, state_shape, action_dim, lr, gamma, epsilon_decay, epsilon_end, update_traget_freq=8000):
        super().__init__()
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.gamma_max = 0.998
        self.optimizer = tf.optimizers.Adam(lr=lr)

        self.QNN = QNN(action_dim)
        self.target_Q = copy.deepcopy(self.QNN)
        self.target_Q.trainable = False

        self.update_freq = update_traget_freq
        self.cntr = 0

    def call(self, states):
        return self.QNN(states)

    @tf.function
    def _learn(self, states, actions, states_, rewards, dones, gpu):
        device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        with tf.device(device):
            with tf.GradientTape() as tape:
                values = self.QNN(states)
                values_ = tf.stop_gradient(self.target_Q(states_))
                action_indexes = tf.stack([tf.range(tf.shape(actions)[0], dtype=actions.dtype), actions], axis=1)
                loss = tf.reduce_mean(tf.square(tf.reduce_max(values_, axis=1) * dones *
                                                self.gamma + rewards - tf.gather_nd(values, action_indexes)))

            grad = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return loss

    def learn(self, states, actions, states_, rewards, dones, gpu=0):

        loss = self._learn(states, actions, states_, rewards, dones, gpu)

        self.cntr += 1
        if not self.cntr % self.update_freq:
            # print('update')
            self.target_Q.set_weights(self.QNN.get_weights())

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return loss




class Agent:
    def __init__(self,  state_shape, action_dim, lr, gamma, epsilon_decay=0.999, epsilon_end=0.1):
        self.Q = Q(state_shape, action_dim, lr, gamma, epsilon_decay, epsilon_end)

        self.action_dim = action_dim

    @tf.function
    def _choose_action(self, state):
        return self.Q(state)

    def choose_action(self, state):
        if np.random.random() < self.Q.epsilon:
            a = np.random.randint(0, self.action_dim)
        else:
            a = np.argmax(self._choose_action(state))

        return a

    def learn(self, *args, **kwargs):
        return self.Q.learn(*args, **kwargs)
