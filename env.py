import gym
import numpy as np
import datetime
import tensorflow as tf
import copy

from replay_buffer import ReplayBuffer
from learning import Agent


class Player:

    def __init__(self, env_id="LunarLander-v2", lr=1e-3, gamma=0.99, epsilon_decay=0.995,
                 epsilon_end=0.01):
        self.env = gym.make(env_id)
        self.replay_buffer = ReplayBuffer(1e6, self.env.observation_space.shape)
        self.agent = Agent(self.env.observation_space.shape, self.env.action_space.n, lr,
                           gamma, epsilon_decay, epsilon_end)

        # Logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/' + current_time + '/train'
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()
        self.cntr = 0
        tf.summary.experimental.set_step(self.cntr)

    def main_loop(self, n_games=1000, batch_size=128):
        ep_max = 500
        for i in range(n_games):
            done = False
            episode_r = 0
            s = self.env.reset()
            actions = [0] * self.env.action_space.n
            c = 0
            while not done:
                a = self.agent.choose_action(s[np.newaxis])
                actions[a] += 1
                s_, r, done, _ = self.env.step(a)
                if self.cntr > 2000000:
                    self.env.render()
                episode_r += r
                self.replay_buffer.store_transition(s, a, s_, r, done)
                s = s_

                batch = self.replay_buffer.sample_batch(batch_size)
                if batch is not None:
                    loss = self.agent.learn(*batch)
                    self.log_loss(loss)

                c += 1
                if c > ep_max:
                    break

            self.log_stats(episode_r)
            if not i % 10:
                print(episode_r, actions)

    def log_loss(self, loss):

        tf.summary.scalar(name="q/loss", data=loss)
        self.cntr += 1
        tf.summary.experimental.set_step(self.cntr)

    def log_stats(self, episodic):
        tf.summary.scalar(name="q/reward", data=episodic)
        tf.summary.scalar(name="q/eps", data=self.agent.Q.epsilon)
        tf.summary.scalar(name="q/gamma", data=self.agent.Q.gamma)






