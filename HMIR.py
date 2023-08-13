'''
@qingyang li
RL for making decisions on model update
'''

import os
import numpy as np
import tensorflow as tf

import tensorlayer as tl


class HMIR(object):
    """
    DDPG class
    """

    def __init__(self, params):
        self.params = params
        # memory：
        s_dim = self.params['n_properties'] * self.params['n_nodes']
        a_dim = 1
        a_bound = self.params['a_bound']
        self.memory = np.zeros((self.params['memory_capacity'], s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.actor_loss = []
        self.critic_loss = []
        self.accum_reward = 0
        self.reward = []
        self.action_value = []

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=128, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.DropconnectDense(0.8)(x)
            # x = tl.layers.Dense(n_units=16, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(x)
            # x = tl.layers.DropconnectDense(0.8)(x)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)
            return tl.models.Model(inputs=inputs, outputs=x)

        def get_critic(input_state_shape, input_action_shape):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=128, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.DropconnectDense(0.8)(x)
            # x = tl.layers.Dense(n_units=16, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(x)
            # x = tl.layers.DropconnectDense(0.8)(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor([None, s_dim])
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        self.critic_target = get_critic([None, s_dim], [None, a_dim])
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.params['TAU'])  # soft replacement

        self.actor_opt = tf.optimizers.Adam(self.params['LR_A'])
        self.critic_opt = tf.optimizers.Adam(self.params['LR_C'])

    def ema_update(self):
        """

        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        action = self.actor(np.array([s], dtype=np.float32))[0]
        # print(action)
        # sess = tf.compat.v1.Session()
        # action_result = sess.run(action)
        action_result = action.numpy()
        # print(action_result)
        self.action_value.append(action_result)
        return action_result

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(self.params['memory_capacity'], size=self.params['batch_size'])
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1:-self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # Critic：
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + self.params['GAMMA'] * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
            self.critic_loss.append(q[0])  # record loss of critic network
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor：
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)
            self.actor_loss.append(q[0] * (-1))  # record loss of actor network
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.reward.append(self.accum_reward)

        self.ema_update()

    # save s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        #
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        #
        transition = np.hstack((s, a, [r], s_))

        # pointer
        # index

        index = self.pointer % self.params['memory_capacity']  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        self.accum_reward += r


    def save_model(self, filepath1, filepath2):
        self.actor.save_weights(filepath1)
        self.critic.save_weights(filepath2)

    def load_model(self, filepath1, filepath2):
        self.actor.load_weights(filepath1)
        self.critic.load_weights(filepath2)

