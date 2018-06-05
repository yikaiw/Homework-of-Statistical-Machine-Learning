from __future__ import absolute_import
import tensorflow as tf
import gym
import numpy as np
import time
import tianshou as ts


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation_dim = env.observation_space.shape
    action_dim = env.action_space.n
    batch_size = 32
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)

    observation_ph = tf.placeholder(tf.float32, shape=(None,) + observation_dim)

    def my_network():
        net = tf.layers.dense(observation_ph, 64, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 64, activation=tf.nn.tanh)
        action_value = tf.layers.dense(net, action_dim, activation=None)
        return None, action_value

    critic = ts.value_function.DQN(my_network, observation_placeholder=observation_ph, has_old_net=True)
    actor = ts.policy.DQN(critic, epsilon_train=0.1, epsilon_test=0.05)
    soft_update_op = ts.get_soft_update_op(1e-2, [critic])

    critic_loss = ts.losses.value_mse(critic)
    critic_optimizer = tf.train.AdamOptimizer(1e-3)
    critic_train_op = critic_optimizer.minimize(critic_loss, var_list=list(critic.trainable_variables))

    data_buffer = ts.data.VanillaReplayBuffer(capacity=10000, nstep=1)
    process_functions = [ts.data.advantage_estimation.nstep_q_return(1, critic, use_target_network=True)]

    data_collector = ts.data.DataCollector(
        env=env,
        policy=actor,
        data_buffer=data_buffer,
        process_functions=process_functions,
        managed_networks=[critic]
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        critic.sync_weights()
        start_time = time.time()
        data_collector.collect(num_timesteps=5000)
        for i in range(int(1e8)):
            data_collector.collect(num_timesteps=1, episode_cutoff=200)
            feed_dict = data_collector.next_batch(batch_size)
            sess.run(critic_train_op, feed_dict=feed_dict)
            sess.run(soft_update_op)

            if i % 1000 == 0:
                print('Step {}, elapsed time: {:.1f} min'.format(i, (time.time() - start_time) / 60))
                ts.data.test_policy_in_env(actor, env, num_episodes=5, episode_cutoff=200)
