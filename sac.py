import os
import numpy as np
import gym
import gym_dodge
import tensorflow.compat.v1 as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tf.disable_eager_execution()

def placeholders(*args):
    return [tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,)) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

LOG_STD_MAX = 2
LOG_STD_MIN = -20
def mlp_gaussian_policy(x, a, hidden_sizes, activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=None)
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    pre_sum = -0.5 * (((pi - mu) / (tf.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    logp_pi = tf.reduce_sum(pre_sum, axis=1)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    x = 1 - pi ** 2
    clip_up = tf.cast(x > 1, tf.float32)
    clip_low = tf.cast(x < -1, tf.float32)
    logp_pi -= tf.reduce_sum(tf.log(x + tf.stop_gradient((1 - x) * clip_up - (1 + x) * clip_low) + 1e-6), axis=1)
    return mu, pi, logp_pi

def mlp_actor_critic(x, a, hidden_sizes=[512,512], action_space=None):
    activation = tf.nn.relu
    with tf.variable_scope('pi'):
        mu, pi, logp_pi = mlp_gaussian_policy(x, a, hidden_sizes, activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale
    vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([x,a], axis=-1))
    return mu, pi, logp_pi, q1, q2


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

def sac(env):
    epochs              = 2000
    replay_size         = 100000
    batch_size          = 256
    start_predict       = 10
    max_ep_len          = 1000
    save_freq           = 10
    gamma               = 0.99
    polyak              = 0.995
    lr                  = 0.0003
    epsilon             = 0.0001
    hidden_sizes        = [256, 256]
    render              = True

    action_space = env.action_space
    print(env.observation_space.shape)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    x_ph, a_ph, x2_ph, r_ph, d_ph = placeholders(obs_dim, act_dim, obs_dim, None, None)
    x_ph = tf.identity(x_ph, "x_ph")

    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1_a, q2_a  = mlp_actor_critic(x_ph, a_ph, hidden_sizes=hidden_sizes, action_space=action_space)
    mu = tf.identity(mu, 'get_action')

    with tf.variable_scope('main', reuse=True):
        _, _, _, q1_pi, q2_pi = mlp_actor_critic(x_ph, pi, hidden_sizes=hidden_sizes, action_space=action_space)
        _, pi_next, logp_pi_next, _, _ = mlp_actor_critic(x2_ph, a_ph, hidden_sizes=hidden_sizes, action_space=action_space)

    with tf.variable_scope('target'):
        _, _, _, q1_pi_targ, q2_pi_targ = mlp_actor_critic(x2_ph, pi_next, hidden_sizes=hidden_sizes, action_space=action_space)

    target_entropy = tf.cast(-act_dim, tf.float32)
    log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
    alpha = tf.exp(log_alpha)

    min_q_pi = tf.minimum(q1_pi, q2_pi)
    min_q_pi_targ = tf.minimum(q1_pi_targ, q2_pi_targ)
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*(min_q_pi_targ - alpha*logp_pi_next))
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_a)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_a)**2)
    value_loss = q1_loss + q2_loss
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi)
    alpha_backup = tf.stop_gradient(logp_pi + target_entropy)
    alpha_loss  = -tf.reduce_mean(log_alpha * alpha_backup)

    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=get_vars('main/q'))

    alpha_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
    with tf.control_dependencies([train_value_op]):
        train_alpha_op = alpha_optimizer.minimize(alpha_loss, var_list=get_vars('log_alpha'))

    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    step_ops = [pi_loss, q1_loss, q2_loss, q1_a, q2_a, logp_pi, target_entropy, alpha_loss, alpha,
                train_pi_op, train_value_op, train_alpha_op, target_update]

    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    ep_num = 0

    while(ep_num < epochs):

        if ep_num > start_predict:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        if render :
            env.render()
        ep_ret += r
        ep_len += 1

        if ep_len==max_ep_len :
            d = False

        replay_buffer.store(o, a, r, o2, d)
        o = o2

        if d or (ep_len == max_ep_len):
            ep_num += 1
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'], x2_ph: batch['obs2'], a_ph: batch['acts'], r_ph: batch['rews'], d_ph: batch['done']}
                sess.run(step_ops, feed_dict)
            print(ep_num, " \tLENGTH :", ep_len, " \tREWARD :", ep_ret)
            if ep_num%save_freq == 0 :
                saver.save(sess, 'model/model' + str(ep_num))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

if __name__ == '__main__':
    env = gym.make('Dodge-v0')
    sac(env)
