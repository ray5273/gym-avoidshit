import gym
import gym_dodge
import tensorflow.compat.v1 as tf
env = gym.make('Dodge-v0')

class Load :
    def __init__(self, env, modelname):
        tf.disable_eager_execution()
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(modelname+'.meta')
        saver.restore(self.sess, modelname)
        tf.get_default_graph()

    def get_action(self, state):
        return self.sess.run("get_action:0", feed_dict={"x_ph:0":state.reshape(1, -1)})[0]

done = False
state = env.reset()
model = Load(env, "model2/model21300")
env.render()
while not done:
    action = model.get_action(state)
    state,reward,done,_ = env.step(action)
    env.render()

