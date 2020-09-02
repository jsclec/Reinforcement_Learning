from typing import Iterable
import numpy as np
import tensorflow as tf
import gym
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter



class PiApproximationWithNN(nn.Module):
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        super(PiApproximationWithNN, self).__init__()
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
        #tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        
        self.num_inputs = state_dims
        self.num_actions = num_actions
        self.alpha = alpha

        self.Network = nn.Sequential(nn.Linear(self.num_inputs, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32),nn.Linear(32, self.num_actions), nn.Softmax(dim = -1))

        self.optimizer = optim.Adam(self.Network.parameters(), lr = self.alpha)

        




        





    def __call__(self,s) -> int:
        # TODO: implement this method

        #simple forward pass through the network

        action_prob = self.Network(torch.FloatTensor(s))                                              #will output the probabilites of each action

        action = np.random.choice(self.num_actions, p = action_prob.detach().numpy())

        return action

    def update(self, s, a, gamma_t, delta):         #tried doing this with a single state-action pair but got errors

        self.optimizer.zero_grad()
        
        s_tensor = torch.FloatTensor(s)

        delta_tensor = torch.FloatTensor(delta)

        action_tensor = torch.LongTensor(a)

        logprob = torch.log(self.Network(s_tensor)) #this gives us the log probabilities of all actions from these states

        #print(logprob.data.cpu().numpy(),"\n")

        update = gamma_t*delta_tensor * logprob[np.arange(len(action_tensor)), action_tensor] #here we want to choose only the log probabilities of the actions weve taken

        loss = -1*update.mean() #it's supposed to learn faster if you normalize the update using the mean

        #print(loss.data.cpu().numpy())

        loss.backward() #this calculates the gradients

        self.optimizer.step() #this applies them
    

    


    

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        self.states_num = state_dims
        
        tf.compat.v1.reset_default_graph()
        
        self.sess = tf.compat.v1.Session()
        
        self.X = tf.compat.v1.placeholder('float' ,shape = [None, self.states_num])
        self.Y = tf.compat.v1.placeholder('float' , shape = [None,1])

        self.weights = {
            'h1': tf.Variable(tf.random.normal([self.states_num, 32])),
            'h2': tf.Variable(tf.random.normal([32,32])),
            'out': tf.Variable(tf.random.normal([32, 1]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random.normal([32])),
            'b2': tf.Variable(tf.random.normal([32])),
            'out': tf.Variable(tf.random.normal([1]))
        }

        self.layer_1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1']) 
        self.layer_1 = tf.nn.relu(self.layer_1)


        self.layer_2 = tf.add(tf.matmul(self.layer_1, self.weights['h2']),self.biases['b2'])
        self.layer_2 = tf.nn.relu(self.layer_2)
        
        self.output = tf.reshape(tf.matmul(self.layer_2, self.weights['out']) + self.biases['out'], name = "out", shape = [1])

        self.optimizer  = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.001)

        self.G = tf.compat.v1.placeholder(tf.float32, shape =  [1], name = "G")

        self.loss_op = tf.compat.v1.losses.mean_squared_error(self.G,self.output)

        self.train_op = self.optimizer.minimize(.5*self.loss_op)

        

        init = tf.compat.v1.global_variables_initializer()
        
        self.sess.run(init)
        

    def __call__(self,s) -> float:
        # TODO: implement this method
        out = self.sess.run(self.output, feed_dict = {self.X: np.reshape((s), (1,-1))})
        return out[0]
        raise NotImplementedError()

    def update(self,s,G):
        # TODO: implement this method

        self.sess.run(self.train_op, feed_dict = {self.G: np.reshape((G),(1,)), self.X: np.reshape((s), (1,-1))})

    


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    

    

    G_o = []

    disc_rewards = []

    num_actions = env.action_space.n

    

    

    for episode in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []                                            #####why do i have 34 states and rewards but only 33 actions
        actions = []
        values = []
        delta = []
        probs = []

        steps = 0

        done = False
        

        while (not done):
            action = pi.__call__(s_0)

            #action = np.random.choice(num_actions, p = action_prob)

            #probs.append(action_prob[action])
            
            new_state , reward, done, _ = env.step(action)

            states.append(s_0)

            rewards.append(reward)

            actions.append(action)

            values.append(V.__call__(s_0))

            s_0 = new_state

            steps +=1

            

            if done:

                disc_rewards = np.array([gamma**i*rewards[i] for i in range(len(rewards))])

                

                disc_rewards = disc_rewards[::-1].cumsum()[::-1]

                

                
                
                

                G_o.append(disc_rewards[0])

                for t in range(steps):
            
                    V.update(states[t], disc_rewards[t])

                delta = np.array([disc_rewards[t] - values[t] for t in range(len(values))])

                pi.update(states, actions, gamma , delta)

        

    return G_o

    raise NotImplementedError()

