from typing import Iterable
import numpy as np
import tensorflow as tf
import gym
import torch
from torch import nn
from torch import optim




class PiApproximationWithNN(nn.Module):
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        super(PiApproximationWithNN, self).__init__()
       
        
        self.num_inputs = state_dims
        self.num_actions = num_actions
        self.alpha = alpha

        #intialize a two hidden layer network with ReLu activation functions and 32 nodes
        self.Network = nn.Sequential(nn.Linear(self.num_inputs, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32),nn.Linear(32, self.num_actions), nn.Softmax(dim = -1))

        self.optimizer = optim.Adam(self.Network.parameters(), lr = self.alpha)   #optimizer to minimize the loss function

        




        





    def __call__(self,s) -> int:
        # TODO: implement this method

        #simple forward pass through the network

        action_prob = self.Network(torch.FloatTensor(s))                                              #will output the probabilites of each action

        action = np.random.choice(self.num_actions, p = action_prob.detach().numpy())

        return action

    def update(self, s, a, gamma_t, delta):         #update the network using a mini batch of returns

        self.optimizer.zero_grad()
        
        s_tensor = torch.FloatTensor(s)

        delta_tensor = torch.FloatTensor(delta)

        action_tensor = torch.LongTensor(a)

        logprob = torch.log(self.Network(s_tensor)) #this gives us the log probabilities of all actions from these states

        update = gamma_t*delta_tensor * logprob[np.arange(len(action_tensor)), action_tensor] #here we want to choose only the log probabilities of the actions weve taken

        loss = -1*update.mean() #network will learn faster if you normalize the update using the mean

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

        #now let's approximate the value function using Tensorflow
  
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

        

        init = tf.compat.v1.global_variables_initializer()      #initializes the computational graph
        
        self.sess.run(init)
        

    def __call__(self,s) -> float:
        # TODO: implement this method
        out = self.sess.run(self.output, feed_dict = {self.X: np.reshape((s), (1,-1))})
        return out[0]
        raise NotImplementedError()

    def update(self,s,G):

        self.sess.run(self.train_op, feed_dict = {self.G: np.reshape((G),(1,)), self.X: np.reshape((s), (1,-1))})

    


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    

    #implement REINFORCE with and without baseline
    

    

    G_o = []                         #return from the starting state

    disc_rewards = []                #store the discounted rewards working backwards from the terminal state

    num_actions = env.action_space.n

    

    

    for episode in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []                                            
        actions = []
        values = []
        delta = []
        probs = []

        steps = 0

        done = False
        

        while (not done):
            action = pi.__call__(s_0)
            
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
            
                    V.update(states[t], disc_rewards[t])                                        #update the value function

                delta = np.array([disc_rewards[t] - values[t] for t in range(len(values))])     #this is essentially our target for the batch

                pi.update(states, actions, gamma , delta)                                       #update the network

        

    return G_o


