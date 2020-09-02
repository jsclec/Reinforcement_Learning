# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
   
    
    V = initV

    print(V)

    delta = 0

    Q = np.zeros((env.spec.nS, env.spec.nA))

    stop_cond = True

    num_iter = 0


    while(stop_cond and num_iter < 1000):

        delta = 0

        for s in range(len(V)):

            num_iter +=1      

            v = V[s]    
            
            action_sum = 0
            
                                                                                    # update in place V(s) = sum over a pi(a|s) * sum over s' 

            for a in range(env.spec.nA):
              
                s_prime = env.TD

                Reward_possibility = env.R

                update = Reward_possibility[s][a] + env.spec.gamma*V                # r+gamma*V(S')

                target = np.multiply(s_prime[s][a], update)                         # p(s"|s,a)*[r+gamma*V(s')]

                Q[s][a] = np.sum(target)

                action_sum += pi.action_prob(s,a)* Q[s][a]
            
            V[s] = action_sum

            if num_iter == 1:
                delta = abs(v-V[s])
            else:
                delta = max(delta, abs(v - V[s]))

        if delta < theta:

            stop_cond = False
    
    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
   
    policy_stable = True

    class policy(Policy):

        def __init__(self):
            self.s = np.ones((env.spec.nS,env.spec.nA))/env.spec.nA   #start with normalized equal probabilities
            self.Q = np.zeros((env.spec.nS, env.spec.nA))             #intialize a Q matrix for pi
            self.V = np.zeros((env.spec.nS))                          #initialize a V array for pi



        def action_prob(self, state, act) -> float:                   #using a deterministic policy to set the probabilities for pi(a|s)
            if act == int(self.action(state)):
                return 1
            else:
                return 0


        def action(self, state:int) -> int:
            act = np.argmax(self.Q[state])
            return act
        

                                                                     #for each policy iteration, get the value function and Q

    pi = policy()
    
    old_policy = np.zeros((env.spec.nA))

    
    num_iter = 0


                                                                     #update pi to choose the greedy Q[s][a]
    while(policy_stable):

        pi.V, pi.Q = value_prediction(env, pi, pi.V, theta)

        num_stable = 0

        num_iter += 1

        for i in range(env.spec.nS):   

            old_policy = pi.s[i]

                                                                     #pi(s) equals the max over all a q(s,a)
            for action in range(env.spec.nA):

                pi.s[i][action] = pi.action_prob(i,action)           #pi(a|s) = 1 if argmax_a {Q(S,a)}

            truth = np.equal(old_policy, pi.s[i])
                                                                     #check if policy was updated
            if all(truth):

                num_stable += 1                                      #if pi(s) is stable, increment 

        if num_stable == env.spec.nS:                                #if num of stable states is |S|, then we're done
            
            policy_stable = False
        
    
    return pi.V, pi

