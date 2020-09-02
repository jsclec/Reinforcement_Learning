# -*- coding: utf-8 -*-


from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],             ###this is a trajectory auto generated by a testing file that gives (s,a,r,s')  
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:

    # Sutton Book p. 144
    

    V = initV

    df = env_spec.gamma

    G = 0.0


    for episode in range(len(trajs)):                       

        steps = len(trajs[episode])

        T = steps

        G = 0.0

        

        for t in range(steps):

            tau = t - n + 1

            s_up = trajs[episode][tau][0]

            G = 0.0                              #re-intialize the return for each episode


            if tau >= 0:                         #here tau will be the state that we are attempting to update

                for i in range(tau+1, min(tau+n, T)+1):


                    G = G + pow(df, i-tau-1)*trajs[episode][i-1][2]             #discounted return for all n-1 steps from current state

                  

                if (tau + n) < T:

                    G = G + pow(df, n)*V[trajs[episode][tau+n][0]]              #the discounted value of the state n steps away

                V[s_up] = V[s_up] + alpha*(G - V[s_up])                         #update for V

            if tau == T-1:                                                      #if we are at the pre-terminal state, then we are done

                break



                

    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
   

    
    # Sutton Book p. 149

    class policy(Policy):
    
        def __init__(self, initQ):
            self.s = np.ones((env_spec.nS,env_spec.nA))/env_spec.nA   #start with normalized equal probabilities
            self.Q = initQ                                            #initialize a V array for pi



        def action_prob(self, state, act) -> float:                   #using a deterministic policy to set the probabilities for pi(a|s)
            if act == int(self.action(state)):
                return 1
            else:
                return 0


        def action(self, state:int) -> int:                           #greedy action selection
            act = np.argmax(self.Q[state])
            return act

        def policyUpdate(self,state):                                 #policy update
            for i in range(env_spec.nA):
              pi.s[state][i] = pi.action_prob(state,i)

    pi = policy(initQ)

    df = env_spec.gamma

    G = 0.0

    for episode in range(len(trajs)):

        steps = len(trajs[episode])

        T = steps

        G = 0.0

        for t in range(steps):

            tau = t - n + 1

            state_update = trajs[episode][tau][0]

            action = trajs[episode][tau][1]

            G = 0.0

            rho = 1.0

            if tau >= 0:

                for i in range(tau + 1, min(tau + n, T-1)):                         #relative importance of the return given by the ratio of likelihoods of choosing action under each policy

                    num = pi.s[trajs[episode][i][0]][trajs[episode][i][1]]

                    den = bpi.action_prob(trajs[episode][i][0], trajs[episode][i][1])

                    rho = rho * (num/den)

                for i in range(tau+1, min(tau+n, T)+1):

                    G = G + pow(df, i-tau-1)*trajs[episode][i-1][2]

                if (tau + n) < T:

                    G = G + pow(df, n)*pi.Q[trajs[episode][tau+n][0]][trajs[episode][tau+n][1]]

                pi.Q[state_update][action] = pi.Q[state_update][action] + rho*alpha*(G - pi.Q[state_update][action])             #

                pi.policyUpdate(state_update)

            if tau == T-1:

                break

    
    Q = pi.Q
    

    return Q, pi

