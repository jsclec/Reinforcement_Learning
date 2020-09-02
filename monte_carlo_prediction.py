from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
 
    # Sutton Book p. 109

    Q = initQ

    G = 0

    W = 1

    C = np.zeros((env_spec.nS, env_spec.nA))



    for episode in range(len(trajs)):

        rev = len(trajs[episode])

        G = 0

        W = 1

        for step in range(rev):

            if (W != 0):

                s = trajs[episode][rev-step-1][0]

                a = trajs[episode][rev-step-1][1]

                r = trajs[episode][rev-step-1][2]

                G = env_spec.gamma*G + r
            
                C[s][a] = C[s][a] + 1

                Q[s][a] = Q[s][a] + (W/C[s][a])*(G - Q[s][a])

                num = pi.action_prob(s,a)

                den = bpi.action_prob(s,a)

                W = W*(num/den)
    
    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:

    #Sutton Book p. 110

    Q = initQ

    G = 0

    W = 1

    C = np.zeros((env_spec.nS, env_spec.nA))



    for episode in range(len(trajs)):

        rev = len(trajs[episode])

        G = 0

        W = 1

        for step in range(rev):

            if (W != 0):

                s = trajs[episode][rev-step-1][0]       #we are working backwards since we are finding returns from each state

                a = trajs[episode][rev-step-1][1]

                r = trajs[episode][rev-step-1][2]

                G = env_spec.gamma*G + r        #return including discounted rewards for the ith state
            
                C[s][a] = C[s][a] + W           #cumulative weight of all returns so far

                Q[s][a] = Q[s][a] + (W/C[s][a])*(G - Q[s][a])

                num = pi.action_prob(s,a)

                den = bpi.action_prob(s,a)

                W = W*(num/den)             #equivalent to product of rho
    
    return Q




    
