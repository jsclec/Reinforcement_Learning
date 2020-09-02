import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        
        #we can use tile coding to create features for the state space
        self.num_bins = [int(np.ceil((state_high[i] - state_low[i])/tile_width[i])+1) for i in range(len(state_high))]  
        
        
        self.num_actions = num_actions
        self.num_tilings = num_tilings

        self.num_dims = len(state_low)
        
        self.grid_size = 1

        for i in range(len(self.num_bins)):
            self.grid_size *= self.num_bins[i]
        
        
        
        


        self.tilings = []
        
        self.tiling_offset = [tile_width/num_tilings]                #use a tiling offset to increase granularity                   
    
        for action in range(num_actions): 

            for tile in range(num_tilings):           #for each tiling we are going to partition the state space along each dimension

                tiling = []

                for feature in range(len(state_low)): 
                           
                    bin = self.num_bins[feature]

                    offset = self.tiling_offset[0][feature]
                
                    low = state_low[feature]
                
                
                    high = state_high[feature] + tile_width[feature]
                
                    tiling.append(np.linspace(low, high, bin+1) - tile*offset)       
                
                self.tilings.append(tiling)
            
        
            
        
            
        

    def feature_vector_len(self) -> int:              
        return self.num_actions*self.num_tilings*self.grid_size         #return the feature vector length 
        

    def __call__(self, s, done, a) -> np.array:
        


        length = self.feature_vector_len()
        bin_id = []
        
        if done:                                    #if done we are at the terminal state and by definition V=0 (x=0)
            return np.zeros(length)
        
        action_range = a*self.num_tilings           #because of the way I defined the tile coding, each action represents a partition of the tiling

        
        end_range = (a+1)*self.num_tilings
        
        for tiling in self.tilings[action_range:end_range]:                            
            _id = []

            
            for i in range(self.num_dims):
                f_i = s[i]
                
                t_i = tiling[i]
                
                bin_ = np.digitize(f_i, t_i)
                _id.append(bin_)
            
            bin_id.append(_id)
        
        
        x = np.zeros(length)

        index = [0]*len(bin_id)



        for i in range(len(bin_id)):                                    #find the bins that the current state resides in for each tiling        
            index[i] = a*self.grid_size*self.num_tilings + int(bin_id[i][0]-1 + self.num_bins[1]*(bin_id[i][1]-1) + i*(self.grid_size - 1))
        
        
        for i in index:
            x[i] = 1
        
        return x



def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):

        nA = env.action_space.n
        
        Q = [np.dot(w, X.__call__(s,done,a)) for a in range(nA)]
        

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    nA = env.action_space.n

    switch_to_greedy = np.ceiling(num_episode/4)

    

    

    
    for epi in range(num_episode):
        state = env.reset()
        done  = False
        
        action = epsilon_greedy_policy(state,done, w, 0)

        

        states = [state]
        actions = [action]

        x = X.__call__(state, done, action)

        
        
        reward = 0 

        z = np.zeros(X.feature_vector_len())
        
        Q_old = np.dot(w, x)
        
        delta = 0.0
        

        for i in range(1000):

            next_state, reward, done, info = env.step(action)     
            
            if done: break
            
            if epi < switch_to_greedy:
                action = epsilon_greedy_policy(next_state, done, w, (switch_to_greedy - epi)/(switch_to_greedy*10))
            else:
                action = epsilon_greedy_policy(next_state, done, w, 0)

            x_prime = X.__call__(next_state, done, action)

            Q = np.dot(w, x_prime)
            
            


            delta = reward + gamma*Q - Q_old
            

            z = lam*gamma*z + (1-alpha*lam*gamma*np.dot(z,x))*x
            

            w = w + alpha*(delta + Q - Q_old)*z - alpha*(Q- Q_old)*x

            Q_old = Q
            
            x = x_prime

        
        
    
    

            
    
    return w

        



