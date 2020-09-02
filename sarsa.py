import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        
        
        self.num_bins = [int(np.ceil((state_high[i] - state_low[i])/tile_width[i])+1) for i in range(len(state_high))]  
        
        
        self.num_actions = num_actions
        self.num_tilings = num_tilings

        self.num_dims = len(state_low)
        
        self.grid_size = 1

        for i in range(len(self.num_bins)):
            self.grid_size *= self.num_bins[i]
        
        
        
        


        self.tilings = []
        
        self.tiling_offset = [tile_width/num_tilings]                                  
        #maybe here we have an outer loop for the number of actions
        for action in range(num_actions): 

            for tile in range(num_tilings):           #for each tiling

                tiling = []

                for feature in range(len(state_low)): 
                           
                    bin = self.num_bins[feature]

                    offset = self.tiling_offset[0][feature]
                
                    low = state_low[feature]
                
                
                    high = state_high[feature] + tile_width[feature]
                
                    tiling.append(np.linspace(low, high, bin+1) - tile*offset)       
                
                self.tilings.append(tiling)
            
        
            
        
            
        

    def feature_vector_len(self) -> int:                      #this works
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method

        
        return self.num_actions*self.num_tilings*self.grid_size
        raise NotImplementedError()

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        really this is the feature vector for x(s,a) which should be d-dimensional in length
        if done is True, then return 0^d
        """
        #so the implementation can be something like the way we did the bin_id in previous assignment
        length = self.feature_vector_len()
        bin_id = []

        #print("we are in call and state is", s, " and action is ", a)
        
        if done:
            return np.zeros(length)
        
        action_range = a*self.num_tilings

        #print("action range is: ", action_range)
        
        end_range = (a+1)*self.num_tilings
        
        #print("end range is ", end_range)
        
        for tiling in self.tilings[action_range:end_range]:                            #fix this 
            _id = []

            #print("tiling is ", tiling)
            
            for i in range(self.num_dims):
                f_i = s[i]
                #print("f_i is ", f_i)
                t_i = tiling[i]
                #print("t_i is ", t_i)
                bin_ = np.digitize(f_i, t_i)
                _id.append(bin_)
            
            bin_id.append(_id)
        #print("bin_id \n", bin_id)
        
        x = np.zeros(length)

        index = [0]*len(bin_id)



        for i in range(len(bin_id)):                                            
            index[i] = a*self.grid_size*self.num_tilings + int(bin_id[i][0]-1 + self.num_bins[1]*(bin_id[i][1]-1) + i*(self.grid_size - 1))
        
        
        for i in index:
            x[i] = 1
        
        return x


        

        # TODO: implement this method
        raise NotImplementedError()

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

        








    #TODO: implement this function
    raise NotImplementedError()
